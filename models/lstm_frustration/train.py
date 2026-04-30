"""
Training pipeline for FrustrationLSTM.

Design notes (answers to pre-flight checklist):
  1. pos_weight computed from TRAINING SPLIT labels only, after stratified split.
     Using full-dataset rate would leak val-set distribution into the loss.
  2. train_test_split uses stratify= on the label array so the 60/40 class ratio
     is preserved in both splits - important for stable pos_weight and fair AUC.
  3. Per-segment AUC is logged at the END of each epoch for three cohorts:
       is_peak        - lunch/dinner peak hours (higher frustration baseline)
       is_rain        - sessions during rain (confound for delay model)
       is_post_complaint - post-complaint return users (hair-trigger cohort)
     These are the hardest cohorts; overall AUC masking poor segment performance
     is a common portfolio mistake.
  4. Best checkpoint written to models/lstm_frustration/checkpoints/best_<run_id[:8]>.pt
     so you can recover the best weights even if the final epoch is worse.

Usage:
  python -m models.lstm_frustration.train
  python -m models.lstm_frustration.train --epochs 20 --seed 123
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from features.sequence_features import build_feature_matrices_from_df
from .model import FrustrationLSTM, LSTMConfig


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FrustrationDataset(Dataset):
    def __init__(
        self,
        session_ids: List[str],
        feature_matrices: Dict[str, np.ndarray],
        labels: Dict[str, int],
        max_seq_len: int = 64,
    ):
        self.session_ids = session_ids
        self.feature_matrices = feature_matrices
        self.labels = labels
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.session_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sid = self.session_ids[idx]
        mat = self.feature_matrices.get(sid, np.zeros((1, 9), dtype=np.float32))
        label = self.labels[sid]

        T = min(len(mat), self.max_seq_len)
        padded = np.zeros((self.max_seq_len, 9), dtype=np.float32)
        padded[:T] = mat[:T]

        return (
            torch.tensor(padded, dtype=torch.float32),
            torch.tensor(T, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Segment AUC helper
# ---------------------------------------------------------------------------

def _compute_segment_aurocs(
    val_ids: List[str],
    probs: List[float],
    labels: List[int],
    seg_meta: pd.DataFrame,
    min_positives: int = 30,
) -> Dict[str, float]:
    """
    Compute AUC for each boolean segment column in seg_meta.
    val_ids order must match probs/labels order (DataLoader shuffle=False).
    Skips segments with fewer than min_positives positive examples.
    """
    val_df = pd.DataFrame({
        "session_id": val_ids,
        "prob": probs,
        "label": labels,
    }).set_index("session_id").join(seg_meta, how="left")

    results = {}
    for col in seg_meta.columns:
        if col not in val_df.columns:
            continue
        grp = val_df[val_df[col].fillna(False)]
        n_pos = int(grp["label"].sum())
        if len(grp) < 50 or n_pos < min_positives or grp["label"].nunique() < 2:
            continue
        results[col] = float(roc_auc_score(grp["label"], grp["prob"]))
    return results


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    events_path: str = "data/raw/events.parquet",
    sessions_path: str = "data/raw/sessions.parquet",
    users_path: str = "data/raw/user_profiles.parquet",
    checkpoint_dir: str = "models/lstm_frustration/checkpoints",
    artefacts_dir: str = "models/lstm_frustration/artefacts",
    epochs: int = 15,
    batch_size: int = 512,
    lr: float = 3e-4,
    seed: int = 42,
) -> str:
    """
    Train FrustrationLSTM. Returns the MLflow run_id.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -- Load data -------------------------------------------------------------
    print("Loading sessions, events, user profiles...")
    sessions = pd.read_parquet(sessions_path)
    events = pd.read_parquet(events_path)

    # Segment metadata for per-cohort AUC (joined at eval time, not train time)
    users = pd.read_parquet(users_path)[["user_id", "is_post_complaint_return"]]
    sessions = sessions.merge(users, on="user_id", how="left")
    sessions["is_peak"] = sessions["hour_of_day"].isin(
        list(range(11, 14)) + list(range(18, 22))
    )
    # Normalise column name: parquet may store as 'is_raining' bool
    sessions["is_rain"] = sessions["is_raining"].astype(bool)

    seg_meta = sessions.set_index("session_id")[
        ["is_peak", "is_rain", "is_post_complaint_return"]
    ].rename(columns={"is_post_complaint_return": "is_post_complaint"})

    labels = dict(zip(sessions["session_id"], sessions["is_frustrated"].astype(int)))

    # -- Build per-event feature matrices --------------------------------------
    print(f"Building (T, 9) feature matrices for {len(labels):,} sessions...")
    feature_matrices = build_feature_matrices_from_df(events)
    seq_lens = [m.shape[0] for m in feature_matrices.values()]
    print(f"  Mean seq len: {np.mean(seq_lens):.1f}  |  p95: {np.percentile(seq_lens, 95):.0f}")

    # -- Stratified train / val split ------------------------------------------
    # stratify= preserves the ~60/40 class ratio in both splits.
    all_ids = list(labels.keys())
    all_label_vals = [labels[sid] for sid in all_ids]

    train_ids, val_ids = train_test_split(
        all_ids,
        test_size=0.15,
        random_state=seed,
        stratify=all_label_vals,   # <- requirement 2
    )

    # -- pos_weight from TRAINING SPLIT ONLY -----------------------------------
    # Using full-dataset rate would leak val distribution into loss weighting.
    train_label_arr = np.array([labels[sid] for sid in train_ids])
    pos_rate_train = train_label_arr.mean()
    pos_weight_val = (1.0 - pos_rate_train) / pos_rate_train
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    print(f"  Train pos rate: {pos_rate_train:.3f}  =>  pos_weight: {pos_weight_val:.3f}")

    # -- Datasets & DataLoaders ------------------------------------------------
    cfg = LSTMConfig()
    train_ds = FrustrationDataset(train_ids, feature_matrices, labels, cfg.max_seq_len)
    val_ds = FrustrationDataset(val_ids, feature_matrices, labels, cfg.max_seq_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    # shuffle=False required: val_ids order must match collected probs for segment AUC
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # -- Model, optimiser, scheduler -------------------------------------------
    model = FrustrationLSTM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_dl), epochs=epochs
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ckpt_dir = Path(checkpoint_dir)
    art_dir = Path(artefacts_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    # -- Training loop ---------------------------------------------------------
    best_auroc = 0.0
    best_ckpt_path: Optional[Path] = None

    with mlflow.start_run(run_name="lstm_frustration") as run:
        run_id = run.info.run_id
        print(f"MLflow run_id: {run_id}")

        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "input_size": cfg.input_size,
            "seed": seed,
            "train_size": len(train_ids),
            "val_size": len(val_ids),
            "pos_rate_train": round(float(pos_rate_train), 4),
            "pos_weight": round(float(pos_weight_val), 4),
        })

        for epoch in range(1, epochs + 1):
            # -- Train ---------------------------------------------------------
            model.train()
            train_loss = 0.0
            for features, lengths, targets in tqdm(
                train_dl, desc=f"Ep {epoch:02d}", leave=False, ncols=80
            ):
                features = features.to(device)
                lengths = lengths.to(device)
                targets = targets.to(device)

                logits = model(features, lengths)
                loss = criterion(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()

            avg_train = train_loss / len(train_dl)

            # -- Validate ------------------------------------------------------
            model.eval()
            val_loss = 0.0
            all_probs: List[float] = []
            all_labels: List[int] = []

            with torch.no_grad():
                for features, lengths, targets in val_dl:
                    features = features.to(device)
                    lengths = lengths.to(device)
                    targets = targets.to(device)
                    logits = model(features, lengths)
                    val_loss += criterion(logits, targets).item()
                    all_probs.extend(torch.sigmoid(logits).cpu().tolist())
                    all_labels.extend(targets.cpu().int().tolist())

            avg_val = val_loss / len(val_dl)
            auroc = roc_auc_score(all_labels, all_probs)

            # F1 at 0.5 threshold (useful gut-check alongside AUC)
            preds_05 = [int(p >= 0.5) for p in all_probs]
            f1 = f1_score(all_labels, preds_05, zero_division=0)

            # -- Per-segment AUC -----------------------------------------------
            # val_ids order matches all_probs/all_labels because shuffle=False
            seg_aurocs = _compute_segment_aurocs(
                val_ids, all_probs, all_labels, seg_meta
            )

            # -- Log to MLflow -------------------------------------------------
            metrics: Dict[str, float] = {
                "train_loss": avg_train,
                "val_loss": avg_val,
                "val_auroc": auroc,
                "val_f1": f1,
            }
            for seg_name, seg_auc in seg_aurocs.items():
                metrics[f"val_auroc_{seg_name}"] = seg_auc

            mlflow.log_metrics(metrics, step=epoch)

            # -- Console summary -----------------------------------------------
            seg_str = "  ".join(
                f"{k.replace('is_', '')}={v:.3f}" for k, v in seg_aurocs.items()
            )
            print(
                f"  Ep {epoch:02d}"
                f"  train={avg_train:.4f}"
                f"  val={avg_val:.4f}"
                f"  AUC={auroc:.4f}"
                f"  F1={f1:.3f}"
                + (f"  [{seg_str}]" if seg_str else "")
            )

            # -- Checkpoint best model -----------------------------------------
            if auroc > best_auroc:
                best_auroc = auroc
                best_ckpt_path = ckpt_dir / f"best_{run_id[:8]}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "auroc": auroc,
                        "model_state_dict": model.state_dict(),
                        "cfg": cfg,
                        "run_id": run_id,
                    },
                    best_ckpt_path,
                )

        # -- Final artefact ----------------------------------------------------
        final_path = art_dir / "model.pt"
        torch.save(model.state_dict(), final_path)
        mlflow.log_artifact(str(best_ckpt_path), "checkpoints")
        mlflow.pytorch.log_model(model, "lstm_frustration_model")
        mlflow.log_metrics({"best_val_auroc": best_auroc})

        print(f"\n-- Training complete ----------------------------------------")
        print(f"  Best val AUROC : {best_auroc:.4f}  (target >0.88)")
        print(f"  Best checkpoint: {best_ckpt_path}")
        print(f"  Final weights  : {final_path}")
        print(f"  MLflow run_id  : {run_id}")
        print(f"------------------------------------------------------------")

    return run_id


def _resolve_data_paths(data_arg: str) -> tuple[str, str, str]:
    """
    Accept either a directory or a sessions.parquet file path.
    Both resolve to the same three files in the same directory:
      --data data/processed/
      --data data/processed/sessions.parquet
    Returns (sessions_path, events_path, users_path).
    """
    p = Path(data_arg)
    data_dir = p.parent if p.suffix == ".parquet" else p
    return (
        str(data_dir / "sessions.parquet"),
        str(data_dir / "events.parquet"),
        str(data_dir / "user_profiles.parquet"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train FrustrationLSTM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --data is the convenient one-flag interface; individual flags override it
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Directory or sessions.parquet path containing sessions.parquet, "
            "events.parquet, and user_profiles.parquet. "
            "Overrides --events / --sessions / --users when provided."
        ),
    )
    parser.add_argument("--events",   default="data/raw/events.parquet")
    parser.add_argument("--sessions", default="data/raw/sessions.parquet")
    parser.add_argument("--users",    default="data/raw/user_profiles.parquet")
    parser.add_argument(
        "--mlflow-experiment",
        default="friction-detector",
        metavar="NAME",
        help="MLflow experiment name (created if it does not exist)",
    )
    parser.add_argument("--checkpoint-dir", default="models/lstm_frustration/checkpoints")
    parser.add_argument("--artefacts-dir",  default="models/lstm_frustration/artefacts")
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--batch-size", type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    # --data overrides individual path flags
    if args.data is not None:
        sessions_path, events_path, users_path = _resolve_data_paths(args.data)
    else:
        sessions_path = args.sessions
        events_path = args.events
        users_path = args.users

    # Validate that all three files exist before kicking off training
    missing = [p for p in (sessions_path, events_path, users_path) if not Path(p).exists()]
    if missing:
        print("ERROR: required data files not found:")
        for p in missing:
            print(f"  {p}")
        print("\nRun the data generator first:")
        data_dir = Path(sessions_path).parent
        print(f"  python -m simulator.generate_dataset --output {data_dir}/")
        raise SystemExit(1)

    mlflow.set_experiment(args.mlflow_experiment)

    train(
        events_path=events_path,
        sessions_path=sessions_path,
        users_path=users_path,
        checkpoint_dir=args.checkpoint_dir,
        artefacts_dir=args.artefacts_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
