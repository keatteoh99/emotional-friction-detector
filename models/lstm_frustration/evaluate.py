"""
Evaluation for FrustrationLSTM: AUC, calibration, feature ablation,
and per-segment precision monitoring (MLflow).

Run after training:
  python -m models.lstm_frustration.evaluate \
      --weights models/lstm_frustration/artefacts/model_best.pt \
      --events  data/raw/events.parquet \
      --sessions data/raw/sessions.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from features.sequence_features import build_feature_matrices_from_df
from .predict import batch_score_sessions


def evaluate(
    weights_path: str = "models/lstm_frustration/artefacts/model_best.pt",
    events_path: str = "data/raw/events.parquet",
    sessions_path: str = "data/raw/sessions.parquet",
    run_name: str = "lstm_frustration_eval",
) -> dict:
    print("Loading data...")
    sessions = pd.read_parquet(sessions_path)
    events = pd.read_parquet(events_path)

    print("Building feature matrices...")
    feature_matrices = build_feature_matrices_from_df(events)

    print("Scoring sessions...")
    scores = batch_score_sessions(feature_matrices, weights_path)

    labels_s = sessions.set_index("session_id")["is_frustrated"]
    eval_df = pd.DataFrame({
        "p_frustrated": scores,
        "is_frustrated": labels_s,
    }).dropna()

    y_true = eval_df["is_frustrated"].astype(int).to_numpy()
    y_prob = eval_df["p_frustrated"].to_numpy()

    # ── Core metrics ──────────────────────────────────────────────────────────
    auroc = roc_auc_score(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)

    # Precision at 60% recall (operational threshold)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    idx = np.argmin(np.abs(recalls - 0.60))
    p_at_60r = float(precisions[idx])
    thresh_60r = float(thresholds[min(idx, len(thresholds) - 1)])

    print(f"\n── LSTM Frustration Detector ───────────────────────────────")
    print(f"  AUROC:                    {auroc:.4f}  (target >0.88)")
    print(f"  Avg Precision:            {avg_precision:.4f}")
    print(f"  Precision @ 60% recall:   {p_at_60r:.4f}  (threshold={thresh_60r:.3f})")

    # ── Per-segment evaluation ────────────────────────────────────────────────
    if "frustration_trigger" in sessions.columns:
        print(f"\n  Per-trigger AUROC:")
        segment_aurocs = {}
        for trigger, grp in sessions.set_index("session_id").join(
            eval_df, how="inner"
        ).groupby("frustration_trigger"):
            if len(grp) < 30:
                continue
            try:
                seg_auroc = roc_auc_score(grp["is_frustrated"], grp["p_frustrated"])
                segment_aurocs[trigger] = seg_auroc
                print(f"    {trigger:25s}  n={len(grp):6,}  AUROC={seg_auroc:.4f}")
            except ValueError:
                pass

    # ── Calibration ───────────────────────────────────────────────────────────
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
    calibration_error = float(np.mean(np.abs(frac_pos - mean_pred)))
    print(f"\n  Mean calibration error:   {calibration_error:.4f}")
    print(f"────────────────────────────────────────────────────────────")

    metrics = {
        "auroc": auroc,
        "avg_precision": avg_precision,
        "precision_at_60_recall": p_at_60r,
        "threshold_at_60_recall": thresh_60r,
        "calibration_error": calibration_error,
    }

    with mlflow.start_run(run_name=run_name):
        mlflow.log_metrics(metrics)
        mlflow.log_param("n_eval_sessions", len(eval_df))

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",  default="models/lstm_frustration/artefacts/model_best.pt")
    parser.add_argument("--events",   default="data/raw/events.parquet")
    parser.add_argument("--sessions", default="data/raw/sessions.parquet")
    args = parser.parse_args()
    evaluate(args.weights, args.events, args.sessions)
