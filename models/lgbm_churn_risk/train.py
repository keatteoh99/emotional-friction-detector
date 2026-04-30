"""
Training script for ChurnRiskModel (LightGBM stage-2).

Requires:
  - LSTM inference scores on training sessions (p_frustrated column)
  - Joined user cross-session features
  - Churn labels (7-day churn proxy: no order within 7 days of frustrated session)

Usage:
  python -m models.lgbm_churn_risk.train \
      --scored data/processed/sessions_scored.parquet \
      --output models/lgbm_churn_risk/artefacts/churn_model.lgb
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from .model import ChurnRiskModel, FEATURE_COLS


def build_churn_labels(session_df: pd.DataFrame, window_days: int = 7) -> pd.Series:
    """
    Proxy churn label: user placed no order within `window_days` after
    a frustrated session. Real platforms would use actual churn events.
    """
    if "session_date" not in session_df.columns:
        # Simulation fallback: proxy label built from delay severity + user risk factors.
        # Deliberately does NOT use churn_sensitivity so that feature remains predictive
        # without being an identity leak. Mirrors real-world churn drivers:
        #   - severe delays push borderline users out
        #   - post-complaint returners have lower patience threshold
        #   - high-LTV users are more invested and churn slightly less
        import numpy as np
        rng = np.random.default_rng(42)
        s = session_df
        base_prob = 0.12
        # Delay contribution: 0 at 5 min, caps at 0.25 at 30 min
        delay = s["delay_minutes"] if "delay_minutes" in s.columns else pd.Series(5.0, index=s.index)
        delay_contrib = np.clip((delay - 5.0) / 25.0, 0.0, 0.25)
        # Post-complaint users: +15pp churn risk
        pcr = s["is_post_complaint_return"].astype(float) if "is_post_complaint_return" in s.columns \
            else pd.Series(0.0, index=s.index)
        pcr_contrib = pcr * 0.15
        # High-LTV users: -4pp churn (they're more invested in the platform)
        ltv = s["ltv_estimate_myr"] if "ltv_estimate_myr" in s.columns else pd.Series(100.0, index=s.index)
        ltv_contrib = np.where(ltv > 150, -0.04, 0.04)
        # Worst-trigger sessions carry higher churn risk
        trigger = s["frustration_trigger"] if "frustration_trigger" in s.columns \
            else pd.Series("", index=s.index)
        trigger_contrib = (trigger == "delay_gt_8").astype(float) * 0.08
        # Behavioural intensity: frantic tappers (high CV) are more agitated than
        # resigned waiters with the same delay — +10pp at CV=3, +0pp at CV<=1.0
        tap_cv = s["tap_interval_cv"] if "tap_interval_cv" in s.columns \
            else pd.Series(0.0, index=s.index)
        tap_contrib = np.clip((tap_cv - 1.0) / 20.0, 0.0, 0.10)
        probs = np.clip(base_prob + delay_contrib + pcr_contrib + ltv_contrib + trigger_contrib + tap_contrib, 0.02, 0.70)
        return pd.Series(
            (rng.random(len(s)) < probs).astype(int),
            index=s.index,
            name="churned_7d",
        )

    session_df = session_df.sort_values("session_date")
    next_order = (
        session_df.groupby("user_id")["session_date"]
        .shift(-1)
    )
    days_to_next = (next_order - session_df["session_date"]).dt.days
    return (days_to_next > window_days).fillna(True).astype(int).rename("churned_7d")


def train(
    scored_path: str = "data/processed/sessions_scored.parquet",
    output_dir: str = "models/lgbm_churn_risk/artefacts",
    seed: int = 42,
) -> None:
    df = pd.read_parquet(scored_path)

    # Only train on frustrated sessions (stage-2 model)
    frustrated = df[df["is_frustrated"]].copy()
    if "churned_7d" not in frustrated.columns:
        frustrated["churned_7d"] = build_churn_labels(frustrated)

    available_features = [f for f in FEATURE_COLS if f in frustrated.columns]
    if len(available_features) < len(FEATURE_COLS):
        missing = set(FEATURE_COLS) - set(available_features)
        print(f"Warning: missing features {missing} — will be zero-filled")
        for col in missing:
            frustrated[col] = 0.0

    X = frustrated[FEATURE_COLS]
    y = frustrated["churned_7d"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=seed, stratify=y
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="lgbm_churn_risk"):
        model = ChurnRiskModel()
        model.fit(X_train, y_train, X_val, y_val)

        val_probs = model.predict_proba(X_val)
        auroc = roc_auc_score(y_val, val_probs)
        avg_precision = average_precision_score(y_val, val_probs)

        print(f"\nValidation AUROC:            {auroc:.4f}")
        print(f"Validation Avg Precision:    {avg_precision:.4f}")
        print(f"\nTop-10 feature importance:\n{model.feature_importance_.head(10).to_string()}")

        mlflow.log_metrics({"val_auroc": auroc, "val_avg_precision": avg_precision})
        mlflow.log_param("churn_rate", float(y.mean()))

        model_path = str(out / "churn_model.lgb")
        model.save(model_path)
        mlflow.lightgbm.log_model(model.booster, "lgbm_churn_model")
        print(f"\nSaved model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", default="data/processed/sessions_scored.parquet")
    parser.add_argument("--output", default="models/lgbm_churn_risk/artefacts")
    args = parser.parse_args()
    train(scored_path=args.scored, output_dir=args.output)
