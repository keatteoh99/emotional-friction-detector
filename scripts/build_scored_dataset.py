"""
Build sessions_scored.parquet: LSTM scores + joined user/session features.
Used as input for LightGBM churn risk training.

Usage:
    python -m scripts.build_scored_dataset
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.sequence_features import (
    build_feature_matrices_from_df,
    extract_session_aggregate_features,
)
from models.lstm_frustration.predict import batch_score_sessions

WEIGHTS = "models/lstm_frustration/artefacts/model.pt"
SESSIONS = "data/processed/sessions.parquet"
EVENTS   = "data/processed/events.parquet"
USERS    = "data/processed/user_profiles.parquet"
OUTPUT   = "data/processed/sessions_scored.parquet"


def main() -> None:
    print("Loading data...")
    sessions = pd.read_parquet(SESSIONS)
    events   = pd.read_parquet(EVENTS)
    users    = pd.read_parquet(USERS)

    print("Building feature matrices...")
    feat_mats = build_feature_matrices_from_df(events)

    print(f"Scoring {len(sessions):,} sessions with LSTM...")
    scores = batch_score_sessions(feat_mats, WEIGHTS)
    sessions["p_frustrated"] = sessions["session_id"].map(scores).fillna(0.0)

    print("Computing per-session aggregate features...")
    agg_cols = [
        "tap_interval_cv", "bg_fg_cycle_rate_per_min",
        "support_latency_ratio", "anxiety_event_rate_per_min",
    ]
    agg = extract_session_aggregate_features(events)
    sessions = sessions.join(agg[agg_cols], on="session_id", how="left")
    for col in agg_cols:
        sessions[col] = sessions[col].fillna(0.0)

    print("Joining user cross-session features...")
    user_cols = [
        "user_id", "ltv_estimate_myr", "tenure_days", "lifetime_order_count",
        "avg_order_value_myr", "churn_sensitivity", "is_post_complaint_return",
        "prior_delays_30d",
    ]
    sessions = sessions.merge(users[user_cols], on="user_id", how="left")

    print("Computing per-user session history features...")
    user_hist = sessions.groupby("user_id").agg(
        frustration_rate=("is_frustrated", "mean"),
        support_contact_rate=("support_contact_made", "mean"),
        cancellation_rate=("completed", lambda x: 1 - x.mean()),
        avg_delay_minutes=("delay_minutes", "mean"),
        p90_delay_minutes=("delay_minutes", lambda x: float(np.percentile(x, 90))),
        n_frustrated=("is_frustrated", "sum"),
    ).reset_index()
    user_hist["is_repeat_frustrated"] = (user_hist["n_frustrated"] >= 2).astype(int)
    sessions = sessions.merge(
        user_hist.drop(columns="n_frustrated"), on="user_id", how="left"
    )
    sessions["ltv_x_frustration_rate"] = (
        sessions["ltv_estimate_myr"] * sessions["frustration_rate"]
    )

    sessions.to_parquet(OUTPUT, index=False)
    print(f"Saved {len(sessions):,} rows to {OUTPUT}")

    frust = sessions["is_frustrated"]
    p = sessions["p_frustrated"]
    print(f"  p_frustrated overall mean: {p.mean():.3f}")
    print(f"  p_frustrated | frustrated: {p[frust].mean():.3f}")
    print(f"  p_frustrated | not frust.: {p[~frust].mean():.3f}")


if __name__ == "__main__":
    main()
