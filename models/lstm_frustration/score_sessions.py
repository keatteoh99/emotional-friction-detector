"""
Build data/processed/sessions_scored.parquet for LightGBM training.

Scores all sessions with the trained LSTM, then joins user cross-session
features and per-session aggregate features into a single flat table.

Usage:
    python -m models.lstm_frustration.score_sessions
    python -m models.lstm_frustration.score_sessions --data data/processed/ --weights models/lstm_frustration/artefacts/model.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features.sequence_features import (
    build_feature_matrices_from_df,
    extract_session_aggregate_features,
)
from .predict import batch_score_sessions


def build_scored_dataset(
    sessions_path: str = "data/processed/sessions.parquet",
    events_path: str = "data/processed/events.parquet",
    users_path: str = "data/processed/user_profiles.parquet",
    weights_path: str = "models/lstm_frustration/artefacts/model.pt",
    output_path: str = "data/processed/sessions_scored.parquet",
) -> pd.DataFrame:
    print("Loading data...")
    sessions = pd.read_parquet(sessions_path)
    events   = pd.read_parquet(events_path)
    users    = pd.read_parquet(users_path)

    print("Building feature matrices...")
    feat_mats = build_feature_matrices_from_df(events)

    print(f"Scoring {len(sessions):,} sessions with LSTM...")
    scores = batch_score_sessions(feat_mats, weights_path)
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

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sessions.to_parquet(output_path, index=False)
    print(f"Saved {len(sessions):,} rows -> {output_path}")

    frust = sessions["is_frustrated"]
    p = sessions["p_frustrated"]
    print(f"  p_frustrated overall:     {p.mean():.3f}")
    print(f"  p_frustrated | frustrated: {p[frust].mean():.3f}")
    print(f"  p_frustrated | calm:       {p[~frust].mean():.3f}")
    return sessions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score all sessions with LSTM and build joined dataset")
    parser.add_argument("--data", default=None, help="Directory with sessions/events/user_profiles parquets")
    parser.add_argument("--sessions", default="data/processed/sessions.parquet")
    parser.add_argument("--events",   default="data/processed/events.parquet")
    parser.add_argument("--users",    default="data/processed/user_profiles.parquet")
    parser.add_argument("--weights",  default="models/lstm_frustration/artefacts/model.pt")
    parser.add_argument("--output",   default="data/processed/sessions_scored.parquet")
    args = parser.parse_args()

    if args.data:
        d = Path(args.data)
        sessions_path = str(d / "sessions.parquet")
        events_path   = str(d / "events.parquet")
        users_path    = str(d / "user_profiles.parquet")
    else:
        sessions_path = args.sessions
        events_path   = args.events
        users_path    = args.users

    build_scored_dataset(
        sessions_path=sessions_path,
        events_path=events_path,
        users_path=users_path,
        weights_path=args.weights,
        output_path=args.output,
    )
