"""
Cross-session (user-level) features derived from session history.
These form the user-profile input to the LightGBM churn risk model,
combined with the LSTM's p_frustrated output.

Features capture whether today's frustration is an outlier or part of
a trend — a single bad experience behaves differently to a third in a row.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def extract_cross_session_features(
    user_df: pd.DataFrame,
    session_df: pd.DataFrame,
    lookback_days: int = 30,
) -> pd.DataFrame:
    """
    Compute cross-session features per user from their session history.

    Parameters
    ----------
    user_df
        user_profiles.parquet — one row per user, with LTV, segment, etc.
    session_df
        sessions.parquet — must include 'session_date' column (date).
    lookback_days
        Rolling window in days for rate-based features.

    Returns
    -------
    DataFrame indexed by user_id with cross-session feature columns.
    """
    if "session_date" not in session_df.columns:
        raise ValueError("session_df must contain a 'session_date' date column.")

    agg = (
        session_df.groupby("user_id")
        .agg(
            total_sessions=("session_id", "count"),
            frustrated_sessions=("is_frustrated", "sum"),
            support_contacts=("support_contact_made", "sum"),
            cancelled_sessions=("completed", lambda x: (~x).sum()),
            avg_delay_minutes=("delay_minutes", "mean"),
            p90_delay_minutes=("delay_minutes", lambda x: x.quantile(0.9)),
            last_session_eta=("base_eta_minutes", "last"),
        )
        .reset_index()
    )

    agg["frustration_rate"] = agg["frustrated_sessions"] / agg["total_sessions"].clip(lower=1)
    agg["support_contact_rate"] = agg["support_contacts"] / agg["total_sessions"].clip(lower=1)
    agg["cancellation_rate"] = agg["cancelled_sessions"] / agg["total_sessions"].clip(lower=1)

    # Join with user profile features
    profile_cols = [
        "user_id", "tenure_days", "lifetime_order_count",
        "avg_order_value_myr", "ltv_estimate_myr", "churn_sensitivity",
        "is_post_complaint_return", "prior_delays_30d", "segment",
    ]
    merged = agg.merge(user_df[profile_cols], on="user_id", how="left")

    # Interaction feature: LTV × frustration_rate (high LTV + high frustration = priority)
    merged["ltv_x_frustration_rate"] = (
        merged["ltv_estimate_myr"] * merged["frustration_rate"]
    )

    # Repeated frustration indicator (at-risk pattern)
    merged["is_repeat_frustrated"] = (
        (merged["frustrated_sessions"] >= 2) & (merged["total_sessions"] >= 3)
    ).astype(int)

    return merged.set_index("user_id")
