"""
Production drift monitor: detects distribution shift in input features
and model output scores.

Checks:
  1. PSI (Population Stability Index) on key features — PSI > 0.2 = alert
  2. p_frustrated score distribution shift (weekly baseline vs current)
  3. Frustration rate shift (could indicate simulation/real-world divergence)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-9,
) -> float:
    """
    Population Stability Index.
    PSI < 0.1  → no shift
    PSI < 0.2  → slight shift
    PSI ≥ 0.2  → significant shift — investigate
    """
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    bins = np.linspace(min_val, max_val, n_bins + 1)
    bins[0] -= 1e-6
    bins[-1] += 1e-6

    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)

    exp_pct = (exp_counts + epsilon) / len(expected)
    act_pct = (act_counts + epsilon) / len(actual)

    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return psi


def monitor_feature_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Compute PSI for each feature between baseline and current windows.
    Returns a DataFrame sorted by PSI descending.
    """
    rows = []
    for col in feature_cols:
        if col not in baseline_df.columns or col not in current_df.columns:
            continue
        psi = compute_psi(
            baseline_df[col].dropna().to_numpy(dtype=float),
            current_df[col].dropna().to_numpy(dtype=float),
        )
        rows.append({
            "feature": col,
            "psi": round(psi, 4),
            "alert": psi >= 0.20,
            "baseline_mean": round(float(baseline_df[col].mean()), 4),
            "current_mean": round(float(current_df[col].mean()), 4),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def monitor_score_drift(
    baseline_scores: np.ndarray,
    current_scores: np.ndarray,
) -> dict:
    """Quick summary of p_frustrated score distribution shift."""
    psi = compute_psi(baseline_scores, current_scores)
    return {
        "psi": round(psi, 4),
        "alert": psi >= 0.20,
        "baseline_mean": round(float(baseline_scores.mean()), 4),
        "current_mean": round(float(current_scores.mean()), 4),
        "baseline_p95": round(float(np.percentile(baseline_scores, 95)), 4),
        "current_p95": round(float(np.percentile(current_scores, 95)), 4),
    }
