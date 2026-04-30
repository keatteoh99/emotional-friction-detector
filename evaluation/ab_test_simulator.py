"""
A/B test simulator for the intervention holdback experiment.

Design:
  - Unit of randomisation: session (not user) — holdback is at scoring layer
  - Treatment: intervention fires (voucher / empathy / CS escalation)
  - Control:   holdback — scored but no intervention
  - Primary metric: 7-day churn rate
  - Secondary metrics: order frequency 7d post-session, revenue per user

The simulator generates synthetic post-session outcomes based on assumed
lift parameters so the intervention matrix notebook can be explored before
real outcomes are observed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from intervention.decision_engine import InterventionType


@dataclass
class InterventionEffect:
    """Assumed churn reduction (relative) and revenue lift for each action."""
    churn_reduction_relative: float  # fraction, e.g. 0.23 = 23% fewer churns
    revenue_lift_myr: float          # direct revenue impact per treated user


DEFAULT_EFFECTS: Dict[str, InterventionEffect] = {
    InterventionType.VOUCHER_PLUS_EMPATHY.value: InterventionEffect(0.23, 4.50),
    InterventionType.EMPATHY_MESSAGE.value: InterventionEffect(0.11, 1.20),
    InterventionType.CS_ESCALATION.value: InterventionEffect(0.18, 0.80),
    InterventionType.NO_INTERVENTION.value: InterventionEffect(0.0, 0.0),
    InterventionType.HOLDBACK.value: InterventionEffect(0.0, 0.0),
}


def simulate_outcomes(
    decisions_df: pd.DataFrame,
    base_churn_rate: float = 0.35,
    effects: Dict[str, InterventionEffect] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate post-session churn and revenue outcomes.

    Parameters
    ----------
    decisions_df
        Must contain columns: session_id, user_id, intervention_type, is_holdback,
        p_frustrated, p_churn, ltv_estimate_myr.
    base_churn_rate
        Baseline 7-day churn rate for frustrated users in control group.
    effects
        Per-intervention-type assumed effects.
    seed
        RNG seed for reproducibility.

    Returns
    -------
    decisions_df with added columns: churned_7d, revenue_7d_myr.
    """
    if effects is None:
        effects = DEFAULT_EFFECTS

    rng = np.random.default_rng(seed)
    df = decisions_df.copy()

    churned = np.zeros(len(df), dtype=int)
    revenue = np.zeros(len(df), dtype=float)

    for i, row in df.iterrows():
        effect = effects.get(row["intervention_type"], InterventionEffect(0.0, 0.0))
        adjusted_churn = base_churn_rate * (1 - effect.churn_reduction_relative)
        # Personalise with model's p_churn as scaling factor
        personal_churn = float(np.clip(row["p_churn"] * (adjusted_churn / base_churn_rate), 0, 1))
        did_churn = int(rng.random() < personal_churn)
        churned[i] = did_churn
        if not did_churn:
            revenue[i] = float(rng.lognormal(np.log(25), 0.5)) + effect.revenue_lift_myr

    df["churned_7d"] = churned
    df["revenue_7d_myr"] = np.round(revenue, 2)
    return df


def compute_ab_summary(outcomes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise A/B test results by intervention type vs holdback.
    Returns a DataFrame with one row per intervention type.
    """
    control = outcomes_df[outcomes_df["is_holdback"]]
    control_churn = control["churned_7d"].mean() if len(control) > 0 else np.nan

    rows = []
    for itype, grp in outcomes_df[~outcomes_df["is_holdback"]].groupby("intervention_type"):
        churn_rate = grp["churned_7d"].mean()
        avg_revenue = grp["revenue_7d_myr"].mean()
        lift = (control_churn - churn_rate) / control_churn if control_churn else np.nan
        rows.append({
            "intervention_type": itype,
            "n_sessions": len(grp),
            "churn_rate": round(churn_rate, 4),
            "control_churn_rate": round(control_churn, 4),
            "churn_lift": round(lift, 4),
            "avg_revenue_7d_myr": round(avg_revenue, 2),
        })

    summary = pd.DataFrame(rows).sort_values("churn_lift", ascending=False)
    return summary
