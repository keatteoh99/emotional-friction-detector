"""
Synthetic user profile generator for food delivery platform simulation.

Demographics follow a Pareto-like distribution: ~30% of users (high_value segment)
generate ~70% of platform LTV. Post-complaint return (PCR) users are a small but
high-churn-risk cohort (~5% of base) who returned after lodging a support complaint.
City mix is weighted toward KL (50%) with Penang/JB/Kuching making up the rest.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


CITIES: dict[str, dict] = {
    "KL":      {"rain_prob": 0.35, "fringe_frac": 0.15},
    "Penang":  {"rain_prob": 0.45, "fringe_frac": 0.25},
    "JB":      {"rain_prob": 0.40, "fringe_frac": 0.30},
    "Kuching": {"rain_prob": 0.55, "fringe_frac": 0.35},
}

CUISINES = ["local_malay", "chinese", "indian", "western", "japanese", "korean"]


@dataclass
class UserProfile:
    user_id: str
    tenure_days: int
    lifetime_order_count: int
    avg_order_value_myr: float
    ltv_estimate_myr: float          # 12-month projected LTV
    churn_sensitivity: float         # [0,1] latent; higher → churns faster when frustrated
    is_post_complaint_return: bool   # returned after lodging support complaint
    prior_delays_30d: int            # delays experienced in past 30 days
    segment: str                     # "high_value" | "regular" | "at_risk"
    city: str
    zone: str                        # "central" | "suburban" | "fringe"
    preferred_cuisine: str
    rain_sensitivity: float          # [0,1] amplifies frustration during rain


def generate_user_profiles(
    n_users: int,
    rng: np.random.Generator,
) -> List[UserProfile]:
    city_names = list(CITIES.keys())
    city_weights = [0.50, 0.25, 0.15, 0.10]
    profiles: List[UserProfile] = []

    for _ in range(n_users):
        city = str(rng.choice(city_names, p=city_weights))
        city_meta = CITIES[city]

        # Tenure: exponential, clipped to [7, 1800] days
        tenure = int(np.clip(rng.exponential(scale=300), 7, 1800))

        # Order frequency: correlated with tenure, ~4.5 orders/month
        monthly_orders = float(np.clip(rng.normal(4.5, 2.5), 0.5, 30.0))
        lifetime_orders = max(1, int(monthly_orders * tenure / 30))

        # AOV: log-normal centred around RM 25
        aov = float(np.clip(rng.lognormal(np.log(25), 0.45), 8.0, 120.0))

        # 12-month LTV = monthly_orders * 12 * AOV * platform_margin
        margin = float(rng.uniform(0.08, 0.15))
        ltv = round(monthly_orders * 12 * aov * margin, 2)

        # Segment and churn sensitivity
        if ltv > 200:
            segment = "high_value"
            # High-value users are stickier (lower sensitivity)
            churn_sens = float(rng.beta(2, 6))
        elif ltv > 60:
            segment = "regular"
            churn_sens = float(rng.beta(3, 4))
        else:
            segment = "at_risk"
            churn_sens = float(rng.beta(5, 3))

        # Post-complaint return: ~5% of users, elevated churn sensitivity
        is_pcr = rng.random() < 0.05
        if is_pcr:
            churn_sens = min(1.0, churn_sens + float(rng.uniform(0.15, 0.35)))

        # Zone distribution
        fringe = city_meta["fringe_frac"]
        suburban = 0.35
        central = max(0.0, 1.0 - fringe - suburban)
        zone = str(rng.choice(["central", "suburban", "fringe"], p=[central, suburban, fringe]))

        profiles.append(UserProfile(
            user_id=str(uuid.uuid4()),
            tenure_days=tenure,
            lifetime_order_count=lifetime_orders,
            avg_order_value_myr=round(aov, 2),
            ltv_estimate_myr=ltv,
            churn_sensitivity=round(churn_sens, 4),
            is_post_complaint_return=bool(is_pcr),
            prior_delays_30d=int(rng.poisson(lam=1.2)),
            segment=segment,
            city=city,
            zone=zone,
            preferred_cuisine=str(rng.choice(CUISINES)),
            rain_sensitivity=round(float(rng.beta(2, 3)), 4),
        ))

    return profiles


def profiles_to_dataframe(profiles: List[UserProfile]) -> pd.DataFrame:
    return pd.DataFrame([vars(p) for p in profiles])
