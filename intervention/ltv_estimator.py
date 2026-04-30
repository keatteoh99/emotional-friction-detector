"""
LTV estimator for gating intervention type.

In production this would be a dedicated ML model or a lookup from a data warehouse.
Here we implement a simple analytical estimator:
  LTV_12m = monthly_order_rate × 12 × AOV × platform_margin

The estimator also provides an intervention ROI check:
  expected_retention_value = LTV × p_churn_reduction
  voucher_cost = 2.0 MYR
  ROI = expected_retention_value / voucher_cost
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LTVEstimate:
    user_id: str
    ltv_12m_myr: float
    monthly_order_rate: float
    avg_order_value_myr: float
    segment: str
    voucher_roi: float   # at assumed 15% churn reduction and RM2 cost


class LTVEstimator:
    PLATFORM_MARGIN = 0.11     # 11% net margin after rider + venue cut
    VOUCHER_COST_MYR = 2.0
    ASSUMED_CHURN_REDUCTION = 0.15  # baseline lift assumption for ROI calc

    def estimate(
        self,
        user_id: str,
        tenure_days: int,
        lifetime_order_count: int,
        avg_order_value_myr: float,
        p_churn: float = 0.5,
    ) -> LTVEstimate:
        monthly_order_rate = lifetime_order_count / max(1, tenure_days / 30.0)
        monthly_order_rate = float(np.clip(monthly_order_rate, 0.1, 30.0))

        ltv_12m = monthly_order_rate * 12 * avg_order_value_myr * self.PLATFORM_MARGIN

        if ltv_12m >= 200:
            segment = "high_value"
        elif ltv_12m >= 60:
            segment = "regular"
        else:
            segment = "at_risk"

        # ROI: expected retained LTV vs voucher cost
        churn_reduction_myr = ltv_12m * p_churn * self.ASSUMED_CHURN_REDUCTION
        voucher_roi = churn_reduction_myr / self.VOUCHER_COST_MYR if self.VOUCHER_COST_MYR > 0 else 0.0

        return LTVEstimate(
            user_id=user_id,
            ltv_12m_myr=round(ltv_12m, 2),
            monthly_order_rate=round(monthly_order_rate, 2),
            avg_order_value_myr=round(avg_order_value_myr, 2),
            segment=segment,
            voucher_roi=round(voucher_roi, 2),
        )

    def is_voucher_roi_positive(self, estimate: LTVEstimate, min_roi: float = 2.0) -> bool:
        return estimate.voucher_roi >= min_roi
