"""
LTV-gated, score-tiered intervention decision engine.

Firing condition (from project spec):
  P(frustrated) × P(churn|frustrated) × LTV_30d > intervention_cost

Operational tiers (simplified from the expected-value framework):
  combined_score = p_frustrated × p_churn_given_frustrated

  combined < 0.60                → no intervention
  combined 0.60–0.74             → empathy message
  combined 0.75–0.89             → RM2 voucher + empathy
  combined ≥ 0.90                → CS escalation
  LTV below dormancy threshold   → voucher downgraded to empathy (ROI negative)

A/B holdback: 20% of sessions are scored but not acted on.
Applied at the SCORING layer (per-session, not per-user) so holdback and
treatment users share the same scoring latency — delivery timing cannot
confound A/B results.

Note on combined_score: multiplying two probabilities compresses the range
(e.g., p_f=0.85, p_c=0.80 → combined=0.68). Tier boundaries are calibrated
to this compressed range, not to raw p_frustrated alone.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class InterventionType(str, Enum):
    VOUCHER_PLUS_EMPATHY = "voucher_rm2_plus_empathy"
    EMPATHY_MESSAGE = "empathy_message"
    CS_ESCALATION = "cs_escalation"
    NO_INTERVENTION = "no_intervention"
    HOLDBACK = "holdback"  # scored but no action — A/B control group


@dataclass
class ScoringResult:
    session_id: str
    user_id: str
    p_frustrated: float
    p_churn_given_frustrated: float
    ltv_estimate_myr: float
    experiment_id: Optional[str] = None


@dataclass
class InterventionDecision:
    session_id: str
    user_id: str
    intervention_type: InterventionType
    p_frustrated: float
    p_churn: float
    combined_score: float
    is_holdback: bool
    rationale: str


class DecisionEngine:
    """
    Parameters
    ----------
    fire_threshold
        Minimum combined score to trigger any intervention. Below this → no action.
    voucher_threshold
        combined_score ≥ this → RM2 voucher tier.
    escalation_threshold
        combined_score ≥ this → CS escalation.
    ltv_dormant_threshold_myr
        Users with LTV below this receive at most empathy (no vouchers).
    holdback_rate
        Fraction of qualifying sessions held back as A/B control.
    """

    def __init__(
        self,
        fire_threshold: float = 0.60,
        voucher_threshold: float = 0.75,
        escalation_threshold: float = 0.90,
        ltv_dormant_threshold_myr: float = 30.0,
        holdback_rate: float = 0.20,
    ):
        self.fire_threshold = fire_threshold
        self.voucher_threshold = voucher_threshold
        self.escalation_threshold = escalation_threshold
        self.ltv_dormant = ltv_dormant_threshold_myr
        self.holdback_rate = holdback_rate

    def decide(
        self,
        result: ScoringResult,
        _holdback_rand: Optional[float] = None,   # injectable for deterministic tests
    ) -> InterventionDecision:
        combined = result.p_frustrated * result.p_churn_given_frustrated

        # Fields shared across all branches; is_holdback set per-branch
        base = dict(
            session_id=result.session_id,
            user_id=result.user_id,
            p_frustrated=result.p_frustrated,
            p_churn=result.p_churn_given_frustrated,
            combined_score=round(combined, 4),
        )

        # Gate 1: combined score below fire threshold
        if combined < self.fire_threshold:
            return InterventionDecision(
                **base, is_holdback=False,
                intervention_type=InterventionType.NO_INTERVENTION,
                rationale=f"combined_{combined:.3f}_lt_{self.fire_threshold}",
            )

        # Gate 2: A/B holdback (at scoring layer, not user layer)
        rand = _holdback_rand if _holdback_rand is not None else random.random()
        if rand < self.holdback_rate:
            return InterventionDecision(
                **base, is_holdback=True,
                intervention_type=InterventionType.HOLDBACK,
                rationale="ab_holdback_control",
            )

        # Gate 3: dormant user — voucher ROI is negative, cap at empathy
        is_dormant = result.ltv_estimate_myr < self.ltv_dormant

        # Tier selection
        if combined >= self.escalation_threshold:
            itype = InterventionType.CS_ESCALATION
            rationale = f"score_{combined:.3f}_gte_{self.escalation_threshold}_escalate"
        elif combined >= self.voucher_threshold and not is_dormant:
            itype = InterventionType.VOUCHER_PLUS_EMPATHY
            rationale = f"score_{combined:.3f}_voucher_ltv_{result.ltv_estimate_myr:.0f}"
        else:
            # 0.60–0.74 range, OR voucher tier but dormant user
            itype = InterventionType.EMPATHY_MESSAGE
            reason = "dormant_ltv" if is_dormant and combined >= self.voucher_threshold else "mid_score"
            rationale = f"score_{combined:.3f}_{reason}"

        return InterventionDecision(**base, is_holdback=False, intervention_type=itype, rationale=rationale)
