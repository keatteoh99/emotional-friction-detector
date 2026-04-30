"""
Tests for the intervention decision engine.
Covers combined-score tiers, dormant LTV guard, holdback injection,
and rationale population.
"""
import pytest

from intervention.decision_engine import (
    DecisionEngine,
    InterventionType,
    ScoringResult,
)


def _result(
    p_frustrated: float = 0.90,
    p_churn: float = 0.80,
    ltv: float = 100.0,
) -> ScoringResult:
    # combined = p_frustrated × p_churn = 0.90 × 0.80 = 0.72 (mid-score tier)
    return ScoringResult(
        session_id="s1",
        user_id="u1",
        p_frustrated=p_frustrated,
        p_churn_given_frustrated=p_churn,
        ltv_estimate_myr=ltv,
    )


class TestDecisionEngine:
    engine = DecisionEngine()

    # ── Gate 1: combined score threshold ─────────────────────────────────────

    def test_below_threshold_no_intervention(self):
        # combined = 0.50 × 0.50 = 0.25 < 0.60
        dec = self.engine.decide(_result(0.50, 0.50), _holdback_rand=0.99)
        assert dec.intervention_type == InterventionType.NO_INTERVENTION
        assert not dec.is_holdback

    def test_at_threshold_fires(self):
        # combined = 0.80 × 0.76 ≈ 0.608 ≥ 0.60
        dec = self.engine.decide(_result(0.80, 0.76), _holdback_rand=0.99)
        assert dec.intervention_type != InterventionType.NO_INTERVENTION

    # ── Gate 2: holdback ──────────────────────────────────────────────────────

    def test_holdback_fires_below_rate(self):
        dec = self.engine.decide(_result(), _holdback_rand=0.05)
        assert dec.intervention_type == InterventionType.HOLDBACK
        assert dec.is_holdback

    def test_holdback_does_not_fire_above_rate(self):
        dec = self.engine.decide(_result(), _holdback_rand=0.50)
        assert not dec.is_holdback

    # ── Score tiers ───────────────────────────────────────────────────────────

    def test_mid_score_tier_gets_empathy(self):
        # combined = 0.90 × 0.72 = 0.648 → 0.60–0.74 tier
        dec = self.engine.decide(_result(0.90, 0.72, ltv=100.0), _holdback_rand=0.99)
        assert dec.intervention_type == InterventionType.EMPATHY_MESSAGE

    def test_high_score_tier_gets_voucher(self):
        # combined = 0.95 × 0.82 = 0.779 → 0.75–0.89 tier
        dec = self.engine.decide(_result(0.95, 0.82, ltv=100.0), _holdback_rand=0.99)
        assert dec.intervention_type == InterventionType.VOUCHER_PLUS_EMPATHY

    def test_escalation_tier(self):
        # combined = 0.98 × 0.94 = 0.921 → ≥ 0.90 tier
        dec = self.engine.decide(_result(0.98, 0.94, ltv=100.0), _holdback_rand=0.99)
        assert dec.intervention_type == InterventionType.CS_ESCALATION

    # ── LTV dormant guard ─────────────────────────────────────────────────────

    def test_dormant_user_voucher_downgraded_to_empathy(self):
        # combined in voucher tier, but LTV is below dormant threshold
        dec = self.engine.decide(_result(0.95, 0.82, ltv=20.0), _holdback_rand=0.99)
        assert dec.intervention_type == InterventionType.EMPATHY_MESSAGE

    def test_dormant_user_escalation_still_fires(self):
        # CS escalation is NOT blocked by dormant guard (salvage case)
        dec = self.engine.decide(_result(0.98, 0.94, ltv=10.0), _holdback_rand=0.99)
        assert dec.intervention_type == InterventionType.CS_ESCALATION

    def test_non_dormant_high_score_gets_voucher(self):
        # combined = 0.779, LTV > 30 → should get voucher
        dec = self.engine.decide(_result(0.95, 0.82, ltv=50.0), _holdback_rand=0.99)
        assert dec.intervention_type == InterventionType.VOUCHER_PLUS_EMPATHY

    # ── combined_score field ──────────────────────────────────────────────────

    def test_combined_score_equals_product(self):
        r = _result(0.80, 0.75)
        dec = self.engine.decide(r, _holdback_rand=0.99)
        assert dec.combined_score == pytest.approx(0.80 * 0.75, abs=1e-3)

    # ── Rationale ─────────────────────────────────────────────────────────────

    def test_rationale_populated(self):
        dec = self.engine.decide(_result(), _holdback_rand=0.99)
        assert len(dec.rationale) > 0

    def test_holdback_rationale(self):
        dec = self.engine.decide(_result(), _holdback_rand=0.01)
        assert "holdback" in dec.rationale

    # ── Custom thresholds ─────────────────────────────────────────────────────

    def test_custom_fire_threshold(self):
        engine = DecisionEngine(fire_threshold=0.30)
        # combined = 0.50 × 0.50 = 0.25 — below default 0.60 but above custom 0.30
        # Wait: 0.25 < 0.30 still → no intervention
        dec = engine.decide(_result(0.60, 0.50), _holdback_rand=0.99)
        # combined = 0.30 — exactly at threshold, depends on < vs <=
        # Test a clear case: combined = 0.40 × 0.80 = 0.32 > 0.30 → fires
        dec2 = engine.decide(_result(0.40, 0.80), _holdback_rand=0.99)
        assert dec2.intervention_type != InterventionType.NO_INTERVENTION

    def test_custom_ltv_dormant_threshold(self):
        engine = DecisionEngine(ltv_dormant_threshold_myr=100.0)
        # LTV=80, combined in voucher tier → should downgrade
        dec = engine.decide(_result(0.95, 0.82, ltv=80.0), _holdback_rand=0.99)
        assert dec.intervention_type == InterventionType.EMPATHY_MESSAGE
