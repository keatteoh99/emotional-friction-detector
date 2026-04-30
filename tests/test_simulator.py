"""Unit tests for the simulator module."""
import numpy as np
import pytest

from simulator.delay_distribution import (
    DelayContext,
    RestaurantTier,
    TimeSlot,
    is_peak_slot,
    sample_base_eta,
    sample_delivery_delay,
)
from simulator.session_generator import (
    _frustration_label,
    generate_session,
)
from simulator.user_profiles import UserProfile, generate_user_profiles


def _default_ctx(time_slot=TimeSlot.OFF_PEAK, raining=False, tier="B", zone="central"):
    return DelayContext(
        time_slot=time_slot,
        is_raining=raining,
        restaurant_tier=RestaurantTier(tier),
        day_of_week=2,
        zone=zone,
    )


def _make_user(**kwargs) -> UserProfile:
    defaults = dict(
        user_id="u1",
        tenure_days=180,
        lifetime_order_count=50,
        avg_order_value_myr=25.0,
        ltv_estimate_myr=120.0,
        churn_sensitivity=0.4,
        is_post_complaint_return=False,
        prior_delays_30d=0,
        segment="regular",
        city="KL",
        zone="central",
        preferred_cuisine="chinese",
        rain_sensitivity=0.3,
    )
    defaults.update(kwargs)
    return UserProfile(**defaults)


# ---------------------------------------------------------------------------
# delay_distribution
# ---------------------------------------------------------------------------

class TestDelayDistribution:
    def test_peak_delay_higher_than_offpeak(self):
        rng = np.random.default_rng(0)
        ctx_off = _default_ctx(TimeSlot.OFF_PEAK)
        ctx_peak = _default_ctx(TimeSlot.DINNER_PEAK)
        n = 5000
        off_delays = [sample_delivery_delay(ctx_off, rng) for _ in range(n)]
        peak_delays = [sample_delivery_delay(ctx_peak, rng) for _ in range(n)]
        assert np.mean(peak_delays) > np.mean(off_delays) + 1.5  # peak modifier N(3,2) → expected gap ~3

    def test_rain_increases_delay(self):
        rng = np.random.default_rng(1)
        ctx_dry = _default_ctx(raining=False)
        ctx_wet = _default_ctx(raining=True)
        n = 5000
        dry = [sample_delivery_delay(ctx_dry, rng) for _ in range(n)]
        wet = [sample_delivery_delay(ctx_wet, rng) for _ in range(n)]
        assert np.mean(wet) > np.mean(dry) + 1.5  # rain modifier N(3,2.5) → expected gap ~3

    def test_tier_c_delay_higher(self):
        rng = np.random.default_rng(2)
        ctx_a = _default_ctx(tier="A")
        ctx_c = _default_ctx(tier="C")
        n = 5000
        a_delays = [sample_delivery_delay(ctx_a, rng) for _ in range(n)]
        c_delays = [sample_delivery_delay(ctx_c, rng) for _ in range(n)]
        assert np.mean(c_delays) > np.mean(a_delays) + 3.0  # tier-C modifier N(5,3) → expected gap ~5

    def test_base_eta_minimum(self):
        rng = np.random.default_rng(3)
        ctx = _default_ctx()
        for _ in range(100):
            eta = sample_base_eta(ctx, rng)
            assert eta >= 15

    def test_is_peak_slot(self):
        assert is_peak_slot(12) == TimeSlot.LUNCH_PEAK
        assert is_peak_slot(19) == TimeSlot.DINNER_PEAK
        assert is_peak_slot(23) == TimeSlot.LATE_NIGHT
        assert is_peak_slot(10) == TimeSlot.OFF_PEAK


# ---------------------------------------------------------------------------
# frustration_label
# ---------------------------------------------------------------------------

class TestFrustrationLabel:
    def test_rule1_hard_threshold(self):
        frustrated, trigger = _frustration_label(9.0, 0, False)
        assert frustrated is True
        assert trigger == "delay_gt_8"

    def test_rule2_repeat_offender(self):
        frustrated, trigger = _frustration_label(5.0, 2, False)
        assert frustrated is True
        assert trigger == "delay_gt_4_repeat"

    def test_rule2_requires_prior_delays(self):
        frustrated, _ = _frustration_label(5.0, 1, False)
        assert frustrated is False

    def test_rule3_post_complaint_return(self):
        frustrated, trigger = _frustration_label(2.5, 0, True)
        assert frustrated is True
        assert trigger == "delay_gt_2_pcr"

    def test_not_frustrated(self):
        frustrated, trigger = _frustration_label(1.0, 0, False)
        assert frustrated is False
        assert trigger == "none"

    def test_rule1_takes_precedence_over_rule3(self):
        frustrated, trigger = _frustration_label(10.0, 0, True)
        assert trigger == "delay_gt_8"  # rule 1 fires first


# ---------------------------------------------------------------------------
# generate_user_profiles
# ---------------------------------------------------------------------------

class TestUserProfiles:
    def test_reproducible_with_seed(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        p1 = generate_user_profiles(10, rng1)
        p2 = generate_user_profiles(10, rng2)
        assert p1[0].user_id != p2[0].user_id  # UUIDs are random
        # But numerical fields should match
        assert p1[0].tenure_days == p2[0].tenure_days
        assert p1[0].ltv_estimate_myr == p2[0].ltv_estimate_myr

    def test_churn_sensitivity_bounded(self):
        rng = np.random.default_rng(99)
        profiles = generate_user_profiles(1000, rng)
        for p in profiles:
            assert 0.0 <= p.churn_sensitivity <= 1.0

    def test_pcr_fraction_approximately_correct(self):
        rng = np.random.default_rng(7)
        profiles = generate_user_profiles(5000, rng)
        pcr_rate = sum(p.is_post_complaint_return for p in profiles) / len(profiles)
        assert 0.02 <= pcr_rate <= 0.10  # target ~5% ± 2σ


# ---------------------------------------------------------------------------
# generate_session
# ---------------------------------------------------------------------------

class TestGenerateSession:
    def test_session_events_sorted(self):
        rng = np.random.default_rng(42)
        user = _make_user()
        session = generate_session(user, rng, hour=19, day_of_week=5)
        ts = [e.ts_offset_seconds for e in session.events]
        assert ts == sorted(ts)

    def test_session_has_order_placed(self):
        rng = np.random.default_rng(0)
        user = _make_user()
        session = generate_session(user, rng, hour=12, day_of_week=1)
        etypes = [e.event_type.value for e in session.events]
        assert "order_placed" in etypes

    def test_frustrated_session_more_refreshes(self):
        rng = np.random.default_rng(5)
        user = _make_user(prior_delays_30d=3)  # will trigger frustration on smaller delays
        n_frustrated_refreshes, n_calm_refreshes = [], []

        for _ in range(200):
            s = generate_session(user, rng, hour=19, day_of_week=4)
            if s.is_frustrated:
                n_frustrated_refreshes.append(s.eta_refresh_count)
            else:
                n_calm_refreshes.append(s.eta_refresh_count)

        if n_frustrated_refreshes and n_calm_refreshes:
            assert np.mean(n_frustrated_refreshes) > np.mean(n_calm_refreshes)
