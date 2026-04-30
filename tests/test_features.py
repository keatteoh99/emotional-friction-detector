"""
Tests for sequence_features.py — both the per-event LSTM input builder
and the session-level aggregate features.
"""
import numpy as np
import pandas as pd
import pytest

from features.sequence_features import (
    FEATURE_DIM,
    FEATURE_NAMES,
    EVENT_WEIGHTS,
    SequenceFeaturizer,
    build_per_event_features,
    build_feature_matrices_from_df,
    compute_tap_interval_cv,
    compute_eta_refresh_compression_ratio,
    extract_session_aggregate_features,
    EventRecord,
    _rolling_cv,
)


# ---------------------------------------------------------------------------
# _rolling_cv  (internal, but critical — tests its contract directly)
# ---------------------------------------------------------------------------

class TestRollingCV:
    def test_empty_returns_zero(self):
        from collections import deque
        assert _rolling_cv(deque()) == 0.0

    def test_single_point_returns_zero(self):
        from collections import deque
        assert _rolling_cv(deque([100.0])) == 0.0

    def test_equal_intervals_returns_zero(self):
        from collections import deque
        # timestamps 0, 100, 200, 300 → intervals [100, 100, 100] → std=0 → CV=0
        d = deque([0, 100, 200, 300], maxlen=10)
        assert _rolling_cv(d) == pytest.approx(0.0, abs=1e-9)

    def test_high_cv_for_bursty_timestamps(self):
        from collections import deque
        # Burst: 10, 8, 200, 9, 11 → large variance
        ts = np.cumsum([0, 10, 8, 200, 9, 11]).tolist()
        d = deque(ts, maxlen=10)
        assert _rolling_cv(d) > 1.0

    def test_maxlen_limits_window(self):
        from collections import deque
        # maxlen=4 → only last 4 timestamps kept → 3 intervals
        d = deque(maxlen=4)
        for t in [0, 300, 600, 10, 12, 14]:  # last 4: 600, 10, 12, 14
            d.append(t)
        # intervals: 10-600=-590 (sorted: 10,12,14,600) → intervals [2, 2, 586]
        # std >> mean → high CV
        cv = _rolling_cv(d)
        assert cv > 1.0


# ---------------------------------------------------------------------------
# SequenceFeaturizer — per-event feature vectors
# ---------------------------------------------------------------------------

class TestSequenceFeaturizer:
    def test_output_shape(self):
        f = SequenceFeaturizer()
        vec = f.process_event("order_placed", 0.0)
        assert vec.shape == (FEATURE_DIM,)
        assert vec.dtype == np.float32

    def test_feature_names_count(self):
        assert len(FEATURE_NAMES) == FEATURE_DIM

    def test_first_event_zero_time_since_last(self):
        f = SequenceFeaturizer()
        vec = f.process_event("order_placed", 0.0)
        # Feature 2: log1p(time_since_last) — first event has Δt=0 → log1p(0)=0
        assert vec[1] == pytest.approx(0.0)

    def test_event_weight_applied(self):
        f = SequenceFeaturizer()
        vec = f.process_event("eta_refreshed", 10.0)
        assert vec[0] == pytest.approx(EVENT_WEIGHTS["eta_refreshed"])

    def test_unknown_event_type_zero_weight(self):
        f = SequenceFeaturizer()
        vec = f.process_event("some_unknown_event", 5.0)
        assert vec[0] == pytest.approx(0.0)

    def test_time_since_last_increases(self):
        f = SequenceFeaturizer()
        f.process_event("order_placed", 0.0)
        v1 = f.process_event("eta_viewed", 60.0)   # Δt=60
        f2 = SequenceFeaturizer()
        f2.process_event("order_placed", 0.0)
        v2 = f2.process_event("eta_viewed", 300.0)  # Δt=300
        # Feature 2: log1p(Δt) — larger gap → larger value
        assert v2[1] > v1[1]

    def test_back_tap_count_increments(self):
        f = SequenceFeaturizer()
        f.process_event("order_placed", 0.0)
        f.process_event("app_background", 60.0)
        f.process_event("app_background", 120.0)
        vec = f.process_event("eta_viewed", 180.0)
        # Feature 4: back_tap_count / 3 — 2 bg events → 2/3
        assert vec[3] == pytest.approx(2 / 3.0)

    def test_rapid_tap_count_increments(self):
        f = SequenceFeaturizer()
        f.process_event("order_placed", 0.0)
        f.process_event("eta_refreshed", 100.0)   # first tap
        f.process_event("eta_refreshed", 103.0)   # Δt=3s < 5s threshold → rapid
        vec = f.process_event("map_tapped", 106.0) # Δt=3s → rapid again
        # 2 rapid taps total
        assert vec[4] == pytest.approx(2 / 3.0)

    def test_page_revisit_count(self):
        f = SequenceFeaturizer()
        f.process_event("order_placed", 0.0)
        f.process_event("eta_viewed", 30.0)   # first eta_viewed after start → revisit
        f.process_event("eta_viewed", 200.0)  # second revisit
        vec = f.process_event("map_tapped", 300.0)
        # Feature 6: page_revisit / 4 — 2 revisits → 2/4
        assert vec[5] == pytest.approx(2 / 4.0)

    def test_delay_notification_flag_fires(self):
        f = SequenceFeaturizer()
        f.process_event("order_placed", 0.0)
        vec_before = f.process_event("eta_refreshed", 60.0)
        assert vec_before[6] == pytest.approx(0.0)
        f.process_event("delay_notification", 120.0)
        vec_after = f.process_event("eta_refreshed", 180.0)
        assert vec_after[6] == pytest.approx(1.0)

    def test_tap_interval_cv_zero_initially(self):
        f = SequenceFeaturizer()
        vec = f.process_event("order_placed", 0.0)
        assert vec[7] == pytest.approx(0.0)  # no taps yet

    def test_tap_interval_cv_nonzero_after_bursty_taps(self):
        f = SequenceFeaturizer()
        f.process_event("order_placed", 0.0)
        f.process_event("eta_refreshed", 10.0)
        f.process_event("eta_refreshed", 18.0)
        f.process_event("eta_refreshed", 250.0)  # long gap
        f.process_event("eta_refreshed", 260.0)
        vec = f.process_event("eta_refreshed", 268.0)
        # 5 refreshes with bursty intervals → CV > 0
        assert vec[7] > 0.0

    def test_tap_interval_cv_frustrated_vs_calm(self):
        # Frustrated: burst-pause-burst pattern
        f_frustrated = SequenceFeaturizer()
        f_frustrated.process_event("order_placed", 0.0)
        for t in [10, 18, 300, 308, 315]:  # bursts at 10-18 and 300-315
            f_frustrated.process_event("eta_refreshed", float(t))
        vec_f = f_frustrated.process_event("eta_refreshed", 320.0)

        # Calm: steady 5-minute intervals
        f_calm = SequenceFeaturizer()
        f_calm.process_event("order_placed", 0.0)
        for t in [300, 600, 900, 1200, 1500]:
            f_calm.process_event("eta_refreshed", float(t))
        vec_c = f_calm.process_event("eta_refreshed", 1800.0)

        assert vec_f[7] > vec_c[7], (
            f"Frustrated CV {vec_f[7]:.4f} should exceed calm CV {vec_c[7]:.4f}"
        )

    def test_session_elapsed_increases(self):
        f = SequenceFeaturizer()
        f.process_event("order_placed", 0.0)
        v1 = f.process_event("eta_viewed", 60.0)
        v2 = f.process_event("eta_viewed", 600.0)
        # Feature 9: log(elapsed) / 8 — later event → larger
        assert v2[8] > v1[8]

    def test_reset_clears_state(self):
        f = SequenceFeaturizer()
        f.process_event("order_placed", 0.0)
        for t in [30, 60, 90, 120]:
            f.process_event("eta_refreshed", float(t))
        f.reset()
        vec = f.process_event("order_placed", 0.0)
        # After reset: back to first-event state
        assert vec[3] == 0.0   # back_tap_count
        assert vec[7] == 0.0   # tap_interval_cv


# ---------------------------------------------------------------------------
# build_per_event_features
# ---------------------------------------------------------------------------

class TestBuildPerEventFeatures:
    def _make_events(self, types_and_ts):
        return [EventRecord(et, t) for et, t in types_and_ts]

    def test_output_shape(self):
        evs = self._make_events([
            ("order_placed", 0), ("eta_refreshed", 90), ("map_tapped", 200),
        ])
        mat = build_per_event_features(evs)
        assert mat.shape == (3, FEATURE_DIM)
        assert mat.dtype == np.float32

    def test_empty_events_returns_empty_array(self):
        mat = build_per_event_features([])
        assert mat.shape == (0, FEATURE_DIM)

    def test_feature_7_is_tap_interval_cv(self):
        # Single tap → CV=0
        evs = self._make_events([("order_placed", 0), ("eta_refreshed", 60)])
        mat = build_per_event_features(evs)
        assert mat[1, 7] == pytest.approx(0.0)

    def test_matrices_from_df_group_correctly(self):
        rows = [
            {"session_id": "s1", "event_type": "order_placed", "ts_offset_seconds": 0.0},
            {"session_id": "s1", "event_type": "eta_refreshed", "ts_offset_seconds": 90.0},
            {"session_id": "s2", "event_type": "order_placed", "ts_offset_seconds": 0.0},
        ]
        df = pd.DataFrame(rows)
        mats = build_feature_matrices_from_df(df)
        assert "s1" in mats and "s2" in mats
        assert mats["s1"].shape == (2, FEATURE_DIM)
        assert mats["s2"].shape == (1, FEATURE_DIM)


# ---------------------------------------------------------------------------
# Session-level aggregate features (backwards compat)
# ---------------------------------------------------------------------------

class TestSessionAggregateFeatures:
    def _make_df(self, session_id, events):
        return pd.DataFrame([
            {"session_id": session_id, "event_type": et,
             "ts_offset_seconds": t, "eta_remaining_minutes": eta}
            for et, t, eta in events
        ])

    def test_tap_interval_cv_zero_for_single_tap(self):
        assert compute_tap_interval_cv(np.array([100.0])) == 0.0

    def test_tap_interval_cv_zero_for_equal_intervals(self):
        cv = compute_tap_interval_cv(np.array([0, 100, 200, 300], dtype=float))
        assert cv == pytest.approx(0.0, abs=1e-9)

    def test_compression_ratio_below_one_for_shrinking_intervals(self):
        ts = np.cumsum([0, 90, 50, 28, 16], dtype=float)
        assert compute_eta_refresh_compression_ratio(ts) < 1.0

    def test_extract_aggregate_produces_correct_columns(self):
        df = self._make_df("s1", [
            ("order_placed",  0, 30), ("eta_refreshed", 90, 28),
            ("map_tapped", 200, None), ("order_received", 1800, 0),
        ])
        feats = extract_session_aggregate_features(df)
        assert "tap_interval_cv" in feats.columns
        assert "eta_refresh_count" in feats.columns
        assert "contacted_support" in feats.columns
        assert feats.loc["s1", "eta_refresh_count"] == 1
        assert feats.loc["s1", "contacted_support"] == 0
