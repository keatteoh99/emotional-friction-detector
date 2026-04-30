"""
Per-event feature extraction for the LSTM frustration detector.

Produces a (T, 9) feature matrix per session, where each row is one event.
This is the direct input to the LSTM — not post-hoc session aggregates.

Feature 8 (tap_interval_cv) is the most important:
  Frustrated users exhibit a burst-pause rhythm — rapid taps, then silence,
  then rapid again. This is invisible to raw tap counts but captured as
  coefficient of variation over a rolling window of the last 5 tap events.
  A calm user who taps 8 times at steady 5-minute intervals has tap_interval_cv ≈ 0.02.
  A frustrated user who taps 8 times in three bursts has tap_interval_cv > 1.2.
  The ROLLING computation means the LSTM sees this anxiety *building*, not just
  the final aggregate — enabling early intervention before the session ends.

Online (Flink) usage — step a SequenceFeaturizer through incoming events:
    featurizer = SequenceFeaturizer()
    for event in stream:
        feat_vec = featurizer.process_event(event.type, event.ts)
        # feed to LSTM.online_step()

Batch (training) usage — build full feature matrix for a session:
    feat_matrix = build_per_event_features(event_list)  # (T, 9)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

# Outcome events: happen AFTER or AS A RESULT of frustration — not predictive
# behavioral precursors. Excluded from the LSTM feature sequence entirely.
# In a real Flink streaming scorer, these events arrive after the scoring window
# has already fired; including them would constitute label leakage.
OUTCOME_EVENTS: frozenset[str] = frozenset({
    "support_chat_opened",   # user reacts to frustration — not a cause
    "support_chat_closed",   # consequence of support_chat_opened
    "order_cancelled",       # session outcome, not behavioral signal
    "order_received",        # session end — not available at real-time scoring
    "rating_submitted",      # post-delivery — session is over
})

# Frustration weight for the BEHAVIORAL events the LSTM sees.
# support_chat_opened / order_cancelled are excluded above — they ARE the label,
# not predictors of it. eta_refreshed is common even for calm users (low weight).
# promo_viewed is slightly negative — calm user browsing rather than checking ETA.
EVENT_WEIGHTS: dict[str, float] = {
    "eta_refreshed":        0.18,
    "map_tapped":           0.15,
    "app_background":       0.08,
    "app_foreground":       0.06,
    "eta_viewed":           0.04,
    "delay_notification":   0.85,   # explicit platform delay push-notification
    "order_placed":         0.00,
    "promo_viewed":        -0.05,
}

# Events that constitute "taps" for interval CV computation
_TAP_TYPES = frozenset({
    "eta_refreshed", "map_tapped", "eta_viewed", "app_foreground",
})

FEATURE_DIM = 9
FEATURE_NAMES = [
    "event_weight",
    "log1p_time_since_last",
    "eta_refresh_count_2min_norm",
    "back_tap_count_norm",
    "rapid_tap_count_norm",
    "page_revisit_count_norm",
    "delay_notification_flag",
    "tap_interval_cv",
    "log1p_session_elapsed_norm",
]


# ---------------------------------------------------------------------------
# Stateful per-event featurizer (LSTM input builder)
# ---------------------------------------------------------------------------

class SequenceFeaturizer:
    """
    Stateful featurizer that produces one 9-dim feature vector per event.

    Designed for both offline batch processing (training) and online
    streaming (Flink operator) where events arrive one at a time.

    Parameters
    ----------
    tap_cv_window
        Number of most-recent tap events used for rolling tap_interval_cv.
        Default 5 → need 6 tap timestamps for 5 intervals.
    refresh_window_seconds
        Lookback window for eta_refresh_count feature.
    rapid_tap_threshold_seconds
        Two taps are "rapid" if the gap is below this value.
    """

    def __init__(
        self,
        tap_cv_window: int = 5,
        refresh_window_seconds: float = 120.0,
        rapid_tap_threshold_seconds: float = 5.0,
    ):
        self.tap_cv_window = tap_cv_window
        self.refresh_window_seconds = refresh_window_seconds
        self.rapid_tap_threshold = rapid_tap_threshold_seconds
        self._reset()

    def _reset(self) -> None:
        # Tap timestamps for rolling CV — maxlen gives us window+1 points → window intervals
        self._tap_ts: deque[float] = deque(maxlen=self.tap_cv_window + 1)

        # Recent events for windowed refresh count (type, timestamp)
        self._recent: deque[Tuple[str, float]] = deque()

        self._session_start: Optional[float] = None
        self._prev_event_ts: float = 0.0
        self._prev_tap_ts: Optional[float] = None

        # Cumulative counters
        self._back_tap_count: int = 0
        self._rapid_tap_count: int = 0
        self._page_revisit_count: int = 0
        self._delay_notified: bool = False
        self._n_events_seen: int = 0

    def reset(self) -> None:
        """Reset for a new session."""
        self._reset()

    def process_event(
        self,
        event_type: str,
        ts: float,
    ) -> np.ndarray:
        """
        Process one event and return its 9-dimensional feature vector.

        Parameters
        ----------
        event_type
            String event type matching EVENT_WEIGHTS keys.
        ts
            Timestamp in seconds since order_placed (ts_offset_seconds).

        Returns
        -------
        np.ndarray of shape (9,) dtype float32.
        """
        if self._session_start is None:
            self._session_start = ts

        # ── Windowed refresh count (prune stale entries first) ────────────────
        cutoff = ts - self.refresh_window_seconds
        while self._recent and self._recent[0][1] < cutoff:
            self._recent.popleft()
        self._recent.append((event_type, ts))
        eta_refresh_2min = sum(1 for etype, _ in self._recent if etype == "eta_refreshed")

        # ── Tap-type events ───────────────────────────────────────────────────
        if event_type in _TAP_TYPES:
            # Rapid tap: gap to previous tap < threshold
            if self._prev_tap_ts is not None and (ts - self._prev_tap_ts) < self.rapid_tap_threshold:
                self._rapid_tap_count += 1
            self._prev_tap_ts = ts
            self._tap_ts.append(ts)

        # ── Other per-type counters ───────────────────────────────────────────
        if event_type == "app_background":
            self._back_tap_count += 1

        # Page revisit: any eta_viewed after the first event is a revisit
        if event_type == "eta_viewed" and self._n_events_seen > 0:
            self._page_revisit_count += 1

        if event_type == "delay_notification":
            self._delay_notified = True

        # ── Rolling tap_interval_cv ───────────────────────────────────────────
        tap_cv = _rolling_cv(self._tap_ts)

        # ── Assemble feature vector ───────────────────────────────────────────
        time_since_last = max(0.0, ts - self._prev_event_ts)
        session_elapsed = ts - self._session_start

        feat = np.array([
            EVENT_WEIGHTS.get(event_type, 0.0),       # 1. event_weight
            float(np.log1p(time_since_last)),           # 2. log(Δt)
            min(eta_refresh_2min, 10) / 5.0,           # 3. eta_refresh_count_2min / 5
            min(self._back_tap_count, 6) / 3.0,        # 4. back_tap_count / 3
            min(self._rapid_tap_count, 6) / 3.0,       # 5. rapid_tap_count / 3
            min(self._page_revisit_count, 8) / 4.0,    # 6. page_revisit_count / 4
            float(self._delay_notified),                # 7. delay_notification_flag
            tap_cv,                                     # 8. tap_interval_cv  ← primary signal
            float(np.log1p(session_elapsed)) / 8.0,    # 9. log(elapsed) / 8
        ], dtype=np.float32)

        self._prev_event_ts = ts
        self._n_events_seen += 1
        return feat


def _rolling_cv(timestamps: deque) -> float:
    """
    CV of intervals between the buffered tap timestamps.

    With maxlen = tap_cv_window + 1, the deque holds at most window+1
    timestamps → window intervals. Returns 0.0 if fewer than 2 timestamps.

    High CV  → burst-pause pattern → frustrated
    Low CV   → steady rhythm → not frustrated
    """
    if len(timestamps) < 2:
        return 0.0
    intervals = np.diff(sorted(timestamps))
    mean = intervals.mean()
    if mean < 1e-9:
        return 0.0
    return float(intervals.std() / mean)


# ---------------------------------------------------------------------------
# Batch builder (training pipeline entry point)
# ---------------------------------------------------------------------------

@dataclass
class EventRecord:
    event_type: str
    ts_offset_seconds: float


def build_per_event_features(
    events: Sequence[EventRecord],
    tap_cv_window: int = 5,
    refresh_window_seconds: float = 120.0,
) -> np.ndarray:
    """
    Build (T, 9) feature matrix for a single session's event list.

    Events must be ordered by ts_offset_seconds (ascending).
    Returns np.ndarray of shape (len(events), 9) dtype float32.

    Example
    -------
    >>> events = [EventRecord("order_placed", 0), EventRecord("eta_refreshed", 90), ...]
    >>> feats = build_per_event_features(events)  # (T, 9)
    """
    featurizer = SequenceFeaturizer(
        tap_cv_window=tap_cv_window,
        refresh_window_seconds=refresh_window_seconds,
    )
    rows = []
    for ev in events:
        if ev.event_type in OUTCOME_EVENTS:
            continue   # never feed post-frustration outcomes to the model
        rows.append(featurizer.process_event(ev.event_type, ev.ts_offset_seconds))
    return np.stack(rows, axis=0) if rows else np.zeros((0, FEATURE_DIM), dtype=np.float32)


def build_feature_matrices_from_df(
    events_df: pd.DataFrame,
    tap_cv_window: int = 5,
    refresh_window_seconds: float = 120.0,
) -> dict[str, np.ndarray]:
    """
    Build per-session (T, 9) feature matrices from the events Parquet table.

    Parameters
    ----------
    events_df
        Long-format DataFrame: [session_id, event_type, ts_offset_seconds]
    Returns
    -------
    dict mapping session_id → np.ndarray of shape (T, 9)
    """
    result = {}
    for session_id, grp in events_df.groupby("session_id"):
        grp = grp.sort_values("ts_offset_seconds")
        evs = [
            EventRecord(row.event_type, row.ts_offset_seconds)
            for row in grp.itertuples()
            if row.event_type not in OUTCOME_EVENTS   # exclude post-frustration outcomes
        ]
        result[session_id] = build_per_event_features(
            evs, tap_cv_window, refresh_window_seconds
        )
    return result


# ---------------------------------------------------------------------------
# Session-level aggregate features (LightGBM / analysis — separate concern)
# ---------------------------------------------------------------------------
# These are NOT the LSTM input. They are post-session summaries used as
# cross-session features in the LightGBM churn head and for EDA.

def compute_tap_interval_cv(tap_timestamps: np.ndarray) -> float:
    """Session-level CV of all tap intervals. Used in LightGBM and EDA."""
    if len(tap_timestamps) < 2:
        return 0.0
    intervals = np.diff(np.sort(tap_timestamps))
    mean = intervals.mean()
    if mean < 1e-9:
        return 0.0
    return float(intervals.std() / mean)


def compute_eta_refresh_compression_ratio(refresh_timestamps: np.ndarray) -> float:
    """Last / first inter-refresh interval. < 1 = compression = escalating urgency."""
    if len(refresh_timestamps) < 3:
        return 1.0
    intervals = np.diff(np.sort(refresh_timestamps))
    if intervals[0] < 1e-9:
        return 1.0
    return float(intervals[-1] / intervals[0])


def extract_session_aggregate_features(session_events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute session-level aggregate features grouped by session_id.
    Used for LightGBM training and exploratory analysis — NOT LSTM input.

    Returns DataFrame indexed by session_id.
    """
    records = []
    _TAP_SET = frozenset({"eta_refreshed", "map_tapped", "eta_viewed"})
    # support_chat_opened excluded: it is an outcome event, not a pre-intervention signal
    _ANXIETY_SET = frozenset({"eta_refreshed", "map_tapped", "app_foreground"})

    for session_id, grp in session_events_df.groupby("session_id"):
        grp = grp.sort_values("ts_offset_seconds")
        ts = grp["ts_offset_seconds"].to_numpy(dtype=float)
        etypes = grp["event_type"].to_numpy(dtype=str)

        tap_ts = ts[np.isin(etypes, list(_TAP_SET))]
        refresh_ts = ts[etypes == "eta_refreshed"]
        bg_ts = ts[etypes == "app_background"]
        fg_ts = ts[etypes == "app_foreground"]
        support_ts = ts[etypes == "support_chat_opened"]
        map_ts = ts[etypes == "map_tapped"]

        session_duration = float(ts[-1]) if len(ts) > 0 else 0.0

        eta_col = grp.loc[grp["event_type"] == "eta_refreshed", "eta_remaining_minutes"].to_numpy(float)
        first_refresh_eta = float(eta_col[0]) if len(eta_col) > 0 else np.nan
        last_refresh_eta = float(eta_col[-1]) if len(eta_col) > 0 else np.nan

        support_open: Optional[float] = float(support_ts[0]) if len(support_ts) > 0 else None
        support_latency = (
            float(np.clip(support_open / session_duration, 0, 1))
            if support_open is not None and session_duration > 0 else 1.0
        )

        bg_fg_cycles = min(len(bg_ts), len(fg_ts))
        bg_fg_rate = float(bg_fg_cycles / (session_duration / 60)) if session_duration > 0 else 0.0

        anxiety_count = int(np.isin(etypes, list(_ANXIETY_SET)).sum())

        records.append({
            "session_id": session_id,
            "tap_interval_cv": compute_tap_interval_cv(tap_ts),
            "eta_refresh_count": int(len(refresh_ts)),
            "eta_refresh_compression_ratio": compute_eta_refresh_compression_ratio(refresh_ts),
            "first_refresh_eta_minutes": first_refresh_eta,
            "last_refresh_eta_minutes": last_refresh_eta,
            "map_tap_count": int(len(map_ts)),
            "map_tap_interval_cv": compute_tap_interval_cv(map_ts),
            "bg_fg_cycle_rate_per_min": bg_fg_rate,
            "contacted_support": int(support_open is not None),
            "support_latency_ratio": support_latency,
            "session_duration_seconds": session_duration,
            "events_per_minute": float(len(ts) / (session_duration / 60) if session_duration > 0 else 0),
            "anxiety_event_count": anxiety_count,
            "anxiety_event_rate_per_min": float(
                anxiety_count / (session_duration / 60) if session_duration > 60 else 0
            ),
        })

    return pd.DataFrame(records).set_index("session_id")


# keep old name as alias for backwards compat in notebooks
extract_sequence_features = extract_session_aggregate_features
