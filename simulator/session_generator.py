"""
Session event generator — the heart of the simulation.

A session covers order_placed → order_received (or cancellation).
Events model realistic in-app behaviour; frustrated users exhibit three
distinguishing patterns that the LSTM will learn:

  1. ETA refresh bursts with compressing intervals
     Non-frustrated: ~uniform 4-8 min apart
     Frustrated: geometric compression starting at ~90s, shrinking by ~0.55×

  2. App background/foreground cycling (anxious app-switching)

  3. Map tap clustering: bursts followed by pauses (captured by tap_interval_cv)

Frustration is a ground-truth label derived from delay + context rules,
NOT from model output.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from .user_profiles import UserProfile
from .delay_distribution import (
    DelayContext,
    RestaurantTier,
    TimeSlot,
    is_peak_slot,
    sample_base_eta,
    sample_delivery_delay,
)


class EventType(str, Enum):
    ORDER_PLACED = "order_placed"
    ETA_VIEWED = "eta_viewed"
    ETA_REFRESHED = "eta_refreshed"
    MAP_TAPPED = "map_tapped"
    SUPPORT_CHAT_OPENED = "support_chat_opened"
    SUPPORT_CHAT_CLOSED = "support_chat_closed"
    APP_BACKGROUND = "app_background"
    APP_FOREGROUND = "app_foreground"
    PROMO_VIEWED = "promo_viewed"
    ORDER_RECEIVED = "order_received"
    RATING_SUBMITTED = "rating_submitted"
    ORDER_CANCELLED = "order_cancelled"


@dataclass
class SessionEvent:
    event_type: EventType
    ts_offset_seconds: float          # seconds since order_placed
    eta_remaining_minutes: Optional[float]
    metadata: dict = field(default_factory=dict)


@dataclass
class Session:
    session_id: str
    user_id: str
    order_id: str

    # Temporal context
    hour_of_day: int
    day_of_week: int

    # Delivery context
    delay_minutes: float        # actual − promised (positive = late)
    base_eta_minutes: int
    restaurant_tier: str
    is_raining: bool
    zone: str

    # Ground-truth labels
    is_frustrated: bool
    frustration_trigger: str   # which rule fired; "none" if not frustrated

    # Event log
    events: List[SessionEvent] = field(default_factory=list)

    # Populated after generation
    session_duration_seconds: float = 0.0
    eta_refresh_count: int = 0
    support_contact_made: bool = False
    completed: bool = True   # False if cancelled


# ---------------------------------------------------------------------------
# Frustration labelling rules
# ---------------------------------------------------------------------------

def _frustration_label(
    delay: float,
    prior_delays: int,
    is_post_complaint_return: bool,
) -> Tuple[bool, str]:
    """
    Rules (evaluated in precedence order):
      1. delay > 8 min                            → hard threshold
      2. delay > 4 AND prior_delays_30d ≥ 2       → accumulated grievance
      3. delay > 2 AND is_post_complaint_return   → hair-trigger sensitivity
    """
    if delay > 8.0:
        return True, "delay_gt_8"
    if delay > 4.0 and prior_delays >= 2:
        return True, "delay_gt_4_repeat"
    if delay > 2.0 and is_post_complaint_return:
        return True, "delay_gt_2_pcr"
    return False, "none"


# ---------------------------------------------------------------------------
# ETA refresh burst
# ---------------------------------------------------------------------------

def _generate_eta_refresh_burst(
    start_offset: float,
    eta_remaining: float,
    n_refreshes: int,
    rng: np.random.Generator,
    frustrated: bool,
) -> List[SessionEvent]:
    """
    Build the ETA refresh sub-sequence.

    Frustration signal: intervals compress geometrically (urgency escalation).
    Non-frustrated: roughly uniform, spaced ~4-8 min apart.
    Frustrated: starts at ~90s, each subsequent interval ≈ 0.55× the prior.
    Floor at 8s — physically impossible to refresh faster on mobile UI.
    """
    events: List[SessionEvent] = []
    t = start_offset
    eta = eta_remaining

    if frustrated:
        interval = float(rng.uniform(70, 110))
        compression = float(rng.uniform(0.50, 0.62))
    else:
        interval = float(rng.uniform(240, 480))
        compression = float(rng.uniform(0.85, 0.95))

    for i in range(n_refreshes):
        t += interval
        eta = max(0.0, eta - interval / 60)
        events.append(SessionEvent(
            event_type=EventType.ETA_REFRESHED,
            ts_offset_seconds=t,
            eta_remaining_minutes=round(eta, 1),
            metadata={"refresh_index": i, "interval_seconds": round(interval, 1)},
        ))
        interval = max(8.0, interval * compression)

    return events


# ---------------------------------------------------------------------------
# Main session generator
# ---------------------------------------------------------------------------

def generate_session(
    user: UserProfile,
    rng: np.random.Generator,
    hour: int,
    day_of_week: int,
) -> Session:
    city_rain_prob = {"KL": 0.35, "Penang": 0.45, "JB": 0.40, "Kuching": 0.55}
    is_raining = rng.random() < city_rain_prob.get(user.city, 0.35)

    restaurant_tier = str(rng.choice(["A", "B", "C"], p=[0.30, 0.45, 0.25]))
    time_slot = is_peak_slot(hour)

    ctx = DelayContext(
        time_slot=time_slot,
        is_raining=is_raining,
        restaurant_tier=RestaurantTier(restaurant_tier),
        day_of_week=day_of_week,
        zone=user.zone,
    )

    base_eta = sample_base_eta(ctx, rng)
    delay = sample_delivery_delay(ctx, rng)
    actual_minutes = base_eta + delay

    is_frustrated, trigger = _frustration_label(
        delay=delay,
        prior_delays=user.prior_delays_30d,
        is_post_complaint_return=user.is_post_complaint_return,
    )

    session = Session(
        session_id=str(uuid.uuid4()),
        user_id=user.user_id,
        order_id=str(uuid.uuid4()),
        hour_of_day=hour,
        day_of_week=day_of_week,
        delay_minutes=round(delay, 2),
        base_eta_minutes=base_eta,
        restaurant_tier=restaurant_tier,
        is_raining=is_raining,
        zone=user.zone,
        is_frustrated=is_frustrated,
        frustration_trigger=trigger,
    )

    events = _build_event_timeline(
        session=session,
        user=user,
        actual_delivery_minutes=actual_minutes,
        rng=rng,
    )
    session.events = events
    session.session_duration_seconds = max(
        (e.ts_offset_seconds for e in events), default=0.0
    )
    session.eta_refresh_count = sum(
        1 for e in events if e.event_type == EventType.ETA_REFRESHED
    )
    session.support_contact_made = any(
        e.event_type == EventType.SUPPORT_CHAT_OPENED for e in events
    )

    return session


def _build_event_timeline(
    session: Session,
    user: UserProfile,
    actual_delivery_minutes: float,
    rng: np.random.Generator,
) -> List[SessionEvent]:
    events: List[SessionEvent] = []
    t = 0.0
    total_wait_s = max(60.0, actual_delivery_minutes * 60)

    # ── Order placed ──────────────────────────────────────────────────────────
    events.append(SessionEvent(
        event_type=EventType.ORDER_PLACED,
        ts_offset_seconds=t,
        eta_remaining_minutes=float(session.base_eta_minutes),
    ))

    # ── Initial ETA view ──────────────────────────────────────────────────────
    t += float(rng.uniform(5, 30))
    events.append(SessionEvent(
        event_type=EventType.ETA_VIEWED,
        ts_offset_seconds=t,
        eta_remaining_minutes=float(session.base_eta_minutes),
    ))

    # ── Decide interaction volume based on frustration ────────────────────────
    if session.is_frustrated:
        # Ranges deliberately overlap with non-frustrated so the model must learn
        # from PATTERN (tap_interval_cv, refresh compression) not raw event count.
        n_refreshes = int(rng.integers(1, 14))   # mostly high but can be low
        # Support contact probability scales with trigger severity + churn sensitivity
        p_support = {
            "delay_gt_8": 0.55,
            "delay_gt_4_repeat": 0.35,
            "delay_gt_2_pcr": 0.42,
        }.get(session.frustration_trigger, 0.30)
        p_support = min(0.88, p_support + 0.20 * user.churn_sensitivity)
        will_contact_support = rng.random() < p_support
        n_map_taps = int(rng.integers(0, 13))    # overlaps with non-frustrated
        n_bg_cycles = int(rng.integers(0, 7))    # overlaps with non-frustrated
    else:
        n_refreshes = int(rng.integers(0, 9))    # occasional heavy-checker overlaps frustrated low end
        will_contact_support = rng.random() < 0.04
        n_map_taps = int(rng.integers(0, 8))     # overlaps with frustrated
        n_bg_cycles = int(rng.integers(0, 5))    # overlaps with frustrated

    # ── ETA refresh burst ─────────────────────────────────────────────────────
    if n_refreshes > 0:
        refresh_start = t + float(rng.uniform(60, 180))
        events.extend(_generate_eta_refresh_burst(
            start_offset=refresh_start,
            eta_remaining=float(session.base_eta_minutes),
            n_refreshes=n_refreshes,
            rng=rng,
            frustrated=session.is_frustrated,
        ))

    # ── App background / foreground cycling ───────────────────────────────────
    bg_start = t + float(rng.uniform(120, 300))
    for i in range(n_bg_cycles):
        bg_t = bg_start + i * float(rng.uniform(30, 120))
        fg_t = bg_t + float(rng.uniform(15, 90))
        if bg_t < total_wait_s * 0.85:
            events.append(SessionEvent(EventType.APP_BACKGROUND, bg_t, None))
            events.append(SessionEvent(EventType.APP_FOREGROUND, fg_t, None))

    # ── Map taps ──────────────────────────────────────────────────────────────
    map_window = max(31.0, total_wait_s * 0.75)
    for _ in range(n_map_taps):
        tap_t = t + float(rng.uniform(30, map_window))
        events.append(SessionEvent(EventType.MAP_TAPPED, tap_t, None))

    # ── Occasional promo view (non-frustrated time-filling) ───────────────────
    if not session.is_frustrated and rng.random() < 0.25:
        promo_t = t + float(rng.uniform(60, total_wait_s * 0.5))
        events.append(SessionEvent(EventType.PROMO_VIEWED, promo_t, None))

    # ── Support contact ───────────────────────────────────────────────────────
    if will_contact_support:
        support_t = total_wait_s * float(rng.uniform(0.55, 0.80))
        eta_at_support = max(0.0, session.base_eta_minutes - support_t / 60)
        events.append(SessionEvent(
            event_type=EventType.SUPPORT_CHAT_OPENED,
            ts_offset_seconds=support_t,
            eta_remaining_minutes=round(eta_at_support, 1),
            metadata={"issue_type": "delivery_delay"},
        ))
        chat_duration = float(rng.uniform(120, 900))
        events.append(SessionEvent(
            event_type=EventType.SUPPORT_CHAT_CLOSED,
            ts_offset_seconds=support_t + chat_duration,
            eta_remaining_minutes=None,
        ))

    # ── Cancellation (frustrated + high churn sensitivity + very late) ────────
    p_cancel = 0.0
    if session.is_frustrated and session.delay_minutes > 12:
        p_cancel = min(0.35, 0.08 * user.churn_sensitivity * 3)

    if rng.random() < p_cancel:
        cancel_t = total_wait_s * float(rng.uniform(0.40, 0.75))
        events.append(SessionEvent(EventType.ORDER_CANCELLED, cancel_t, None))
        session.completed = False
        events.sort(key=lambda e: e.ts_offset_seconds)
        return events

    # ── Order received ────────────────────────────────────────────────────────
    events.append(SessionEvent(
        event_type=EventType.ORDER_RECEIVED,
        ts_offset_seconds=total_wait_s,
        eta_remaining_minutes=0.0,
    ))

    # ── Rating (less likely when frustrated) ──────────────────────────────────
    p_rate = 0.30 if session.is_frustrated else 0.65
    if rng.random() < p_rate:
        rating_t = total_wait_s + float(rng.uniform(30, 300))
        rating_val = int(rng.integers(1, 4) if session.is_frustrated else rng.integers(3, 6))
        events.append(SessionEvent(
            event_type=EventType.RATING_SUBMITTED,
            ts_offset_seconds=rating_t,
            eta_remaining_minutes=None,
            metadata={"rating": rating_val},
        ))

    events.sort(key=lambda e: e.ts_offset_seconds)
    return events
