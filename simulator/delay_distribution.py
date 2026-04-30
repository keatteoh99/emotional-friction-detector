"""
Delivery delay distributions anchored to Olist dataset delivery gap statistics.

Olist e-commerce dataset (Brazil) shows a median delivery gap of ~2 days with
right-skewed variance. We scale this to food delivery minutes and add platform-
specific modifiers. Base distribution: N(0, 4) represents symmetric variance
around the promised ETA — positive means late, negative means early.

Modifier stack (additive on the base sample):
  Peak hour    → N(3,  2.0)   moderate rider congestion (food delivery, not e-commerce)
  Rain         → N(3,  2.5)   slower riders, higher demand
  Tier-C venue → N(5,  3.0)   slower prep, unreliable kitchen SLA
  Weekend eve  → N(2,  1.5)   surge amplifier on dinner peak
  Fringe zone  → N(3,  1.5)   last-mile distance penalty

Calibration note: original Olist-scale modifiers (peak N(7,2.5), rain N(5,3),
tier-C N(8,4)) overstated food delivery congestion — Olist covers multi-day
e-commerce, not 30-60 min food delivery windows. Recalibrated to produce
~40% frustrated sessions at seed=42 with 200k sessions (target band: 35-45%).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class TimeSlot(str, Enum):
    OFF_PEAK = "off_peak"
    LUNCH_PEAK = "lunch_peak"
    DINNER_PEAK = "dinner_peak"
    LATE_NIGHT = "late_night"


class RestaurantTier(str, Enum):
    A = "A"  # Top-rated, fast prep
    B = "B"  # Mid-tier
    C = "C"  # Slower prep, high variance (+N(8,4))


@dataclass(frozen=True)
class DelayContext:
    time_slot: TimeSlot
    is_raining: bool
    restaurant_tier: RestaurantTier
    day_of_week: int   # 0=Mon … 6=Sun
    zone: str          # "central" | "suburban" | "fringe"


def sample_delivery_delay(ctx: DelayContext, rng: np.random.Generator) -> float:
    """
    Return actual delay vs promised ETA in minutes.
    Positive = late, negative = early (early delivery is possible).
    """
    delay = rng.normal(0.0, 4.0)  # base variance

    if ctx.time_slot in (TimeSlot.LUNCH_PEAK, TimeSlot.DINNER_PEAK):
        delay += rng.normal(3.0, 2.0)
    elif ctx.time_slot == TimeSlot.LATE_NIGHT:
        delay += rng.normal(-1.5, 3.0)  # fewer orders, slight early tendency

    if ctx.is_raining:
        delay += rng.normal(3.0, 2.5)

    if ctx.restaurant_tier == RestaurantTier.C:
        delay += rng.normal(5.0, 3.0)

    # Weekend dinner amplifier
    if ctx.day_of_week >= 5 and ctx.time_slot == TimeSlot.DINNER_PEAK:
        delay += rng.normal(2.0, 1.5)

    if ctx.zone == "fringe":
        delay += rng.normal(3.0, 1.5)

    return float(delay)


def sample_base_eta(ctx: DelayContext, rng: np.random.Generator) -> int:
    """
    Return the ETA shown to the user at order placement, in minutes.
    Platform typically adds a buffer above true expected delivery time.
    """
    base_by_slot = {
        TimeSlot.OFF_PEAK: 25,
        TimeSlot.LUNCH_PEAK: 35,
        TimeSlot.DINNER_PEAK: 38,
        TimeSlot.LATE_NIGHT: 30,
    }
    base = base_by_slot[ctx.time_slot]

    if ctx.restaurant_tier == RestaurantTier.C:
        base += 10
    if ctx.is_raining:
        base += 8
    if ctx.zone == "fringe":
        base += 7

    # U(-5, +5) jitter so ETAs aren't all round numbers
    return max(15, int(base + rng.integers(-5, 6)))


def is_peak_slot(hour: int) -> TimeSlot:
    if 11 <= hour <= 13:
        return TimeSlot.LUNCH_PEAK
    elif 18 <= hour <= 21:
        return TimeSlot.DINNER_PEAK
    elif hour >= 22 or hour <= 4:
        return TimeSlot.LATE_NIGHT
    return TimeSlot.OFF_PEAK
