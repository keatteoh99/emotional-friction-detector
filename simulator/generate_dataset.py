"""
Orchestrate 200k session generation with ~40% frustrated sessions.

Writes three Parquet files into the resolved output directory:
  sessions.parquet       — one row per session (metadata + ground-truth label)
  events.parquet         — one row per event (long format, join on session_id)
  user_profiles.parquet  — one row per synthetic user

CLI:
  # Generate 200k sessions
  python -m simulator.generate_dataset --n 200000 --seed 42 --output data/processed/

  # --output accepts either a directory or a sessions file path:
  python -m simulator.generate_dataset --output data/processed/sessions.parquet

  # Verify an existing dataset without regenerating:
  python -m simulator.generate_dataset --output data/processed/ --stats-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .session_generator import Session, generate_session
from .user_profiles import generate_user_profiles, profiles_to_dataframe

N_SESSIONS_DEFAULT = 200_000
N_USERS_DEFAULT = 15_000
SEED_DEFAULT = 42


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _resolve_output_dir(output_arg: str) -> Path:
    """
    Accept either a directory or a file path ending in .parquet.
    Both of these mean 'write to data/processed/':
      --output data/processed/
      --output data/processed/sessions.parquet
    """
    p = Path(output_arg)
    return p.parent if p.suffix == ".parquet" else p


# ---------------------------------------------------------------------------
# Row serialisers
# ---------------------------------------------------------------------------

def _session_to_row(s: Session) -> dict:
    return {
        "session_id": s.session_id,
        "user_id": s.user_id,
        "order_id": s.order_id,
        "hour_of_day": s.hour_of_day,
        "day_of_week": s.day_of_week,
        "delay_minutes": s.delay_minutes,
        "base_eta_minutes": s.base_eta_minutes,
        "restaurant_tier": s.restaurant_tier,
        "is_raining": s.is_raining,
        "zone": s.zone,
        "is_frustrated": s.is_frustrated,
        "frustration_trigger": s.frustration_trigger,
        "session_duration_seconds": s.session_duration_seconds,
        "eta_refresh_count": s.eta_refresh_count,
        "support_contact_made": s.support_contact_made,
        "completed": s.completed,
        "n_events": len(s.events),
    }


def _events_to_rows(s: Session) -> List[dict]:
    return [
        {
            "session_id": s.session_id,
            "user_id": s.user_id,
            "event_type": e.event_type.value,
            "ts_offset_seconds": e.ts_offset_seconds,
            "eta_remaining_minutes": e.eta_remaining_minutes,
            "metadata": json.dumps(e.metadata),
        }
        for e in s.events
    ]


def _sample_hour(rng: np.random.Generator) -> int:
    """Hour-of-day weighted toward meal times."""
    weights = np.ones(24)
    weights[6:9] = 2.0    # breakfast
    weights[11:14] = 4.0  # lunch
    weights[18:22] = 5.0  # dinner
    weights /= weights.sum()
    return int(rng.choice(np.arange(24), p=weights))


# ---------------------------------------------------------------------------
# Distribution stats reporter
# ---------------------------------------------------------------------------

def print_dataset_stats(
    session_df: pd.DataFrame,
    event_df: pd.DataFrame,
    user_df: pd.DataFrame,
) -> None:
    """
    Print a full distribution report. Run before training to confirm:
      - Frustrated rate is ~40% (model class balance)
      - Delay distribution matches Olist-anchored parameters
      - No degenerate columns (all-zero features, NaN leaks, etc.)
    """
    n_s = len(session_df)
    n_e = len(event_df)
    n_u = len(user_df)
    frustrated_rate = session_df["is_frustrated"].mean()
    support_rate = session_df["support_contact_made"].mean()
    completion_rate = session_df["completed"].mean()

    # Delay distribution
    delays = session_df["delay_minutes"]
    d = delays.describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.99])

    # Frustration trigger breakdown (among frustrated sessions)
    triggers = (
        session_df[session_df["is_frustrated"]]["frustration_trigger"]
        .value_counts(normalize=True)
        .mul(100)
        .round(1)
    )

    # Context breakdown
    peak_mask = session_df["hour_of_day"].isin(list(range(11, 14)) + list(range(18, 22)))
    rain_rate = session_df["is_raining"].mean()
    tier_c_rate = (session_df["restaurant_tier"] == "C").mean()
    fringe_rate = (session_df["zone"] == "fringe").mean()
    peak_rate = peak_mask.mean()

    # LTV / user segments
    seg_counts = user_df["segment"].value_counts(normalize=True).mul(100).round(1)

    # Flag any concerns
    warnings = []
    if not (0.35 <= frustrated_rate <= 0.45):
        warnings.append(f"Frustrated rate {frustrated_rate:.1%} is outside 35–45% band")
    if delays.mean() < 3 or delays.mean() > 12:
        warnings.append(f"Mean delay {delays.mean():.1f}min looks off (expect ~5–9min)")

    ok = lambda cond: "[ok]" if cond else "[!!]"

    print()
    print("--- Dataset distribution report --------------------------------")
    print(f"  Sessions:              {n_s:>10,}")
    print(f"  Events:                {n_e:>10,}  (avg {n_e/n_s:.1f} / session)")
    print(f"  Users:                 {n_u:>10,}")
    print()
    print(f"  Label balance:")
    print(f"    Frustrated:          {frustrated_rate:>8.1%}  {ok(0.35 <= frustrated_rate <= 0.45)} target ~40%")
    print(f"    Not frustrated:      {1 - frustrated_rate:>8.1%}")
    print()
    print(f"  Frustration triggers (% of frustrated sessions):")
    for trigger, pct in triggers.items():
        print(f"    {trigger:<28s} {pct:5.1f}%")
    print()
    print(f"  Delay distribution (minutes, actual - ETA):")
    print(f"    mean={d['mean']:6.2f}  std={d['std']:5.2f}  {ok(3 <= d['mean'] <= 12)}")
    print(f"    p25={d['25%']:6.2f}  p50={d['50%']:6.2f}  p75={d['75%']:6.2f}"
          f"  p90={d['90%']:6.2f}  p99={d['99%']:6.2f}")
    print()
    print(f"  Context breakdown:")
    print(f"    Peak hour (lunch/dinner): {peak_rate:>7.1%}")
    print(f"    Rainy sessions:           {rain_rate:>7.1%}")
    print(f"    Tier-C restaurant:        {tier_c_rate:>7.1%}")
    print(f"    Fringe zone:              {fringe_rate:>7.1%}")
    print()
    print(f"  Behaviour:")
    print(f"    Support contact rate:     {support_rate:>7.1%}")
    print(f"    Session completion rate:  {completion_rate:>7.1%}")
    print(f"    Cancellation rate:        {1 - completion_rate:>7.1%}")
    print()
    print(f"  User segments (% of users):")
    for seg, pct in seg_counts.items():
        print(f"    {seg:<28s} {pct:5.1f}%")

    if warnings:
        print()
        print("  Warnings:")
        for w in warnings:
            print(f"    [!!] {w}")

    print("----------------------------------------------------------------")


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate_dataset(
    n_sessions: int = N_SESSIONS_DEFAULT,
    n_users: int = N_USERS_DEFAULT,
    seed: int = SEED_DEFAULT,
    output_dir: str = "data/raw",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate simulated sessions and write three Parquet files.
    Returns (user_df, session_df, event_df).
    """
    rng = np.random.default_rng(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Generating {n_users:,} user profiles (seed={seed})...")
    users = generate_user_profiles(n_users, rng)
    user_df = profiles_to_dataframe(users)
    user_df.to_parquet(out / "user_profiles.parquet", index=False)
    print(f"      Saved user_profiles.parquet  ({len(user_df):,} rows)")

    user_ids = [u.user_id for u in users]
    user_index = {u.user_id: u for u in users}

    print(f"[2/3] Generating {n_sessions:,} sessions...")
    session_rows: List[dict] = []
    event_rows: List[dict] = []

    for _ in tqdm(range(n_sessions), desc="Sessions", unit="sess", ncols=80):
        uid = user_ids[int(rng.integers(0, len(user_ids)))]
        session = generate_session(
            user=user_index[uid],
            rng=rng,
            hour=_sample_hour(rng),
            day_of_week=int(rng.integers(0, 7)),
        )
        session_rows.append(_session_to_row(session))
        event_rows.extend(_events_to_rows(session))

    session_df = pd.DataFrame(session_rows)
    event_df = pd.DataFrame(event_rows)

    print(f"[3/3] Writing Parquet files to {out}/")
    session_df.to_parquet(out / "sessions.parquet", index=False)
    event_df.to_parquet(out / "events.parquet", index=False)
    print(f"      sessions.parquet     {len(session_df):>10,} rows")
    print(f"      events.parquet       {len(event_df):>10,} rows")
    print(f"      user_profiles.parquet {len(user_df):>9,} rows")

    print_dataset_stats(session_df, event_df, user_df)

    return user_df, session_df, event_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate emotional friction dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --n and --n-sessions are aliases
    parser.add_argument(
        "--n", "--n-sessions",
        dest="n_sessions",
        type=int,
        default=N_SESSIONS_DEFAULT,
        metavar="N",
        help="Number of sessions to generate",
    )
    parser.add_argument(
        "--n-users",
        type=int,
        default=N_USERS_DEFAULT,
        help="Number of synthetic users (sessions are drawn from this pool)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED_DEFAULT,
        help="Global random seed",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        metavar="PATH",
        help=(
            "Output directory or sessions.parquet path. "
            "Both '--output data/processed/' and "
            "'--output data/processed/sessions.parquet' write to data/processed/."
        ),
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help=(
            "Skip generation; load existing parquets from --output and print "
            "distribution stats. Exits with error if files not found."
        ),
    )
    args = parser.parse_args()

    out_dir = _resolve_output_dir(args.output)

    if args.stats_only:
        sessions_path = out_dir / "sessions.parquet"
        events_path = out_dir / "events.parquet"
        users_path = out_dir / "user_profiles.parquet"

        missing = [p for p in (sessions_path, events_path, users_path) if not p.exists()]
        if missing:
            print(f"ERROR: cannot run --stats-only, files not found:")
            for p in missing:
                print(f"  {p}")
            print(f"\nRun without --stats-only first to generate the dataset.")
            raise SystemExit(1)

        print(f"Loading existing dataset from {out_dir}/")
        session_df = pd.read_parquet(sessions_path)
        event_df = pd.read_parquet(events_path)
        user_df = pd.read_parquet(users_path)
        print_dataset_stats(session_df, event_df, user_df)
    else:
        generate_dataset(
            n_sessions=args.n_sessions,
            n_users=args.n_users,
            seed=args.seed,
            output_dir=str(out_dir),
        )
