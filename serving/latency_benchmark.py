"""
P99 latency benchmark for the scoring API.

Tests two paths:
  1. HTTP round-trip to running FastAPI server (end-to-end)
  2. Direct in-process scoring (model inference only, no network)

Target from spec: p99 < 40ms for in-process scoring.
HTTP target is environment-dependent (network + serialisation overhead).

Usage:
  # Start the server first:
  uvicorn serving.main:app --port 8000

  # Then run benchmark:
  python -m serving.latency_benchmark --url http://localhost:8000 --n 1000
  python -m serving.latency_benchmark --mode inprocess --n 5000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Sample payload factory
# ---------------------------------------------------------------------------

def make_sample_payload(n_events: int = 8, ltv: float = 120.0) -> dict:
    """Build a realistic /score request with n_events events."""
    events = [{"event_type": "order_placed", "ts_offset_seconds": 0.0, "eta_remaining_minutes": 35.0}]
    t = 30.0
    interval = 90.0
    for i in range(n_events - 1):
        events.append({
            "event_type": "eta_refreshed",
            "ts_offset_seconds": t,
            "eta_remaining_minutes": max(0.0, 35.0 - t / 60),
            "metadata": {"refresh_index": i, "interval_seconds": interval},
        })
        t += interval
        interval *= 0.6  # compressing intervals (frustrated pattern)
    return {
        "session_id": "bench-session",
        "user_id": "bench-user",
        "ltv_estimate_myr": ltv,
        "events": events,
    }


# ---------------------------------------------------------------------------
# HTTP benchmark
# ---------------------------------------------------------------------------

def benchmark_http(url: str, n: int = 500, n_events: int = 8) -> dict:
    try:
        import httpx
    except ImportError:
        print("httpx not installed. Run: pip install httpx")
        return {}

    payload = make_sample_payload(n_events)
    latencies: List[float] = []

    print(f"HTTP benchmark: {n} requests to {url}/score  (n_events={n_events})")
    # Warmup
    for _ in range(10):
        httpx.post(f"{url}/score", json=payload, timeout=5.0)

    for _ in range(n):
        t0 = time.perf_counter()
        r = httpx.post(f"{url}/score", json=payload, timeout=5.0)
        elapsed = (time.perf_counter() - t0) * 1000
        assert r.status_code == 200, f"Non-200: {r.status_code}"
        latencies.append(elapsed)

    return _report(latencies, "HTTP end-to-end")


# ---------------------------------------------------------------------------
# In-process benchmark (model inference only)
# ---------------------------------------------------------------------------

def benchmark_inprocess(
    weights_path: str = "models/lstm_frustration/artefacts/model_best.pt",
    n: int = 2000,
    n_events: int = 8,
) -> dict:
    if not Path(weights_path).exists():
        print(f"Weights not found at {weights_path}. Train the model first.")
        return {}

    from models.lstm_frustration.predict import OnlineFrustrationScorer
    scorer = OnlineFrustrationScorer.load(weights_path)

    payload = make_sample_payload(n_events)
    event_list = [(e["event_type"], e["ts_offset_seconds"]) for e in payload["events"]]

    latencies: List[float] = []

    print(f"In-process benchmark: {n} sessions  (n_events={n_events})")
    # Warmup
    for _ in range(50):
        scorer.new_session()
        for etype, ts in event_list:
            scorer.step(etype, ts)

    for _ in range(n):
        scorer.new_session()
        t0 = time.perf_counter()
        for etype, ts in event_list:
            scorer.step(etype, ts)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

    return _report(latencies, "In-process inference")


def _report(latencies: List[float], label: str) -> dict:
    arr = np.array(latencies)
    result = {
        "label": label,
        "n": len(arr),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "mean_ms": round(float(arr.mean()), 2),
        "max_ms": round(float(arr.max()), 2),
    }
    print(f"\n── {label} ──────────────────────────────")
    print(f"  p50:  {result['p50_ms']:6.1f}ms")
    print(f"  p95:  {result['p95_ms']:6.1f}ms")
    print(f"  p99:  {result['p99_ms']:6.1f}ms   (target <40ms)")
    print(f"  mean: {result['mean_ms']:6.1f}ms")
    print(f"  max:  {result['max_ms']:6.1f}ms")
    status = "PASS" if result["p99_ms"] < 40.0 else "FAIL"
    print(f"  p99 SLA (<40ms): {status}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["http", "inprocess", "both"], default="inprocess")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--weights", default="models/lstm_frustration/artefacts/model_best.pt")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--n-events", type=int, default=8)
    args = parser.parse_args()

    if args.mode in ("inprocess", "both"):
        benchmark_inprocess(args.weights, args.n, args.n_events)
    if args.mode in ("http", "both"):
        benchmark_http(args.url, args.n, args.n_events)
