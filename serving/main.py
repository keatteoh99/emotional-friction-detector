"""
FastAPI serving layer for the Emotional Friction Detector.

Endpoints:
  POST /score  — score a live session, return intervention decision
  GET  /health — model load status

Models are loaded once at startup via lifespan context. Latency target: P95 < 50ms.
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException

from intervention.decision_engine import DecisionEngine, ScoringResult
from models.lgbm_churn_risk.model import ChurnRiskModel, FEATURE_COLS
from models.lstm_frustration.model import FrustrationLSTM, LSTMConfig
from models.lstm_frustration.train import EVENT_TYPE_VOCAB
from .schemas import HealthResponse, ScoreRequest, ScoreResponse

LSTM_PATH = os.getenv("LSTM_MODEL_PATH", "models/lstm_frustration/artefacts/model.pt")
LGBM_PATH = os.getenv("LGBM_MODEL_PATH", "models/lgbm_churn_risk/artefacts/churn_model.lgb")
DEVICE = torch.device("cpu")  # CPU for serving; GPU adds latency variance
MAX_SEQ_LEN = 64


class ModelState:
    lstm: Optional[FrustrationLSTM] = None
    lgbm: Optional[ChurnRiskModel] = None
    engine: DecisionEngine = DecisionEngine()


state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Load models at startup ────────────────────────────────────────────────
    if Path(LSTM_PATH).exists():
        cfg = LSTMConfig()
        state.lstm = FrustrationLSTM(cfg).to(DEVICE)
        state.lstm.load_state_dict(torch.load(LSTM_PATH, map_location=DEVICE))
        state.lstm.eval()
        print(f"Loaded LSTM from {LSTM_PATH}")
    else:
        print(f"Warning: LSTM weights not found at {LSTM_PATH} — running in stub mode")

    if Path(LGBM_PATH).exists():
        state.lgbm = ChurnRiskModel.load(LGBM_PATH)
        print(f"Loaded LightGBM from {LGBM_PATH}")
    else:
        print(f"Warning: LGBM model not found at {LGBM_PATH} — running in stub mode")

    yield
    # ── Cleanup ───────────────────────────────────────────────────────────────
    state.lstm = None
    state.lgbm = None


app = FastAPI(
    title="Emotional Friction Detector",
    description="Real-time pre-churn intervention scoring API",
    version="0.1.0",
    lifespan=lifespan,
)


def _events_to_tensors(events: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T = min(len(events), MAX_SEQ_LEN)
    event_ids = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.long)
    continuous = torch.zeros(1, MAX_SEQ_LEN, 3, dtype=torch.float32)

    ts_arr = np.array([e.ts_offset_seconds for e in events[:T]], dtype=float)
    eta_arr = np.array([e.eta_remaining_minutes or 0.0 for e in events[:T]], dtype=float)
    interval_arr = np.concatenate([[0.0], np.diff(ts_arr)])

    ts_norm = ts_arr / (ts_arr.max() + 1e-9)
    eta_norm = eta_arr / (eta_arr.max() + 1e-9)
    int_norm = interval_arr / (interval_arr.max() + 1e-9)

    eids = np.array([EVENT_TYPE_VOCAB.get(e.event_type, 0) for e in events[:T]], dtype=int)
    event_ids[0, :T] = torch.tensor(eids)
    continuous[0, :T, 0] = torch.tensor(ts_norm, dtype=torch.float32)
    continuous[0, :T, 1] = torch.tensor(eta_norm, dtype=torch.float32)
    continuous[0, :T, 2] = torch.tensor(int_norm, dtype=torch.float32)
    lengths = torch.tensor([T], dtype=torch.long)

    return event_ids, continuous, lengths


@app.post("/score", response_model=ScoreResponse)
async def score(request: ScoreRequest) -> ScoreResponse:
    t0 = time.perf_counter()

    # ── Stage 1: LSTM frustration score ──────────────────────────────────────
    if request.p_frustrated_override is not None:
        p_frustrated = float(request.p_frustrated_override)
    elif state.lstm is not None:
        event_ids, continuous, lengths = _events_to_tensors(request.events)
        with torch.no_grad():
            logit = state.lstm(event_ids.to(DEVICE), continuous.to(DEVICE), lengths.to(DEVICE))
        p_frustrated = float(torch.sigmoid(logit).item())
    else:
        # Stub: use heuristic when model not loaded (dev mode)
        n_refreshes = sum(1 for e in request.events if e.event_type == "eta_refreshed")
        p_frustrated = float(min(0.95, n_refreshes / 10.0))

    # ── Stage 2: LightGBM churn risk ─────────────────────────────────────────
    if state.lgbm is not None:
        import pandas as pd
        row = {f: 0.0 for f in FEATURE_COLS}
        row["p_frustrated"] = p_frustrated
        row["ltv_estimate_myr"] = request.ltv_estimate_myr
        X = pd.DataFrame([row])
        p_churn = float(state.lgbm.predict_proba(X)[0])
    else:
        p_churn = p_frustrated * 0.65   # stub

    # ── Intervention decision ─────────────────────────────────────────────────
    scoring_result = ScoringResult(
        session_id=request.session_id,
        user_id=request.user_id,
        p_frustrated=p_frustrated,
        p_churn_given_frustrated=p_churn,
        ltv_estimate_myr=request.ltv_estimate_myr,
    )
    decision = state.engine.decide(scoring_result)
    latency_ms = (time.perf_counter() - t0) * 1000

    return ScoreResponse(
        session_id=request.session_id,
        user_id=request.user_id,
        p_frustrated=p_frustrated,
        p_churn=p_churn,
        intervention_type=decision.intervention_type.value,
        is_holdback=decision.is_holdback,
        rationale=decision.rationale,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        lstm_loaded=state.lstm is not None,
        lgbm_loaded=state.lgbm is not None,
    )
