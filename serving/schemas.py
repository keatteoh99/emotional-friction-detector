"""Pydantic v2 request/response schemas for the scoring API."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SessionEventPayload(BaseModel):
    event_type: str
    ts_offset_seconds: float
    eta_remaining_minutes: Optional[float] = None
    metadata: dict = Field(default_factory=dict)


class ScoreRequest(BaseModel):
    session_id: str
    user_id: str
    events: List[SessionEventPayload]
    ltv_estimate_myr: float = Field(..., gt=0)
    p_frustrated_override: Optional[float] = None   # for testing / shadow mode


class ScoreResponse(BaseModel):
    session_id: str
    user_id: str
    p_frustrated: float = Field(..., ge=0.0, le=1.0)
    p_churn: float = Field(..., ge=0.0, le=1.0)
    intervention_type: str
    is_holdback: bool
    rationale: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    lstm_loaded: bool
    lgbm_loaded: bool
    version: str = "0.1.0"
