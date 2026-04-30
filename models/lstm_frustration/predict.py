"""
Online inference wrapper for FrustrationLSTM.

OnlineFrustrationScorer drives the model one event at a time, maintaining
LSTM hidden state across the session. This mirrors how the Flink operator
works in production — events are processed as they arrive, not buffered
to session end.

Scoring never restarts from zero mid-session: if an early event barely
clears the threshold, subsequent calm events can pull the score back down.
The hidden state carries the full session history implicitly.

Usage (Flink operator or FastAPI):
    scorer = OnlineFrustrationScorer.load("models/lstm_frustration/artefacts/model.pt")
    scorer.new_session()
    for event in stream:
        p_frustrated = scorer.step(event.type, event.ts)
        if p_frustrated > THRESHOLD:
            fire_intervention(session_id)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from features.sequence_features import SequenceFeaturizer
from .model import FrustrationLSTM, LSTMConfig, LSTMHiddenState


class OnlineFrustrationScorer:
    """
    Stateful per-session LSTM scorer.

    One instance per active session. Call new_session() to reset state
    between sessions (or just create a new instance).
    """

    def __init__(self, model: FrustrationLSTM, device: str = "cpu"):
        self.model = model
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)

        self._featurizer = SequenceFeaturizer()
        self._state: Optional[LSTMHiddenState] = None
        self._last_p: float = 0.0
        self._n_events: int = 0

    def new_session(self) -> None:
        """Reset featurizer and LSTM state for a fresh session."""
        self._featurizer.reset()
        self._state = None
        self._last_p = 0.0
        self._n_events = 0

    def step(self, event_type: str, ts_offset_seconds: float) -> float:
        """
        Process one event; return updated P(frustrated).

        The score starts near 0 and rises as frustration signals accumulate.
        Returns the same score repeatedly if called with no new events (safe).
        """
        feat = self._featurizer.process_event(event_type, ts_offset_seconds)
        feat_t = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            p, self._state = self.model.online_step(feat_t, self._state)

        self._last_p = p
        self._n_events += 1
        return p

    @property
    def current_score(self) -> float:
        return self._last_p

    @property
    def events_processed(self) -> int:
        return self._n_events

    @classmethod
    def load(
        cls,
        weights_path: str,
        cfg: Optional[LSTMConfig] = None,
        device: str = "cpu",
    ) -> "OnlineFrustrationScorer":
        cfg = cfg or LSTMConfig()
        model = FrustrationLSTM(cfg)
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        return cls(model, device)


def batch_score_sessions(
    feature_matrices: dict[str, np.ndarray],
    weights_path: str,
    device: str = "cpu",
    batch_size: int = 512,
) -> dict[str, float]:
    """
    Score all sessions in one pass using the batch forward().
    Returns dict of session_id → p_frustrated.
    Used for generating LSTM outputs to feed into LightGBM training.
    """
    from torch.utils.data import DataLoader, TensorDataset

    cfg = LSTMConfig()
    model = FrustrationLSTM(cfg)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    session_ids = list(feature_matrices.keys())
    max_len = cfg.max_seq_len
    padded = np.zeros((len(session_ids), max_len, 9), dtype=np.float32)
    lengths = np.ones(len(session_ids), dtype=np.int64)

    for i, sid in enumerate(session_ids):
        mat = feature_matrices[sid]
        T = min(len(mat), max_len)
        padded[i, :T] = mat[:T]
        lengths[i] = max(1, T)

    ds = TensorDataset(
        torch.tensor(padded, dtype=torch.float32),
        torch.tensor(lengths, dtype=torch.long),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    probs = []
    with torch.no_grad():
        for features, lens in dl:
            logits = model(features.to(dev), lens.to(dev))
            probs.extend(torch.sigmoid(logits).cpu().tolist())

    return dict(zip(session_ids, probs))
