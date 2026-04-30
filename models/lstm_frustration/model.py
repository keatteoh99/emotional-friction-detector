"""
LSTM frustration detector.

Input: (T, 9) pre-computed feature matrix from SequenceFeaturizer.
No embedding layer — event_weight already encodes type semantics, making
the 9-feature vector directly interpretable and streaming-compatible.

Two usage modes:
  batch      — forward(features, lengths) for training on packed sequences
  online     — online_step(feat_vec, state) for real-time per-event scoring

Hidden size 64 is intentional: the 9-feature input is already densely
informative. Larger hidden sizes overfit on the simulated data and add
latency without meaningful AUC gain in ablations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


LSTMHiddenState = Tuple[torch.Tensor, torch.Tensor]  # (h, c) each (num_layers, B, hidden)


@dataclass
class LSTMConfig:
    input_size: int = 9         # matches FEATURE_DIM in sequence_features.py
    hidden_size: int = 64       # intentionally modest — see module docstring
    num_layers: int = 2
    dropout: float = 0.30
    max_seq_len: int = 64


class FrustrationLSTM(nn.Module):
    """
    Binary classifier: P(frustrated | event_sequence).
    Returns logits (not sigmoid) — use BCEWithLogitsLoss in training,
    torch.sigmoid() at inference.
    """

    def __init__(self, cfg: LSTMConfig = LSTMConfig()):
        super().__init__()
        self.cfg = cfg
        self.input_norm = nn.LayerNorm(cfg.input_size)
        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_size),
            nn.Linear(cfg.hidden_size, 32),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        features: torch.Tensor,   # (B, T, 9)
        lengths: torch.Tensor,    # (B,) actual sequence lengths
    ) -> torch.Tensor:            # (B,) logits
        """Batch forward — used during training."""
        x = self.input_norm(features)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers, B, hidden) — take last layer
        return self.head(h_n[-1]).squeeze(-1)   # (B,)

    def online_step(
        self,
        feat_vec: torch.Tensor,                 # (1, 9) single event
        state: Optional[LSTMHiddenState] = None,
    ) -> Tuple[float, LSTMHiddenState]:
        """
        Process a single event and return (p_frustrated, new_hidden_state).

        Call this sequentially as events arrive. Pass the returned state
        back on the next call. Reset to None for a new session.

        Example
        -------
        state = None
        for event in session_events:
            feat = featurizer.process_event(event.type, event.ts)
            p_frustrated, state = model.online_step(
                torch.tensor(feat).unsqueeze(0), state
            )
        """
        x = self.input_norm(feat_vec.unsqueeze(1))   # (1, 1, 9)
        out, new_state = self.lstm(x, state)          # out: (1, 1, hidden)
        logit = self.head(out[:, -1, :])              # (1, 1)
        p = float(torch.sigmoid(logit).item())
        return p, new_state
