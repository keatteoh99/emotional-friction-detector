"""
LightGBM churn risk model: P(churn | frustrated, user_profile).

Two-stage design:
  Stage 1 (LSTM):  P(frustrated | session_events)          → p_frustrated
  Stage 2 (LGBM):  P(churn | p_frustrated, user_features)  → p_churn

The LGBM model sees LSTM output as one of its features, enabling it to
condition on real session behaviour without processing raw sequences.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd


FEATURE_COLS = [
    # LSTM output (stage-1 signal)
    "p_frustrated",
    # Cross-session user features
    "ltv_estimate_myr",
    "tenure_days",
    "lifetime_order_count",
    "avg_order_value_myr",
    "churn_sensitivity",
    "is_post_complaint_return",
    "prior_delays_30d",
    "frustration_rate",
    "support_contact_rate",
    "cancellation_rate",
    "avg_delay_minutes",
    "p90_delay_minutes",
    "is_repeat_frustrated",
    "ltv_x_frustration_rate",
    # Current session context
    "delay_minutes",
    "base_eta_minutes",
    "eta_refresh_count",
    "tap_interval_cv",
    "bg_fg_cycle_rate_per_min",
    "support_latency_ratio",
    "anxiety_event_rate_per_min",
]


class ChurnRiskModel:
    def __init__(self, params: Optional[dict] = None):
        self.params = params or {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 50,
            "feature_fraction": 0.80,
            "bagging_fraction": 0.80,
            "bagging_freq": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbose": -1,
        }
        self.booster: Optional[lgb.Booster] = None
        self.feature_importance_: Optional[pd.Series] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
    ) -> None:
        train_data = lgb.Dataset(X_train[FEATURE_COLS], label=y_train)
        val_data = lgb.Dataset(X_val[FEATURE_COLS], label=y_val, reference=train_data)

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=50),
        ]

        self.booster = lgb.train(
            self.params,
            train_set=train_data,
            valid_sets=[val_data],
            num_boost_round=num_boost_round,
            callbacks=callbacks,
        )

        importance = self.booster.feature_importance(importance_type="gain")
        self.feature_importance_ = pd.Series(
            importance, index=FEATURE_COLS
        ).sort_values(ascending=False)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.booster.predict(X[FEATURE_COLS])

    def save(self, path: str) -> None:
        if self.booster is None:
            raise RuntimeError("Nothing to save — model not fitted.")
        self.booster.save_model(path)

    @classmethod
    def load(cls, path: str) -> "ChurnRiskModel":
        model = cls()
        model.booster = lgb.Booster(model_file=path)
        return model
