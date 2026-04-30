"""
Evaluation metrics for both model stages and the intervention layer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


def classification_report_dict(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_positive": int(y_true.sum()),
        "n_total": len(y_true),
    }


def precision_at_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.60,
) -> float:
    """Return precision at the threshold closest to target_recall."""
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    idx = np.argmin(np.abs(recalls - target_recall))
    return float(precisions[idx])


def intervention_lift(
    treatment_df: pd.DataFrame,
    control_df: pd.DataFrame,
    churn_col: str = "churned_7d",
) -> dict:
    """
    Compute lift of intervention vs holdback control.

    Both DataFrames must contain `churn_col` (binary, 1=churned).
    Returns absolute and relative reduction in churn rate.
    """
    treatment_rate = treatment_df[churn_col].mean()
    control_rate = control_df[churn_col].mean()
    absolute_reduction = control_rate - treatment_rate
    relative_reduction = absolute_reduction / control_rate if control_rate > 0 else 0.0

    return {
        "treatment_churn_rate": round(treatment_rate, 4),
        "control_churn_rate": round(control_rate, 4),
        "absolute_reduction": round(absolute_reduction, 4),
        "relative_reduction": round(relative_reduction, 4),
        "n_treatment": len(treatment_df),
        "n_control": len(control_df),
    }
