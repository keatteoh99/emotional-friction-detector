"""
SHAP explainability for the LightGBM churn risk model.

Produces three artefacts:
  1. Global feature importance (SHAP mean |value|)
  2. Beeswarm summary plot (for notebooks / stakeholder decks)
  3. Per-prediction SHAP breakdown (for the API explanation endpoint)

Why SHAP over built-in LightGBM importance:
  LightGBM gain/split importance double-counts correlated features.
  SHAP is additive and consistent — important for the LTV × frustration_rate
  interaction feature which has high built-in importance but only because
  it absorbs variance from its constituent features.

Usage:
  python -m models.lgbm_churn_risk.shap_analysis \
      --model  models/lgbm_churn_risk/artefacts/churn_model.lgb \
      --scored data/processed/sessions_scored.parquet \
      --output models/lgbm_churn_risk/artefacts/shap/
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from .model import ChurnRiskModel, FEATURE_COLS


def compute_shap_values(
    model: ChurnRiskModel,
    X: pd.DataFrame,
    n_background: int = 500,
) -> Optional[np.ndarray]:
    """
    Compute SHAP values for X using TreeExplainer.

    Returns (N, F) array of SHAP values, or None if shap not installed.
    """
    if not SHAP_AVAILABLE:
        print("shap package not installed. Run: pip install shap")
        return None
    if model.booster is None:
        raise RuntimeError("Model not fitted.")

    explainer = shap.TreeExplainer(model.booster)
    sv = explainer.shap_values(X[FEATURE_COLS])
    # Binary classifier returns list [neg_class, pos_class]; take positive class
    return sv[1] if isinstance(sv, list) else sv


def global_importance(shap_values: np.ndarray, feature_names: list) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, sorted descending."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def explain_prediction(
    model: ChurnRiskModel,
    row: pd.DataFrame,
) -> pd.Series:
    """
    Return a Series of SHAP values for a single prediction.
    Useful for the API /explain endpoint.
    """
    if not SHAP_AVAILABLE:
        return pd.Series(dtype=float)
    explainer = shap.TreeExplainer(model.booster)
    sv = explainer.shap_values(row[FEATURE_COLS])
    vals = sv[1][0] if isinstance(sv, list) else sv[0]
    return pd.Series(vals, index=FEATURE_COLS).sort_values(key=abs, ascending=False)


def run_analysis(
    model_path: str = "models/lgbm_churn_risk/artefacts/churn_model.lgb",
    scored_path: str = "data/processed/sessions_scored.parquet",
    output_dir: str = "models/lgbm_churn_risk/artefacts/shap",
    n_sample: int = 5000,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = ChurnRiskModel.load(model_path)
    df = pd.read_parquet(scored_path)
    frustrated = df[df.get("is_frustrated", pd.Series(True, index=df.index))].copy()

    for col in FEATURE_COLS:
        if col not in frustrated.columns:
            frustrated[col] = 0.0

    sample = frustrated.sample(min(n_sample, len(frustrated)), random_state=42)
    shap_values = compute_shap_values(model, sample)
    if shap_values is None:
        return

    importance = global_importance(shap_values, FEATURE_COLS)
    importance.to_csv(out / "feature_importance_shap.csv", index=False)
    print(f"Saved SHAP importance -> {out}/feature_importance_shap.csv")
    print(f"\nTop-10 features by mean |SHAP|:\n{importance.head(10).to_string(index=False)}")

    # Beeswarm plot (requires matplotlib)
    try:
        import io, sys
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Swallow shap's console output (contains non-ASCII chars that crash cp1252)
        _orig_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            shap.summary_plot(shap_values, sample[FEATURE_COLS], show=False)
        finally:
            sys.stdout = _orig_stdout
        plt.tight_layout()
        plt.savefig(out / "beeswarm.png", dpi=120)
        plt.close()
        print(f"Saved beeswarm -> {out}/beeswarm.png")
    except Exception as e:
        print(f"Could not save beeswarm plot: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="models/lgbm_churn_risk/artefacts/churn_model.lgb")
    parser.add_argument("--scored", default="data/processed/sessions_scored.parquet")
    parser.add_argument("--output", default="models/lgbm_churn_risk/artefacts/shap")
    args = parser.parse_args()
    run_analysis(args.model, args.scored, args.output)
