"""Explainability service using existing SHAP results and optional TreeExplainer."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from services.data_loader import load_json_result
from services.model_loader import LoadedModel


def get_global_shap_summary() -> tuple[pd.DataFrame, int]:
    """Return global mean(|SHAP|) feature importance recorded in results/evaluation.json."""
    evaluation = load_json_result("evaluation.json")
    shap_data = evaluation.get("shap", {})
    top_features = shap_data.get("top_features", [])
    sample_rows = int(shap_data.get("sample_rows", 0))
    df = pd.DataFrame(top_features)
    if not df.empty:
        df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return df, sample_rows


def compute_local_shap(
    loaded: LoadedModel,
    feature_values: dict[str, float],
) -> tuple[pd.DataFrame | None, str | None]:
    """Compute transaction-level SHAP values when both the trained model and shap support it.

    Never fabricates SHAP values; returns (None, reason) if unavailable.
    """
    if not loaded.available or loaded.model is None:
        return None, (
            "Transaction-level SHAP explanations require the original trained model artifact "
            "(`model.joblib`). Showing recorded global SHAP results instead."
        )
    try:
        import shap  # type: ignore
    except ImportError:
        return None, "The `shap` package is not installed in this environment."

    try:
        estimator: Any = loaded.model
        if hasattr(estimator, "named_steps"):
            estimator = list(estimator.named_steps.values())[-1]
        explainer = shap.TreeExplainer(estimator)
        ordered = np.array(
            [[float(feature_values.get(name, 0.0)) for name in loaded.features]],
            dtype=np.float32,
        )
        raw_values = explainer.shap_values(ordered)
        if isinstance(raw_values, list):
            vals = np.asarray(raw_values[-1])[0]
        else:
            vals = np.asarray(raw_values)[0]
        df = pd.DataFrame(
            {
                "feature": loaded.features,
                "value": ordered[0],
                "shap_value": vals,
                "abs_shap": np.abs(vals),
            }
        ).sort_values("abs_shap", ascending=False)
        return df.reset_index(drop=True), None
    except Exception as exc:
        return None, f"Loaded model does not support TreeExplainer in this runtime: {exc}"
