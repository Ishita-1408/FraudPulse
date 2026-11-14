"""Prediction service wrapping the existing trained model artifact."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from services.model_loader import LoadedModel


@dataclass(frozen=True)
class PredictionResult:
    """Structured inference output for a single transaction."""

    transaction_id: str
    amount: float
    probability: float
    threshold: float
    is_fraud: bool
    model_prediction: str
    latency_ms: float


def validate_features(feature_values: dict[str, float], expected_features: list[str]) -> np.ndarray:
    """Validate that all 29 expected features are present, finite, and in exact training order."""
    provided_keys = set(feature_values.keys())
    expected_keys = set(expected_features)

    missing = sorted(expected_keys - provided_keys)
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}")

    unexpected = sorted(provided_keys - expected_keys)
    if unexpected:
        raise ValueError(f"Unexpected features provided: {', '.join(unexpected)}")

    ordered_vals: list[float] = []
    for name in expected_features:
        raw = feature_values[name]
        try:
            val = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature '{name}' must be numeric, got {raw!r}") from exc
        if not math.isfinite(val):
            raise ValueError(f"Feature '{name}' must be finite, got {val!r}")
        ordered_vals.append(val)

    return np.array([ordered_vals], dtype=np.float32)


def run_prediction(
    loaded: LoadedModel,
    feature_values: dict[str, float],
    transaction_id: str,
) -> PredictionResult:
    """Execute real model inference using the exact original feature order and threshold."""
    if not loaded.available or loaded.model is None:
        raise RuntimeError("Trained model artifact is not available for inference.")

    ordered = validate_features(feature_values, loaded.features)

    t0 = time.perf_counter()
    proba = float(loaded.model.predict_proba(ordered)[:, 1][0])
    latency_ms = (time.perf_counter() - t0) * 1000.0

    is_fraud = proba >= loaded.threshold
    model_prediction = "Fraudulent" if is_fraud else "Legitimate"

    return PredictionResult(
        transaction_id=transaction_id,
        amount=float(feature_values["Amount"]),
        probability=proba,
        threshold=loaded.threshold,
        is_fraud=is_fraud,
        model_prediction=model_prediction,
        latency_ms=latency_ms,
    )
