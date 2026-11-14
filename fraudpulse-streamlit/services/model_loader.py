"""Read-only loader for existing trained model artifacts (never trains or substitutes models)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import streamlit as st

from services.data_loader import FEATURES, load_json_result
from services.paths import APP_DIR, REPO_ROOT


@dataclass
class LoadedModel:
    """Container for an existing trained model artifact and its metadata."""

    available: bool
    model: Any | None
    features: list[str]
    threshold: float
    model_name: str
    strategy: str
    artifact_path: Path | None
    meta_path: Path | None
    message: str


def _candidate_dirs() -> list[Path]:
    dirs: list[Path] = []
    for env_var in ("FRAUDPULSE_MODEL_DIR", "MODEL_DIR"):
        val = os.getenv(env_var)
        if val:
            dirs.append(Path(val))
    dirs.extend(
        [
            REPO_ROOT / "artifacts",
            REPO_ROOT / "models",
            REPO_ROOT,
            APP_DIR / "artifacts",
        ]
    )
    return dirs


@st.cache_resource(show_spinner=False)
def load_trained_model() -> LoadedModel:
    """Locate and load the existing trained model artifact if present.

    Strictly read-only: never trains, downloads, or fabricates a fallback model.
    """
    baselines = load_json_result("baselines.json")
    best = baselines.get("best", {})
    default_threshold = float(best.get("threshold", 0.9357))
    default_name = str(best.get("model", "xgboost"))
    default_strategy = str(best.get("strategy", "weights"))

    for directory in _candidate_dirs():
        model_path = directory / "model.joblib"
        if model_path.exists():
            meta_path = directory / "model_meta.json"
            features = list(FEATURES)
            threshold = default_threshold
            model_name = default_name
            strategy = default_strategy
            if meta_path.exists():
                with meta_path.open("r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                features = list(meta.get("features", FEATURES))
                threshold = float(meta.get("threshold", default_threshold))
                model_name = str(meta.get("model", default_name))
                strategy = str(meta.get("strategy", default_strategy))
            try:
                model = joblib.load(model_path)
            except Exception as exc:  # pragma: no cover - defensive load guard
                return LoadedModel(
                    available=False,
                    model=None,
                    features=features,
                    threshold=threshold,
                    model_name=model_name,
                    strategy=strategy,
                    artifact_path=model_path,
                    meta_path=meta_path if meta_path.exists() else None,
                    message=f"Found artifact at `{model_path}`, but failed to load it: {exc}",
                )
            return LoadedModel(
                available=True,
                model=model,
                features=features,
                threshold=threshold,
                model_name=model_name,
                strategy=strategy,
                artifact_path=model_path,
                meta_path=meta_path if meta_path.exists() else None,
                message=f"Loaded existing trained artifact from `{model_path.name}`.",
            )

    return LoadedModel(
        available=False,
        model=None,
        features=list(FEATURES),
        threshold=default_threshold,
        model_name=default_name,
        strategy=default_strategy,
        artifact_path=None,
        meta_path=None,
        message=(
            "Trained model artifact (`model.joblib` + optional `model_meta.json`) was not found "
            "in the repository. Live inference is disabled until the original trained artifact is supplied."
        ),
    )
