"""Cached loaders for existing experiment JSON results, figures, and optional dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from services.paths import DATA_DIR, FIGURES_DIR, RESULTS_DIR

FEATURES: list[str] = [f"V{i}" for i in range(1, 29)] + ["Amount"]


@st.cache_data(show_spinner=False)
def load_json_result(filename: str) -> dict[str, Any]:
    """Load a JSON file from the original repository's results/ directory."""
    path: Path = RESULTS_DIR / filename
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_all_results() -> dict[str, dict[str, Any]]:
    """Load all experiment JSON files used across FraudPulse."""
    files = [
        "split.json",
        "baselines.json",
        "evaluation.json",
        "privacy.json",
        "efficiency.json",
        "drift.json",
        "streaming_burst.json",
        "streaming_paced.json",
        "producer.json",
        "pubsub_publish.json",
        "dataflow_scoring_1worker.json",
        "dataflow_scoring_2workers.json",
        "vertex_run.json",
    ]
    return {name.replace(".json", ""): load_json_result(name) for name in files}


def get_figure_path(filename: str) -> Path | None:
    """Return path to an existing figure in results/figures/ if present."""
    path = FIGURES_DIR / filename
    return path if path.exists() else None


def dataset_available() -> bool:
    """Check whether the original creditcard.parquet dataset exists locally."""
    return (DATA_DIR / "creditcard.parquet").exists()


@st.cache_data(show_spinner=False)
def load_test_samples(max_rows: int = 200) -> pd.DataFrame | None:
    """Load a bounded slice of the test window if data/creditcard.parquet exists.

    Never downloads the dataset automatically; returns None when absent.
    """
    parquet_path = DATA_DIR / "creditcard.parquet"
    if not parquet_path.exists():
        return None
    df = pd.read_parquet(parquet_path)
    if "seq" not in df.columns:
        df["seq"] = range(len(df))
    df = df.sort_values("seq", kind="stable").reset_index(drop=True)
    # Match fraud/data.py time-ordered 70/10/20 split
    i_val = int(len(df) * 0.8)
    test_df = df.iloc[i_val:].copy().reset_index(drop=True)
    if len(test_df) <= max_rows:
        return test_df
    # Include frauds + legitimate transactions from the test window for inspection
    frauds = test_df[test_df["Class"] == 1]
    legit = test_df[test_df["Class"] == 0].head(max(0, max_rows - len(frauds)))
    combined = pd.concat([frauds, legit], ignore_index=True)
    return combined.sort_values("seq", kind="stable").reset_index(drop=True)
