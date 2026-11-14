"""Reliable path resolution anchored to this file's location."""

from __future__ import annotations

from pathlib import Path

APP_DIR: Path = Path(__file__).resolve().parent.parent
REPO_ROOT: Path = APP_DIR.parent
RESULTS_DIR: Path = REPO_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
ASSETS_DIR: Path = APP_DIR / "assets"
DATA_DIR: Path = REPO_ROOT / "data"
