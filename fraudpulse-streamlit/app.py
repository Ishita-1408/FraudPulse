"""FraudPulse — Streamlit Entry Point.

Intelligence Behind Every Transaction.
Integrates read-only with the existing ML project without modifying any original files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure fraudpulse-streamlit/ is on sys.path regardless of working directory
APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from components.sidebar import render_sidebar  # noqa: E402
from components.theme import inject_custom_css  # noqa: E402
from pages import analytics, explainability, live_detection, model_intelligence, overview  # noqa: E402
from services.model_loader import load_trained_model  # noqa: E402


def main() -> None:
    st.set_page_config(
        page_title="FraudPulse · Intelligence Behind Every Transaction",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()
    loaded_model = load_trained_model()
    active_page = render_sidebar(loaded_model)

    if active_page == "overview":
        overview.render()
    elif active_page == "live_detection":
        live_detection.render(loaded_model)
    elif active_page == "analytics":
        analytics.render()
    elif active_page == "explainability":
        explainability.render(loaded_model)
    elif active_page == "model_intelligence":
        model_intelligence.render()


if __name__ == "__main__":
    main()
