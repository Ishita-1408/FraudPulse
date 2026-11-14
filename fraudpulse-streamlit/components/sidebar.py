"""Branded sidebar navigation and runtime status indicators."""

from __future__ import annotations

import streamlit as st

from services.data_loader import dataset_available
from services.model_loader import LoadedModel
from services.paths import ASSETS_DIR

NAV_ITEMS: list[tuple[str, str]] = [
    ("overview", "Executive Overview"),
    ("live_detection", "Live Fraud Detection"),
    ("analytics", "Transaction Analytics"),
    ("explainability", "Explainable AI"),
    ("model_intelligence", "Model Intelligence"),
]


def render_sidebar(loaded_model: LoadedModel) -> str:
    """Render the FraudPulse navigation sidebar and return the active page key."""
    with st.sidebar:
        logo_path = ASSETS_DIR / "logo.png"
        col_logo, col_brand = st.columns([1, 3], vertical_alignment="center")
        with col_logo:
            if logo_path.exists():
                st.image(str(logo_path), width=52)
        with col_brand:
            st.markdown(
                """
                <div style="line-height:1.15;">
                    <div style="font-size:1.25rem;font-weight:700;color:#F8FAFC;letter-spacing:-0.01em;">
                        Fraud<span style="color:#06B6D4;">Pulse</span>
                    </div>
                    <div style="font-size:0.72rem;color:#94A3B8;margin-top:2px;">
                        Intelligence Behind Every Transaction.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:0.65rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.70rem;font-weight:600;letter-spacing:0.08em;"
            "text-transform:uppercase;color:#64748B;margin-bottom:0.35rem;'>Platform Navigation</div>",
            unsafe_allow_html=True,
        )

        labels = [label for _, label in NAV_ITEMS]
        keys = [key for key, _ in NAV_ITEMS]
        current_key = st.session_state.get("fp_nav", "overview")
        default_idx = keys.index(current_key) if current_key in keys else 0

        selected_label = st.radio(
            "Navigation",
            labels,
            index=default_idx,
            label_visibility="collapsed",
        )
        selected_key = keys[labels.index(selected_label)]
        st.session_state["fp_nav"] = selected_key

        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.70rem;font-weight:600;letter-spacing:0.08em;"
            "text-transform:uppercase;color:#64748B;margin-bottom:0.45rem;'>Runtime Status</div>",
            unsafe_allow_html=True,
        )

        model_badge = (
            "<span class='fp-badge fp-badge-green'>Model Artifact: Loaded</span>"
            if loaded_model.available
            else "<span class='fp-badge fp-badge-amber'>Model Artifact: Not Supplied</span>"
        )
        data_badge = (
            "<span class='fp-badge fp-badge-green'>Dataset: Local Parquet</span>"
            if dataset_available()
            else "<span class='fp-badge fp-badge-blue'>Dataset: Experiment JSONs</span>"
        )
        st.markdown(
            f"""
            <div style="display:flex;flex-direction:column;gap:0.45rem;">
                {model_badge}
                {data_badge}
                <span class="fp-badge fp-badge-cyan">Selected: XGBoost ({loaded_model.strategy})</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="margin-top:1.25rem;padding:0.75rem;border-radius:10px;
                        background:rgba(15,23,42,0.75);border:1px solid rgba(148,163,184,0.14);
                        font-size:0.74rem;color:#94A3B8;line-height:1.45;">
                <strong style="color:#E2E8F0;">Read-Only Integration</strong><br/>
                Reads original <code>results/</code> metrics and existing model artifacts without altering ML code.
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected_key
