"""Section 4: Explainable AI — Global & Transaction-Level SHAP."""

from __future__ import annotations

import streamlit as st

from components.charts import chart_shap_importance
from components.header import render_page_header
from components.kpi_cards import KpiItem, render_kpi_row
from components.theme import COLORS
from services.data_loader import FEATURES, get_figure_path
from services.explainability_service import compute_local_shap, get_global_shap_summary
from services.model_loader import LoadedModel


def render(loaded_model: LoadedModel) -> None:
    """Render the Explainable AI page."""
    render_page_header(
        title="Explainable AI (SHAP)",
        subtitle="Global feature attribution from the selected XGBoost model and transaction-level explanations when the artifact is present.",
        badge_text="SHAP Interpretability",
        badge_variant="cyan",
    )

    shap_df, sample_rows = get_global_shap_summary()

    if not shap_df.empty:
        top1 = shap_df.iloc[0]
        top2 = shap_df.iloc[1] if len(shap_df) > 1 else top1
        top3 = shap_df.iloc[2] if len(shap_df) > 2 else top1
        render_kpi_row(
            [
                KpiItem("Top Driver #1", str(top1["feature"]), f"mean(|SHAP|) = {float(top1['mean_abs_shap']):.4f}", COLORS["red"]),
                KpiItem("Top Driver #2", str(top2["feature"]), f"mean(|SHAP|) = {float(top2['mean_abs_shap']):.4f}", COLORS["amber"]),
                KpiItem("Top Driver #3", str(top3["feature"]), f"mean(|SHAP|) = {float(top3['mean_abs_shap']):.4f}", COLORS["cyan"]),
                KpiItem("SHAP Sample Size", f"{sample_rows:,}", "Validation/test sample rows", COLORS["blue"]),
            ]
        )

    st.markdown("<div style='height:0.85rem;'></div>", unsafe_allow_html=True)
    st.info(
        "**Feature Privacy Note:** `V1–V28` are anonymised Principal Component Analysis (PCA) components "
        "derived from confidential cardholder fields in the OpenML 1597 dataset; only `Amount` is in its "
        "original monetary unit. SHAP quantifies how much each PCA component shifts the log-odds of fraud."
    )

    c1, c2 = st.columns([1.15, 1.0])
    with c1:
        if not shap_df.empty:
            st.plotly_chart(
                chart_shap_importance(shap_df, "Global Top Contributing Features (mean |SHAP|)"),
                use_container_width=True,
            )
        else:
            st.warning("No global SHAP summary found in `results/evaluation.json`.")

    with c2:
        shap_fig = get_figure_path("shap_summary.png")
        if shap_fig:
            st.image(
                str(shap_fig),
                caption="Original Global SHAP Beeswarm Summary (results/figures/shap_summary.png)",
                use_container_width=True,
            )
        if not shap_df.empty:
            st.dataframe(shap_df, use_container_width=True, hide_index=True)

    # Transaction-level SHAP
    st.markdown("<div class='fp-section-title'>Transaction-Level SHAP Attribution</div>", unsafe_allow_html=True)
    tx_inputs = st.session_state.get("fp_tx_inputs", {f: 0.0 for f in FEATURES})
    local_df, reason = compute_local_shap(loaded_model, tx_inputs)

    if local_df is not None:
        st.success("Computed live TreeExplainer SHAP values for the current transaction inputs from **Live Fraud Detection**.")
        st.plotly_chart(
            chart_shap_importance(local_df.head(12), "Top 12 Transaction-Level SHAP Contributions", value_col="shap_value"),
            use_container_width=True,
        )
        st.dataframe(local_df.head(15), use_container_width=True, hide_index=True)
    else:
        st.warning(reason or "Transaction-level SHAP is unavailable.")
