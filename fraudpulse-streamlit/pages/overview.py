"""Section 1: Executive Overview dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.charts import (
    chart_best_model_metrics,
    chart_fraud_prevalence,
    chart_split_distribution,
)
from components.header import render_page_header
from components.kpi_cards import KpiItem, render_kpi_row
from components.theme import COLORS
from services.data_loader import get_figure_path, load_all_results


def render() -> None:
    """Render the Executive Overview landing page."""
    results = load_all_results()
    split = results.get("split", {})
    baselines = results.get("baselines", {})
    best = baselines.get("best", {})

    render_page_header(
        title="Executive Overview",
        subtitle="FraudPulse · Intelligence Behind Every Transaction — Historical Benchmark & System Summary",
        badge_text="Historical Experiment Results",
        badge_variant="cyan",
    )

    train_s = split.get("train", {})
    val_s = split.get("val", {})
    test_s = split.get("test", {})

    total_tx = int(train_s.get("rows", 0)) + int(val_s.get("rows", 0)) + int(test_s.get("rows", 0))
    total_fraud = int(train_s.get("frauds", 0)) + int(val_s.get("frauds", 0)) + int(test_s.get("frauds", 0))
    total_legit = total_tx - total_fraud
    prevalence_pct = (total_fraud / total_tx * 100.0) if total_tx else 0.0
    test_rows = int(test_s.get("rows", 0))
    test_frauds = int(test_s.get("frauds", 0))

    st.markdown("<div class='fp-section-title'>Dataset Telemetry (Time-Ordered 70 / 10 / 20 Split)</div>", unsafe_allow_html=True)
    render_kpi_row(
        [
            KpiItem("Total Transactions", f"{total_tx:,}", "OpenML 1597 creditcard", COLORS["blue"]),
            KpiItem("Fraudulent Transactions", f"{total_fraud:,}", f"Prevalence: {prevalence_pct:.4f}%", COLORS["red"]),
            KpiItem("Legitimate Transactions", f"{total_legit:,}", "99.8273% of volume", COLORS["green"]),
            KpiItem("Test Window Rows", f"{test_rows:,}", "Last 20% in chronological order", COLORS["cyan"]),
            KpiItem("Test Fraud Cases", f"{test_frauds:,}", f"Test rate: {float(test_s.get('fraud_rate', 0)) * 100:.4f}%", COLORS["amber"]),
        ]
    )

    st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='fp-section-title'>Selected Champion Model — {best.get('model', 'xgboost').upper()} "
        f"({best.get('strategy', 'weights')})</div>",
        unsafe_allow_html=True,
    )
    render_kpi_row(
        [
            KpiItem("PR-AUC (Test)", f"{float(best.get('pr_auc', 0)):.4f}", f"Val PR-AUC: {float(best.get('val_pr_auc', 0)):.4f}", COLORS["cyan"]),
            KpiItem("ROC-AUC (Test)", f"{float(best.get('roc_auc', 0)):.4f}", "Area under ROC curve", COLORS["blue"]),
            KpiItem("Precision", f"{float(best.get('precision', 0)):.4f}", f"TP={best.get('tp', 0)} / Flagged={best.get('flagged', 0)}", COLORS["green"]),
            KpiItem("Recall", f"{float(best.get('recall', 0)):.4f}", f"FN={best.get('fn', 0)} missed", COLORS["amber"]),
            KpiItem("F1-Score", f"{float(best.get('f1', 0)):.4f}", f"Val threshold: {float(best.get('threshold', 0)):.4f}", COLORS["blue"]),
        ]
    )

    st.markdown("<div style='height:0.65rem;'></div>", unsafe_allow_html=True)
    render_kpi_row(
        [
            KpiItem("Recall @ 0.1% FPR", f"{float(best.get('recall_at_fpr_0.1pct', 0)):.4f}", "Ultra-low false-positive operating point", COLORS["cyan"]),
            KpiItem("Recall @ 1.0% FPR", f"{float(best.get('recall_at_fpr_1pct', 0)):.4f}", "1% false-positive tolerance", COLORS["blue"]),
            KpiItem("Cost per 10k Tx", f"${float(best.get('cost_per_10k', 0)):.2f}", "Missed=$200, Review=$5", COLORS["green"]),
            KpiItem("Best Iteration", f"{int(best.get('best_iteration', 0))}", f"Fit time: {float(best.get('fit_seconds', 0)):.2f}s", COLORS["amber"]),
        ]
    )

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_split_distribution(split), use_container_width=True)
    with c2:
        st.plotly_chart(chart_fraud_prevalence(split), use_container_width=True)

    c3, c4 = st.columns([1.15, 1.0])
    with c3:
        st.plotly_chart(chart_best_model_metrics(best), use_container_width=True)
    with c4:
        st.markdown("<div class='fp-section-title'>Recent Interactive Session Activity</div>", unsafe_allow_html=True)
        history = st.session_state.get("fp_history", [])
        if history:
            st.caption("Live predictions executed in this browser session (separate from historical benchmarks above).")
            hist_df = pd.DataFrame(history)[
                ["transaction_id", "amount", "fraud_probability_pct", "prediction", "latency_ms"]
            ]
            st.dataframe(hist_df.head(8), use_container_width=True, hide_index=True)
        else:
            st.info(
                "No interactive predictions have been run in this session yet. "
                "Open **Live Fraud Detection** in the sidebar to inspect or submit a transaction."
            )

    st.markdown("<div class='fp-section-title'>End-to-End Fraud Detection Architecture</div>", unsafe_allow_html=True)
    arch_col1, arch_col2 = st.columns([1.4, 1.0])
    with arch_col1:
        arch_fig = get_figure_path("architecture.png")
        if arch_fig:
            st.image(str(arch_fig), caption="Original Repository Architecture Diagram (results/figures/architecture.png)", use_container_width=True)
        else:
            st.warning("Architecture figure (`results/figures/architecture.png`) not found.")
    with arch_col2:
        st.markdown(
            """
            <div class="fp-card">
                <div style="font-weight:600;color:#F8FAFC;margin-bottom:0.5rem;">System Pipeline Layers</div>
                <ul style="color:#94A3B8;font-size:0.88rem;line-height:1.65;padding-left:1.15rem;margin:0;">
                    <li><strong style="color:#E2E8F0;">Ingestion & Time Split:</strong> OpenML 1597 (284,807 rows) split chronologically 70% / 10% / 20% with zero shuffling.</li>
                    <li><strong style="color:#E2E8F0;">Supervised & Anomaly Core:</strong> 12 baseline configs + Isolation Forest & Autoencoder benchmarks.</li>
                    <li><strong style="color:#E2E8F0;">Calibration & Cost Gate:</strong> Isotonic calibration on validation + $200 missed-fraud / $5 review cost curve.</li>
                    <li><strong style="color:#E2E8F0;">Streaming & Cloud:</strong> Kafka → Spark → ONNX → Parquet, Pub/Sub → Dataflow → BigQuery, and Vertex AI lifecycle.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
