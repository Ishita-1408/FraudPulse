"""Section 3: Transaction Analytics — Model Comparison, Curves, Cost & Calibration."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.charts import (
    chart_confusion_matrix,
    chart_cost_curve,
    chart_reliability,
    chart_split_distribution,
    chart_supervised_comparison,
)
from components.header import render_page_header
from components.kpi_cards import KpiItem, render_kpi_row
from components.theme import COLORS
from services.data_loader import get_figure_path, load_all_results


def render() -> None:
    """Render the Transaction Analytics page."""
    results = load_all_results()
    baselines = results.get("baselines", {})
    evaluation = results.get("evaluation", {})
    split = results.get("split", {})

    render_page_header(
        title="Transaction Analytics",
        subtitle="Supervised model comparison, PR/ROC analysis, confusion matrices, cost-sensitive thresholds, and calibration reliability.",
        badge_text="Test Window (56,962 Tx · 75 Frauds)",
        badge_variant="blue",
    )

    rows_df = pd.DataFrame(baselines.get("rows", []))
    if rows_df.empty:
        st.error("`results/baselines.json` could not be loaded.")
        return

    # Interactive Filters
    f1, f2, f3 = st.columns(3)
    with f1:
        selected_models = st.multiselect(
            "Filter by Model Family",
            options=sorted(rows_df["model"].unique().tolist()),
            default=sorted(rows_df["model"].unique().tolist()),
        )
    with f2:
        selected_strategies = st.multiselect(
            "Filter by Imbalance Strategy",
            options=["none", "weights", "smote"],
            default=["none", "weights", "smote"],
        )
    with f3:
        metric_choice = st.selectbox(
            "Comparison Metric",
            options=[
                "pr_auc",
                "roc_auc",
                "f1",
                "precision",
                "recall",
                "recall_at_fpr_0.1pct",
                "recall_at_fpr_1pct",
                "cost_per_10k",
                "fit_seconds",
            ],
            index=0,
        )

    filtered = rows_df[
        rows_df["model"].isin(selected_models) & rows_df["strategy"].isin(selected_strategies)
    ].copy()

    if filtered.empty:
        st.warning("No configurations match the selected filters.")
        return

    st.plotly_chart(chart_supervised_comparison(filtered, metric_choice), use_container_width=True)

    # Confusion Matrix & Threshold Analysis
    st.markdown("<div class='fp-section-title'>Confusion Matrix & Operating Threshold Analysis</div>", unsafe_allow_html=True)
    config_labels = [f"{r['model']} / {r['strategy']}" for _, r in filtered.iterrows()]
    chosen_cfg = st.selectbox("Inspect Configuration on Test Window", config_labels, index=0)
    chosen_row = filtered.iloc[config_labels.index(chosen_cfg)]

    test_total = int(split.get("test", {}).get("rows", 56962))
    tp = int(chosen_row["tp"])
    fp = int(chosen_row["fp"])
    fn = int(chosen_row["fn"])
    tn = test_total - tp - fp - fn

    cm_col, stat_col = st.columns([1.05, 1.15])
    with cm_col:
        st.plotly_chart(
            chart_confusion_matrix(tp, fp, fn, tn, f"Confusion Matrix — {chosen_cfg}"),
            use_container_width=True,
        )
    with stat_col:
        render_kpi_row(
            [
                KpiItem("Max-F1 Threshold", f"{float(chosen_row['threshold']):.4f}", "Selected on validation", COLORS["cyan"]),
                KpiItem("Flagged Alerts", f"{int(chosen_row['flagged'])}", f"TP={tp}, FP={fp}", COLORS["amber"]),
                KpiItem("Cost-Opt Threshold", f"{float(chosen_row['cost_threshold']):.6f}", f"${float(chosen_row['cost_per_10k']):.2f} / 10k tx", COLORS["green"]),
            ]
        )
        st.markdown("<div style='height:0.65rem;'></div>", unsafe_allow_html=True)
        st.dataframe(
            filtered[
                [
                    "model",
                    "strategy",
                    "pr_auc",
                    "roc_auc",
                    "precision",
                    "recall",
                    "f1",
                    "threshold",
                    "cost_per_10k",
                ]
            ].sort_values("pr_auc", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    # PR / ROC Curves & Fraud Distribution
    st.markdown("<div class='fp-section-title'>Precision-Recall Curves & Fraud Distribution</div>", unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        pr_fig = get_figure_path("pr_curves.png")
        if pr_fig:
            st.image(str(pr_fig), caption="Recorded Precision-Recall & ROC Curves (results/figures/pr_curves.png)", use_container_width=True)
        else:
            st.info("`results/figures/pr_curves.png` not found.")
    with p2:
        st.plotly_chart(chart_split_distribution(split), use_container_width=True)

    # Cost-Sensitive & Calibration
    st.markdown("<div class='fp-section-title'>Cost-Sensitive Evaluation & Calibration Reliability</div>", unsafe_allow_html=True)
    cost_data = evaluation.get("cost", {})
    cal_data = evaluation.get("calibration", {})

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_cost_curve(cost_data), use_container_width=True)
        st.caption(
            f"Cost Model: Missed Fraud = ${cost_data.get('cost_model', {}).get('missed_fraud_cost', 200):.0f} · "
            f"Manual Review = ${cost_data.get('cost_model', {}).get('review_cost', 5):.0f} · "
            f"Flag Nothing = ${cost_data.get('flag_nothing_cost_per_10k', 0):,.2f}/10k."
        )
        cost_fig = get_figure_path("cost_curve.png")
        if cost_fig:
            with st.expander("View Original Cost Curve Figure (cost_curve.png)"):
                st.image(str(cost_fig), use_container_width=True)

    with c2:
        st.plotly_chart(chart_reliability(cal_data), use_container_width=True)
        st.caption(
            f"Method: {cal_data.get('method', 'isotonic')} · "
            f"Brier Raw = `{cal_data.get('brier_raw', 0):.6f}` → "
            f"Calibrated = `{cal_data.get('brier_calibrated', 0):.6f}`."
        )
        rel_fig = get_figure_path("reliability.png")
        if rel_fig:
            with st.expander("View Original Reliability Figure (reliability.png)"):
                st.image(str(rel_fig), use_container_width=True)
