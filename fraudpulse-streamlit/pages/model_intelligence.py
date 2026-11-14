"""Section 5: Model Intelligence — Subsections A through G."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.charts import (
    chart_drift_psi,
    chart_efficiency_latency,
    chart_privacy_tradeoff,
    chart_supervised_comparison,
)
from components.header import render_page_header
from components.kpi_cards import KpiItem, render_kpi_row
from components.theme import COLORS
from services.data_loader import get_figure_path, load_all_results


def render() -> None:
    """Render the Model Intelligence page covering subsections A–G."""
    results = load_all_results()

    render_page_header(
        title="Model Intelligence",
        subtitle="Comprehensive benchmarks across Supervised ML, Anomaly Detection, DP-SGD Privacy, ONNX/Core ML Efficiency, PSI Drift, Streaming, and Vertex AI.",
        badge_text="7 Engineering Subsystems (A–G)",
        badge_variant="blue",
    )

    tabs = st.tabs(
        [
            "A. Supervised ML",
            "B. Anomaly Detection",
            "C. Privacy (DP-SGD)",
            "D. Model Efficiency",
            "E. Model Drift (PSI)",
            "F. Streaming",
            "G. Vertex AI",
        ]
    )

    # A. Supervised ML
    with tabs[0]:
        baselines = results.get("baselines", {})
        df = pd.DataFrame(baselines.get("rows", []))
        st.markdown("#### 12 Supervised Configurations (4 Model Families × 3 Imbalance Strategies)")
        if not df.empty:
            metric = st.selectbox(
                "Select Metric to Plot",
                ["pr_auc", "roc_auc", "precision", "recall", "f1", "fit_seconds"],
                key="mi_sup_metric",
            )
            st.plotly_chart(chart_supervised_comparison(df, metric), use_container_width=True)
            st.dataframe(
                df[
                    [
                        "model",
                        "strategy",
                        "pr_auc",
                        "roc_auc",
                        "precision",
                        "recall",
                        "f1",
                        "fit_seconds",
                        "threshold",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    # B. Anomaly Detection
    with tabs[1]:
        anomaly = results.get("evaluation", {}).get("anomaly", {})
        iso = anomaly.get("isolation_forest", {})
        ae = anomaly.get("autoencoder", {})
        sup_ref = anomaly.get("best_supervised_for_reference", {})
        st.markdown("#### Unsupervised Anomaly Detection vs Best Supervised Baseline")
        render_kpi_row(
            [
                KpiItem("Isolation Forest PR-AUC", f"{float(iso.get('pr_auc', 0)):.4f}", f"ROC-AUC: {float(iso.get('roc_auc', 0)):.4f}", COLORS["amber"]),
                KpiItem("Autoencoder PR-AUC", f"{float(ae.get('pr_auc', 0)):.4f}", f"ROC-AUC: {float(ae.get('roc_auc', 0)):.4f}", COLORS["cyan"]),
                KpiItem("Best Supervised PR-AUC", f"{float(sup_ref.get('pr_auc', 0)):.4f}", f"Recall@0.1% FPR: {float(sup_ref.get('recall_at_fpr_0.1pct', 0)):.4f}", COLORS["green"]),
            ]
        )
        st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
        comp_df = pd.DataFrame(
            [
                {"Model": "Isolation Forest", **iso},
                {"Model": "Autoencoder (20 epochs)", **ae},
            ]
        )
        st.dataframe(
            comp_df[["Model", "pr_auc", "roc_auc", "precision", "recall", "f1", "recall_at_fpr_0.1pct", "cost_per_10k", "fit_seconds"]],
            use_container_width=True,
            hide_index=True,
        )

    # C. Privacy-Preserving ML
    with tabs[2]:
        privacy = results.get("privacy", {})
        st.markdown("#### Differential Privacy (DP-SGD via Opacus)")
        st.info(
            f"**Scope Note:** {privacy.get('mechanism', 'DP-SGD')} applies specifically to the "
            f"**{privacy.get('model', 'MLP 30-64-32-1')}** trained for {privacy.get('epochs', 8)} epochs — "
            "not to the tree-based models (XGBoost, LightGBM, Random Forest)."
        )
        priv_df = pd.DataFrame(privacy.get("rows", []))
        if not priv_df.empty:
            priv_df["epsilon_label"] = priv_df["epsilon_target"].astype(str)
            c1, c2 = st.columns([1.2, 1.0])
            with c1:
                st.plotly_chart(chart_privacy_tradeoff(priv_df), use_container_width=True)
            with c2:
                priv_fig = get_figure_path("privacy_tradeoff.png")
                if priv_fig:
                    st.image(str(priv_fig), caption="Original Privacy Trade-off Figure (privacy_tradeoff.png)", use_container_width=True)
            st.dataframe(
                priv_df[["epsilon_label", "epsilon_spent", "noise_multiplier", "pr_auc", "roc_auc", "precision", "recall", "f1", "fit_seconds"]],
                use_container_width=True,
                hide_index=True,
            )

    # D. Model Efficiency
    with tabs[3]:
        eff = results.get("efficiency", {})
        st.markdown("#### Edge & Production Inference Latency Benchmarks")
        st.caption(
            f"Hardware: `{eff.get('hardware', 'N/A')}` · Threads: `{eff.get('threads', 1)}` · "
            f"Budget: `{eff.get('budget', 'p95 <= 10 ms')}`"
        )
        eff_df = pd.DataFrame(eff.get("rows", []))
        if not eff_df.empty:
            st.plotly_chart(chart_efficiency_latency(eff_df), use_container_width=True)
            st.dataframe(eff_df, use_container_width=True, hide_index=True)

    # E. Model Drift
    with tabs[4]:
        drift = results.get("drift", {})
        st.markdown("#### Population Stability Index (PSI) — Validation vs Test Windows")
        st.caption(
            f"Reference: `{drift.get('reference', 'training window')}` · "
            "Thresholds: `< 0.1` Stable · `0.1–0.25` Moderate · `> 0.25` Significant"
        )
        win_key = st.selectbox(
            "Select Evaluation Window",
            ["validation", "test_first_half", "test_second_half"],
            format_func=lambda k: {"validation": "Validation Window", "test_first_half": "Test First Half", "test_second_half": "Test Second Half"}[k],
        )
        win_data = drift.get(win_key, {})
        render_kpi_row(
            [
                KpiItem("Score PSI", f"{float(win_data.get('score_psi', 0)):.4f}", f"Status: {win_data.get('score_status', 'unknown')}", COLORS["green"]),
                KpiItem("Flagged Features", f"{len(win_data.get('features_flagged', []))}", "PSI >= 0.10", COLORS["amber"]),
            ]
        )
        feat_df = pd.DataFrame(win_data.get("features_flagged", []))
        if not feat_df.empty:
            st.plotly_chart(chart_drift_psi(feat_df, win_key), use_container_width=True)
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # F. Streaming Architecture
    with tabs[5]:
        st.markdown("#### Recorded Streaming Experiments (Historical Benchmarks — Not Live Clusters)")
        st.info(
            "These metrics are recorded from prior local Spark and GCP Dataflow benchmark runs. "
            "No Kafka, Spark, or Dataflow cluster is running inside Streamlit."
        )
        burst = results.get("streaming_burst", {})
        paced = results.get("streaming_paced", {})
        df1 = results.get("dataflow_scoring_1worker", {})
        df2 = results.get("dataflow_scoring_2workers", {})

        st.markdown("**1. Kafka → Spark Structured Streaming → ONNX → Parquet**")
        spark_df = pd.DataFrame(
            [
                {
                    "Run": "Burst (Max Rate)",
                    "Events Scored": burst.get("events_scored"),
                    "Spark Throughput (ev/s)": burst.get("spark_throughput_events_per_second"),
                    "P50 Latency (ms)": burst.get("latency_ms", {}).get("p50"),
                    "P95 Latency (ms)": burst.get("latency_ms", {}).get("p95"),
                    "P99 Latency (ms)": burst.get("latency_ms", {}).get("p99"),
                    "Alerts (TP/FP/FN)": f"{burst.get('alerts', {}).get('count')} ({burst.get('alerts', {}).get('tp')}/{burst.get('alerts', {}).get('fp')}/{burst.get('alerts', {}).get('fn')})",
                },
                {
                    "Run": "Paced (2,000 ev/s)",
                    "Events Scored": paced.get("events_scored"),
                    "Spark Throughput (ev/s)": paced.get("spark_throughput_events_per_second"),
                    "P50 Latency (ms)": paced.get("latency_ms", {}).get("p50"),
                    "P95 Latency (ms)": paced.get("latency_ms", {}).get("p95"),
                    "P99 Latency (ms)": paced.get("latency_ms", {}).get("p99"),
                    "Alerts (TP/FP/FN)": f"{paced.get('alerts', {}).get('count')} ({paced.get('alerts', {}).get('tp')}/{paced.get('alerts', {}).get('fp')}/{paced.get('alerts', {}).get('fn')})",
                },
            ]
        )
        st.dataframe(spark_df, use_container_width=True, hide_index=True)

        st.markdown("**2. GCP Pub/Sub → Dataflow → BigQuery**")
        dataflow_df = pd.DataFrame(
            [
                {
                    "Run": "Dataflow (1 Worker)",
                    "Events Scored": df1.get("events_scored"),
                    "Duplicates": df1.get("duplicates"),
                    "Throughput (ev/s)": df1.get("dataflow_throughput_events_per_second"),
                    "P50 E2E Latency (ms)": df1.get("latency_ms_scored_minus_produced", {}).get("p50"),
                    "P95 E2E Latency (ms)": df1.get("latency_ms_scored_minus_produced", {}).get("p95"),
                    "Wall Seconds": df1.get("job", {}).get("wall_seconds"),
                },
                {
                    "Run": "Dataflow (2 Workers)",
                    "Events Scored": df2.get("events_scored"),
                    "Duplicates": df2.get("duplicates"),
                    "Throughput (ev/s)": df2.get("dataflow_throughput_events_per_second"),
                    "P50 E2E Latency (ms)": df2.get("latency_ms_scored_minus_produced", {}).get("p50"),
                    "P95 E2E Latency (ms)": df2.get("latency_ms_scored_minus_produced", {}).get("p95"),
                    "Wall Seconds": df2.get("job", {}).get("wall_seconds"),
                },
            ]
        )
        st.dataframe(dataflow_df, use_container_width=True, hide_index=True)

    # G. Vertex AI
    with tabs[6]:
        vrun = results.get("vertex_run", {})
        st.markdown("#### Vertex AI MLOps Lifecycle (Historical Pipeline Run)")
        st.info(
            "Recorded Vertex AI Pipeline execution (`results/vertex_run.json`). "
            "This reflects a historical cloud run rather than a live Vertex AI endpoint."
        )
        st.markdown(
            "**Lifecycle:** `Validate` → `Train` → `Evaluate` → `Promotion Gate` → `Model Registry` → `Endpoint` → `Smoke Test`"
        )
        render_kpi_row(
            [
                KpiItem("Pipeline Wall Time", f"{int(vrun.get('wall_seconds', 0)):,} s", "Total end-to-end duration", COLORS["blue"]),
                KpiItem("Tasks Succeeded", f"{sum(1 for t in vrun.get('tasks', []) if t.get('state') == 'SUCCEEDED')}/{len(vrun.get('tasks', []))}", "All pipeline stages", COLORS["green"]),
            ]
        )
        tasks_rows = [
            {
                "Task": t.get("task"),
                "State": t.get("state"),
                "Duration (s)": t.get("seconds"),
                "Key Metrics": ", ".join(f"{k}={v}" for k, v in list(t.get("metrics", {}).items())[:4]) or "—",
            }
            for t in vrun.get("tasks", [])
        ]
        st.dataframe(pd.DataFrame(tasks_rows), use_container_width=True, hide_index=True)
