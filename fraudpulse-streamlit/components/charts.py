"""Reusable Plotly charts for FraudPulse built strictly from recorded experiment data."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from components.theme import COLORS, apply_plotly_theme


def chart_split_distribution(split: dict[str, Any]) -> go.Figure:
    """Stacked bar chart of legitimate vs fraudulent transactions across time-ordered windows."""
    windows = ["train", "val", "test"]
    labels = ["Train (70%)", "Validation (10%)", "Test (20%)"]
    legit = [int(split[w]["rows"]) - int(split[w]["frauds"]) for w in windows]
    fraud = [int(split[w]["frauds"]) for w in windows]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Legitimate",
            x=labels,
            y=legit,
            marker_color=COLORS["green"],
            text=[f"{v:,}" for v in legit],
            textposition="inside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Fraudulent",
            x=labels,
            y=fraud,
            marker_color=COLORS["red"],
            text=[f"{v:,}" for v in fraud],
            textposition="outside",
        )
    )
    fig.update_layout(barmode="stack", yaxis_title="Transactions")
    return apply_plotly_theme(fig, "Time-Ordered Split: Legitimate vs Fraudulent Volume", height=350)


def chart_fraud_prevalence(split: dict[str, Any]) -> go.Figure:
    """Bar chart of fraud prevalence (%) across train, validation, and test windows."""
    windows = ["train", "val", "test"]
    labels = ["Train (70%)", "Validation (10%)", "Test (20%)"]
    rates = [float(split[w]["fraud_rate"]) * 100.0 for w in windows]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=rates,
            marker_color=[COLORS["blue"], COLORS["cyan"], COLORS["amber"]],
            text=[f"{r:.4f}%" for r in rates],
            textposition="outside",
        )
    )
    fig.update_layout(yaxis_title="Fraud Prevalence (%)", yaxis_range=[0, max(rates) * 1.25])
    return apply_plotly_theme(fig, "Fraud Prevalence by Time Window", height=350)


def chart_best_model_metrics(best: dict[str, Any]) -> go.Figure:
    """Horizontal bar summary of key evaluation metrics for the selected XGBoost model."""
    metrics = [
        ("PR-AUC", float(best.get("pr_auc", 0.0))),
        ("ROC-AUC", float(best.get("roc_auc", 0.0))),
        ("Precision", float(best.get("precision", 0.0))),
        ("Recall", float(best.get("recall", 0.0))),
        ("F1-Score", float(best.get("f1", 0.0))),
        ("Recall @ 0.1% FPR", float(best.get("recall_at_fpr_0.1pct", 0.0))),
        ("Recall @ 1.0% FPR", float(best.get("recall_at_fpr_1pct", 0.0))),
    ]
    names = [m[0] for m in metrics][::-1]
    vals = [m[1] for m in metrics][::-1]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker_color=COLORS["cyan"],
            text=[f"{v:.4f}" for v in vals],
            textposition="outside",
        )
    )
    fig.update_layout(xaxis_title="Score", xaxis_range=[0, 1.12])
    return apply_plotly_theme(
        fig,
        f"Selected Model Performance ({best.get('model', 'xgboost')} / {best.get('strategy', 'weights')})",
        height=360,
    )


def chart_supervised_comparison(df: pd.DataFrame, metric: str) -> go.Figure:
    """Grouped bar chart comparing supervised configurations on a chosen metric."""
    fig = px.bar(
        df,
        x="model",
        y=metric,
        color="strategy",
        barmode="group",
        color_discrete_map={
            "none": COLORS["blue"],
            "weights": COLORS["cyan"],
            "smote": COLORS["amber"],
        },
        text_auto=".4f" if metric != "fit_seconds" else ".2f",
    )
    fig.update_layout(xaxis_title="Model Family", yaxis_title=metric.replace("_", " ").upper())
    return apply_plotly_theme(fig, f"Supervised Model Comparison — {metric}", height=390)


def chart_confusion_matrix(tp: int, fp: int, fn: int, tn: int, title: str) -> go.Figure:
    """Plotly heatmap for a 2x2 confusion matrix."""
    z = [[tn, fp], [fn, tp]]
    text = [
        [f"TN: {tn:,}", f"FP: {fp:,}"],
        [f"FN: {fn:,}", f"TP: {tp:,}"],
    ]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Predicted Legitimate", "Predicted Fraud"],
            y=["Actual Legitimate", "Actual Fraud"],
            text=text,
            texttemplate="%{text}",
            colorscale=[[0.0, "#0F172A"], [0.5, "#1D4ED8"], [1.0, "#06B6D4"]],
            showscale=False,
        )
    )
    fig.update_yaxes(autorange="reversed")
    return apply_plotly_theme(fig, title, height=340)


def chart_cost_curve(cost_data: dict[str, Any]) -> go.Figure:
    """Expected cost per 10,000 transactions across decision thresholds."""
    curve = pd.DataFrame(cost_data.get("test_curve", []))
    if curve.empty:
        return apply_plotly_theme(go.Figure(), "Cost-Sensitive Curve (No Data)")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve["threshold"],
            y=curve["cost_per_10k"],
            mode="lines+markers",
            name="Cost per 10k tx",
            line={"color": COLORS["cyan"], "width": 2.5},
            marker={"size": 5},
        )
    )
    opt_t = float(cost_data.get("threshold_from_validation", 0.0))
    opt_c = float(cost_data.get("test_cost_per_10k_at_threshold", 0.0))
    fig.add_trace(
        go.Scatter(
            x=[opt_t],
            y=[opt_c],
            mode="markers+text",
            name="Val-Optimal Threshold",
            text=[f"t={opt_t:.5f}<br>${opt_c:.2f}"],
            textposition="top right",
            marker={"color": COLORS["green"], "size": 11, "symbol": "diamond"},
        )
    )
    fig.update_layout(
        xaxis_type="log",
        xaxis_title="Decision Threshold (log scale)",
        yaxis_title="Expected Cost per 10,000 Transactions ($)",
    )
    return apply_plotly_theme(fig, "Cost-Sensitive Threshold Curve (Test Window)", height=370)


def chart_reliability(calibration: dict[str, Any]) -> go.Figure:
    """Calibration reliability diagram comparing raw vs isotonic-calibrated probabilities."""
    raw = pd.DataFrame(calibration.get("reliability_raw", []))
    cal = pd.DataFrame(calibration.get("reliability_calibrated", []))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 0.02],
            y=[0, 0.02],
            mode="lines",
            name="Perfect Calibration",
            line={"color": COLORS["muted"], "dash": "dash"},
        )
    )
    if not raw.empty:
        fig.add_trace(
            go.Scatter(
                x=raw["mean_pred"],
                y=raw["observed"],
                mode="lines+markers",
                name=f"Raw (Brier={calibration.get('brier_raw', 0):.6f})",
                line={"color": COLORS["amber"], "width": 2},
            )
        )
    if not cal.empty:
        fig.add_trace(
            go.Scatter(
                x=cal["mean_pred"],
                y=cal["observed"],
                mode="lines+markers",
                name=f"Isotonic Calibrated (Brier={calibration.get('brier_calibrated', 0):.6f})",
                line={"color": COLORS["green"], "width": 2.5},
            )
        )
    fig.update_layout(xaxis_title="Mean Predicted Probability", yaxis_title="Observed Fraud Rate")
    return apply_plotly_theme(fig, "Calibration Reliability (Test Window)", height=370)


def chart_shap_importance(df: pd.DataFrame, title: str, value_col: str = "mean_abs_shap") -> go.Figure:
    """Horizontal bar chart of SHAP feature importance."""
    ordered = df.sort_values(value_col, ascending=True)
    colors = (
        [COLORS["red"] if v >= 0 else COLORS["green"] for v in ordered[value_col]]
        if value_col == "shap_value"
        else COLORS["cyan"]
    )
    fig = go.Figure(
        go.Bar(
            x=ordered[value_col],
            y=ordered["feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.4f}" for v in ordered[value_col]],
            textposition="outside",
        )
    )
    fig.update_layout(xaxis_title=value_col.replace("_", " ").title(), yaxis_title="PCA / Amount Feature")
    return apply_plotly_theme(fig, title, height=max(340, 26 * len(ordered) + 80))


def chart_drift_psi(features_df: pd.DataFrame, window_label: str) -> go.Figure:
    """Feature-level PSI drift bar chart with moderate (0.1) and significant (0.25) thresholds."""
    ordered = features_df.sort_values("psi", ascending=True)
    color_map = {"significant": COLORS["red"], "moderate": COLORS["amber"], "stable": COLORS["green"]}
    bar_colors = [color_map.get(s, COLORS["blue"]) for s in ordered["status"]]
    fig = go.Figure(
        go.Bar(
            x=ordered["psi"],
            y=ordered["feature"],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.4f}" for v in ordered["psi"]],
            textposition="outside",
        )
    )
    fig.add_vline(x=0.1, line_dash="dash", line_color=COLORS["amber"], annotation_text="Moderate (0.10)")
    fig.add_vline(x=0.25, line_dash="dash", line_color=COLORS["red"], annotation_text="Significant (0.25)")
    fig.update_layout(xaxis_title="Population Stability Index (PSI)", yaxis_title="Feature")
    return apply_plotly_theme(fig, f"Feature-Level Drift (PSI) — {window_label}", height=440)


def chart_privacy_tradeoff(df: pd.DataFrame) -> go.Figure:
    """Privacy budget (epsilon) vs PR-AUC / ROC-AUC for DP-SGD MLP."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["epsilon_label"],
            y=df["pr_auc"],
            name="PR-AUC",
            marker_color=COLORS["cyan"],
            text=[f"{v:.4f}" for v in df["pr_auc"]],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["epsilon_label"],
            y=df["roc_auc"],
            name="ROC-AUC",
            mode="lines+markers",
            line={"color": COLORS["blue"], "width": 2.5},
            marker={"size": 8},
        )
    )
    fig.update_layout(
        xaxis_title="Privacy Budget (Epsilon Target)",
        yaxis_title="Score",
        yaxis_range=[0, 1.1],
    )
    return apply_plotly_theme(fig, "DP-SGD Privacy vs Utility Trade-off (MLP 30-64-32-1)", height=360)


def chart_efficiency_latency(df: pd.DataFrame) -> go.Figure:
    """P50, P95, and P99 single-transaction inference latency vs 10 ms budget."""
    fig = go.Figure()
    for col, name, color in [
        ("p50_ms", "P50 (ms)", COLORS["green"]),
        ("p95_ms", "P95 (ms)", COLORS["cyan"]),
        ("p99_ms", "P99 (ms)", COLORS["amber"]),
    ]:
        fig.add_trace(
            go.Bar(
                x=df["artifact"],
                y=df[col],
                name=name,
                marker_color=color,
                text=[f"{v:.3f}" for v in df[col]],
                textposition="outside",
            )
        )
    fig.add_hline(y=10.0, line_dash="dash", line_color=COLORS["red"], annotation_text="10 ms P95 Budget")
    fig.update_layout(barmode="group", xaxis_title="Runtime Artifact", yaxis_title="Latency (ms)")
    return apply_plotly_theme(fig, "Single-Transaction Inference Latency vs 10 ms Budget", height=390)
