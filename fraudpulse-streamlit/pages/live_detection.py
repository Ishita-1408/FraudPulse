"""Section 2: Live Fraud Detection interface."""

from __future__ import annotations

import uuid

import pandas as pd
import streamlit as st

from components.header import render_page_header
from components.kpi_cards import KpiItem, render_kpi_row
from components.theme import COLORS
from services.data_loader import FEATURES, load_json_result, load_test_samples
from services.model_loader import LoadedModel
from services.prediction_service import run_prediction


def _init_feature_state() -> None:
    if "fp_tx_inputs" not in st.session_state:
        st.session_state["fp_tx_inputs"] = {f: 0.0 for f in FEATURES}
    if "fp_history" not in st.session_state:
        st.session_state["fp_history"] = []


def render(loaded_model: LoadedModel) -> None:
    """Render the Live Fraud Detection page."""
    _init_feature_state()
    baselines = load_json_result("baselines.json")
    best = baselines.get("best", {})

    badge_text = "Live Inference Ready" if loaded_model.available else "Inference Unavailable (Artifact Missing)"
    badge_variant = "green" if loaded_model.available else "amber"

    render_page_header(
        title="Live Fraud Detection",
        subtitle="Score individual transactions against the original XGBoost model schema, preprocessing, and binary decision threshold.",
        badge_text=badge_text,
        badge_variant=badge_variant,
    )

    if not loaded_model.available:
        st.warning(
            f"**Real-Time Model Inference Unavailable:** {loaded_model.message}\n\n"
            "FraudPulse never trains a replacement model or fabricates predictions. "
            "To enable live scoring, place `model.joblib` (and `model_meta.json`) in "
            "`fraudpulse-streamlit/artifacts/` or `artifacts/`, or set `FRAUDPULSE_MODEL_DIR` / `MODEL_DIR`."
        )
    else:
        st.success(
            f"**Active Model:** `{loaded_model.model_name}` (`{loaded_model.strategy}`) · "
            f"Binary Decision Threshold: `{loaded_model.threshold:.4f}` · Artifact: `{loaded_model.artifact_path}`"
        )

    # Historical test-window transaction selector (only when data/creditcard.parquet exists)
    test_df = load_test_samples()
    if test_df is not None and not test_df.empty:
        st.markdown("<div class='fp-section-title'>Held-Out Test Transaction Selector</div>", unsafe_allow_html=True)
        sample_options = ["Manual Entry"] + [
            f"Test Row seq={int(row['seq'])} · Amount=${float(row['Amount']):.2f}"
            for _, row in test_df.iterrows()
        ]
        choice = st.selectbox(
            "Select a transaction from the held-out test window (actual label stays hidden until revealed after prediction)",
            sample_options,
        )
        if choice != "Manual Entry":
            idx = sample_options.index(choice) - 1
            chosen_row = test_df.iloc[idx]
            if st.button("Load Selected Test Transaction"):
                for feat in FEATURES:
                    st.session_state["fp_tx_inputs"][feat] = float(chosen_row[feat])
                st.session_state["fp_loaded_sample_seq"] = int(chosen_row["seq"])
                st.session_state["fp_loaded_sample_label"] = int(chosen_row["Class"])
                st.session_state["fp_reveal_label"] = False
                st.session_state.pop("fp_last_result", None)
                st.rerun()
    else:
        st.info(
            "**Historical Transaction Selection Unavailable:** `data/creditcard.parquet` is not present locally. "
            "Enter `Amount` and `V1–V28` manually below."
        )

    # Transaction Input Form
    st.markdown("<div class='fp-section-title'>Transaction Feature Input (29-Feature Schema: V1–V28 + Amount)</div>", unsafe_allow_html=True)
    with st.form("live_tx_form"):
        top_c1, top_c2 = st.columns([1.2, 1.8])
        with top_c1:
            default_tx_id = (
                f"TEST-SEQ-{st.session_state['fp_loaded_sample_seq']}"
                if "fp_loaded_sample_seq" in st.session_state
                else f"TX-{uuid.uuid4().hex[:8].upper()}"
            )
            tx_id = st.text_input("Transaction ID", value=default_tx_id)
        with top_c2:
            amount_val = st.number_input(
                "Transaction Amount",
                min_value=0.0,
                value=float(st.session_state["fp_tx_inputs"].get("Amount", 0.0)),
                step=1.0,
                format="%.2f",
            )
            st.session_state["fp_tx_inputs"]["Amount"] = amount_val

        groups = [
            ("PCA Components V1 – V7", [f"V{i}" for i in range(1, 8)]),
            ("PCA Components V8 – V14 (Includes top SHAP drivers V14, V12, V10, V11)", [f"V{i}" for i in range(8, 15)]),
            ("PCA Components V15 – V21", [f"V{i}" for i in range(15, 22)]),
            ("PCA Components V22 – V28", [f"V{i}" for i in range(22, 29)]),
        ]
        for group_title, cols_list in groups:
            with st.expander(group_title, expanded=(group_title.startswith("PCA Components V1 – V7"))):
                grid_cols = st.columns(4)
                for i, feat in enumerate(cols_list):
                    with grid_cols[i % 4]:
                        val = st.number_input(
                            feat,
                            value=float(st.session_state["fp_tx_inputs"].get(feat, 0.0)),
                            step=0.1,
                            format="%.4f",
                            key=f"input_{feat}",
                        )
                        st.session_state["fp_tx_inputs"][feat] = val

        submitted = st.form_submit_button(
            "Predict Fraud",
            type="primary",
            disabled=not loaded_model.available,
            use_container_width=True,
        )

    if submitted and loaded_model.available:
        try:
            result = run_prediction(loaded_model, st.session_state["fp_tx_inputs"], tx_id)
            st.session_state["fp_last_result"] = result
            st.session_state["fp_history"].insert(
                0,
                {
                    "transaction_id": result.transaction_id,
                    "amount": round(result.amount, 2),
                    "fraud_probability_pct": f"{result.probability * 100:.4f}%",
                    "threshold": round(result.threshold, 4),
                    "prediction": result.model_prediction,
                    "latency_ms": round(result.latency_ms, 3),
                },
            )
        except ValueError as exc:
            st.error(f"Invalid transaction input: {exc}")

    last_result = st.session_state.get("fp_last_result")
    if last_result is not None and loaded_model.available:
        st.markdown("<div class='fp-section-title'>Prediction Result</div>", unsafe_allow_html=True)
        decision_color = COLORS["red"] if last_result.is_fraud else COLORS["green"]
        if last_result.is_fraud:
            st.error(
                f"**FRAUD ALERT:** Transaction `{last_result.transaction_id}` classified as **Fraudulent** "
                f"(probability `{last_result.probability * 100:.4f}%` ≥ threshold `{last_result.threshold:.4f}`)."
            )
        else:
            st.success(
                f"**CLEARED:** Transaction `{last_result.transaction_id}` classified as **Legitimate** "
                f"(probability `{last_result.probability * 100:.4f}%` < threshold `{last_result.threshold:.4f}`)."
            )

        render_kpi_row(
            [
                KpiItem("Transaction ID", last_result.transaction_id, f"Amount: ${last_result.amount:,.2f}", COLORS["blue"]),
                KpiItem("Fraud Probability", f"{last_result.probability * 100:.4f}%", f"Raw score: {last_result.probability:.6f}", decision_color),
                KpiItem("Prediction", last_result.model_prediction, f"Threshold: {last_result.threshold:.4f}", decision_color),
                KpiItem("Prediction Latency", f"{last_result.latency_ms:.3f} ms", "Single-transaction inference", COLORS["cyan"]),
            ]
        )

        # Held-out test label reveal (after prediction only)
        if "fp_loaded_sample_label" in st.session_state:
            st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
            if not st.session_state.get("fp_reveal_label", False):
                if st.button("Reveal Actual Test Label"):
                    st.session_state["fp_reveal_label"] = True
                    st.rerun()
            else:
                actual_cls = int(st.session_state["fp_loaded_sample_label"])
                actual_str = "Fraudulent" if actual_cls == 1 else "Legitimate"
                is_correct = (actual_cls == 1) == last_result.is_fraud
                if is_correct:
                    st.success(f"Actual Label: **{actual_str}** (Class={actual_cls}) — Model prediction was **Correct**.")
                else:
                    st.warning(f"Actual Label: **{actual_str}** (Class={actual_cls}) — Model prediction was **Incorrect**.")

    # Session Prediction History
    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
    hist_h1, hist_h2 = st.columns([3, 1])
    with hist_h1:
        st.markdown("<div class='fp-section-title'>Session Prediction History</div>", unsafe_allow_html=True)
    with hist_h2:
        if st.button("Clear / Reset History", use_container_width=True):
            st.session_state["fp_history"] = []
            st.session_state.pop("fp_last_result", None)
            st.session_state.pop("fp_loaded_sample_label", None)
            st.session_state.pop("fp_loaded_sample_seq", None)
            st.session_state["fp_reveal_label"] = False
            st.rerun()

    history = st.session_state.get("fp_history", [])
    if history:
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
    else:
        st.caption(
            f"Session history is empty. Active decision threshold: `{loaded_model.threshold:.4f}` "
            f"(rule: `{baselines.get('threshold_rule', 'max F1 on validation')}`, "
            f"baselines.json reference: `{float(best.get('threshold', 0.9357)):.4f}`)."
        )
