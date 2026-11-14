# FraudPulse — Intelligence Behind Every Transaction

Interactive dark-fintech Streamlit dashboard built around the existing Real-Time Fraud Detection ML project without modifying any original ML code, datasets, or experiment results.

## Local Execution

Install dashboard dependencies and run from the repository root:

```bash
pip install -r fraudpulse-streamlit/requirements.txt
streamlit run fraudpulse-streamlit/app.py
```

## Architecture & Preservation

- **Read-Only Integration:** All paths resolve relative to `Path(__file__)`. The app reads `results/*.json` and `results/figures/*.png` directly from the parent repository.
- **No Retraining / No Fabrication:** If `model.joblib` (and optional `model_meta.json`) is absent, live inference is gracefully disabled with an explanatory message instead of training a replacement or fabricating predictions.
- **Optional Live Inference:** Place the original trained `model.joblib` and `model_meta.json` in `artifacts/` (or set `FRAUDPULSE_MODEL_DIR` / `MODEL_DIR`) to enable real-time scoring and transaction-level SHAP explanations.

## Streamlit Community Cloud Deployment

1. Push the repository to GitHub (ensuring `fraudpulse-streamlit/` and `results/` are committed).
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) → **Create app**.
3. Configure:
   - **Repository:** `<your-github-username>/<your-repo-name>`
   - **Branch:** `main`
   - **Main file path:** `fraudpulse-streamlit/app.py`
   - **Python version (Advanced settings):** `3.11`
4. Streamlit Community Cloud automatically detects `fraudpulse-streamlit/requirements.txt` when the main file is inside `fraudpulse-streamlit/` (or you can mirror it if needed).

## Deployment Limitations

- **Model Artifacts & Raw Dataset:** Because `model.joblib` and `data/creditcard.parquet` are excluded from the repository archive, live single-transaction inference and historical test-row sampling remain disabled on Community Cloud unless you supply the trained artifact via secure storage or an attached artifact directory.
- **Streaming & Cloud Pipelines:** Kafka, Spark Structured Streaming, GCP Dataflow, and Vertex AI sections display recorded historical experiment telemetry from `results/*.json`; no live streaming clusters are started on Streamlit Community Cloud.
