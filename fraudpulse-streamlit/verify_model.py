"""Standalone verification script for the reproduced FraudPulse XGBoost model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
for p in (str(APP_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from services.data_loader import FEATURES  # noqa: E402
from services.model_loader import LoadedModel  # noqa: E402
from services.prediction_service import run_prediction, validate_features  # noqa: E402


def main() -> None:
    model_path = APP_DIR / "artifacts" / "model.joblib"
    meta_path = APP_DIR / "artifacts" / "model_meta.json"
    assert model_path.exists(), f"Missing model artifact: {model_path}"
    assert meta_path.exists(), f"Missing model metadata: {meta_path}"

    model = joblib.load(model_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # 1. Feature count & order
    assert meta["features"] == FEATURES, "Feature list/order mismatch!"
    assert len(FEATURES) == 29, f"Expected 29 features, got {len(FEATURES)}"
    if hasattr(model, "n_features_in_"):
        assert model.n_features_in_ == 29, f"Model expects {model.n_features_in_} features"

    threshold = float(meta["threshold"])
    loaded = LoadedModel(
        available=True,
        model=model,
        features=list(meta["features"]),
        threshold=threshold,
        model_name=meta["model"],
        strategy=meta["strategy"],
        artifact_path=model_path,
        meta_path=meta_path,
        message="ok",
    )

    # 2. Valid single transaction
    valid_tx = {f: 0.0 for f in FEATURES}
    valid_tx["Amount"] = 149.62
    res = run_prediction(loaded, valid_tx, "VERIFY-001")
    assert 0.0 <= res.probability <= 1.0, f"Invalid probability: {res.probability}"
    assert res.model_prediction in ("Fraudulent", "Legitimate")
    assert res.is_fraud == (res.probability >= threshold)
    print(f"[PASS] Single-transaction inference: prob={res.probability:.6f}, pred={res.model_prediction}, latency={res.latency_ms:.3f}ms")

    # 3. Missing feature rejection
    missing_tx = {f: 0.0 for f in FEATURES[:-1]}
    try:
        validate_features(missing_tx, FEATURES)
        raise AssertionError("Expected ValueError for missing feature")
    except ValueError as exc:
        print(f"[PASS] Missing feature rejected: {exc}")

    # 4. Unexpected feature rejection
    extra_tx = {**valid_tx, "Time": 123.0}
    try:
        validate_features(extra_tx, FEATURES)
        raise AssertionError("Expected ValueError for unexpected feature")
    except ValueError as exc:
        print(f"[PASS] Unexpected feature rejected: {exc}")

    # 5. Invalid (NaN) feature rejection
    nan_tx = {**valid_tx, "V1": float("nan")}
    try:
        validate_features(nan_tx, FEATURES)
        raise AssertionError("Expected ValueError for NaN feature")
    except ValueError as exc:
        print(f"[PASS] Non-finite feature rejected: {exc}")

    # 6. Multi-transaction batch scoring
    batch = np.zeros((5, 29), dtype=np.float32)
    batch_probs = model.predict_proba(batch)[:, 1]
    assert batch_probs.shape == (5,)
    assert np.all((batch_probs >= 0.0) & (batch_probs <= 1.0))
    print(f"[PASS] Batch inference (5 rows): probs={batch_probs.round(6).tolist()}")

    # 7. Independent test-window evaluation if data/creditcard.parquet exists
    parquet_path = REPO_ROOT / "data" / "creditcard.parquet"
    if parquet_path.exists():
        from fraud import data as D  # noqa: E402
        from fraud import metrics as M  # noqa: E402

        df = D.load(parquet_path)
        split = D.time_split(df)
        test_scores = model.predict_proba(split.x_test)[:, 1]
        pr_auc = M.pr_auc(split.y_test, test_scores)
        roc_auc = M.roc_auc(split.y_test, test_scores)
        prf = M.precision_recall_f1(split.y_test, test_scores, threshold)
        print(
            f"[PASS] Independent Test Evaluation (n={len(split.y_test):,}, frauds={int(split.y_test.sum())}): "
            f"PR-AUC={pr_auc:.4f} (ref ~0.7944), ROC-AUC={roc_auc:.4f} (ref ~0.9793), "
            f"Precision={prf['precision']:.4f}, Recall={prf['recall']:.4f}, F1={prf['f1']:.4f}"
        )
    else:
        print("[INFO] data/creditcard.parquet not present; skipped full test-set evaluation.")


if __name__ == "__main__":
    main()
