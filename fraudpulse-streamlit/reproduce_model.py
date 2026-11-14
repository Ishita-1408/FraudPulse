"""Reproduce the original selected XGBoost + class-weights model artifact.

Uses the exact unmodified pipeline from `fraud.data`, `fraud.models`, and `fraud.metrics`
and saves `model.joblib` + `model_meta.json` into `fraudpulse-streamlit/artifacts/`.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import xgboost

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fraud import data as D  # noqa: E402
from fraud import metrics as M  # noqa: E402
from fraud import models as Mo  # noqa: E402


def main() -> None:
    print("Loading OpenML 1597 creditcard dataset via fraud.data.load()...")
    df = D.load(REPO_ROOT / D.DATA_PATH)
    split = D.time_split(df)
    summary = split.summary()
    print(f"Split summary: train={summary['train']['rows']:,} ({summary['train']['frauds']} frauds), "
          f"val={summary['val']['rows']:,} ({summary['val']['frauds']} frauds), "
          f"test={summary['test']['rows']:,} ({summary['test']['frauds']} frauds)")

    print("Training original XGBoost (strategy='weights', seed=0) via fraud.models.train()...")
    trained = Mo.train(
        "xgboost",
        "weights",
        split.x_train,
        split.y_train,
        split.x_val,
        split.y_val,
        seed=0,
        n_jobs=4,
    )

    val_score = trained.score(split.x_val)
    test_score = trained.score(split.x_test)
    threshold = M.best_f1_threshold(split.y_val, val_score)
    val_pr_auc = round(M.pr_auc(split.y_val, val_score), 4)
    test_report = M.full_report(split.y_test, test_score, threshold=threshold)

    print(f"Validation PR-AUC: {val_pr_auc:.4f} | Max-F1 Threshold: {threshold:.4f}")
    print(f"Test Report: {json.dumps(test_report, indent=2)}")

    artifacts_dir = APP_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "model.joblib"
    meta_path = artifacts_dir / "model_meta.json"

    joblib.dump(trained.model, model_path)

    pos_weight = float((split.y_train == 0).sum() / max(1, (split.y_train == 1).sum()))
    meta = {
        "model": trained.name,
        "algorithm": "XGBClassifier (gradient boosted decision trees, tree_method='hist')",
        "strategy": trained.strategy,
        "model_version": f"xgboost-{xgboost.__version__}",
        "features": list(D.FEATURES),
        "n_features": len(D.FEATURES),
        "threshold": float(threshold),
        "threshold_rule": "max F1 on validation window",
        "preprocessing": {
            "dtype": "float32",
            "scaling": "none (raw V1-V28 PCA components + Amount)",
            "split": "time-ordered 70/10/20, no shuffling",
        },
        "training_config": {
            "n_estimators": 600,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "scale_pos_weight": round(pos_weight, 4),
            "eval_metric": "aucpr",
            "early_stopping_rounds": 50,
            "seed": 0,
            "best_iteration": trained.extra.get("best_iteration"),
            "n_train_rows": trained.n_train,
            "fit_seconds": trained.fit_seconds,
        },
        "metrics": {
            "val_pr_auc": val_pr_auc,
            **test_report,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved model to {model_path}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
