from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data import load_dataset, save_dataset_profile, summarize_class_balance
from .evaluation import compute_metrics, save_confusion_matrix, save_curves, save_feature_importance, save_metrics
from .models import build_model, make_sample_weight
from .preprocessing import apply_random_oversampling, build_preprocessor


def run_credit_benchmark(config_path: str | Path) -> list[dict[str, Any]]:
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runs = config["runs"]
    all_results: list[dict[str, Any]] = []
    for run in runs:
        result = _run_single_experiment(run, config.get("defaults", {}))
        all_results.append(result)

    summary_path = Path(config["summary_output"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_results).to_csv(summary_path, index=False)
    return all_results


def _run_single_experiment(run: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    random_state = int(run.get("split_seed", defaults.get("split_seed", 42)))
    raw_dir = defaults.get("raw_data_dir", "data/raw")
    artifact_dir = Path(run["output_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    X, y, metadata = load_dataset(
        dataset_name=run["dataset"],
        raw_dir=raw_dir,
        sample_size=run.get("sample_size"),
        random_state=random_state,
    )
    dataset_profile = {
        **metadata,
        **summarize_class_balance(y),
        "sample_size": run.get("sample_size"),
    }
    save_dataset_profile(dataset_profile, artifact_dir / "dataset_profile.json")

    stratify = y if y.nunique() > 1 else None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=random_state,
        stratify=stratify,
    )
    temp_stratify = y_temp if y_temp.nunique() > 1 else None
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=random_state,
        stratify=temp_stratify,
    )

    imbalance_strategy = run.get("imbalance_strategy", "none")
    class_weight = "balanced" if imbalance_strategy == "class_weight" else None

    working_X_train = X_train.copy()
    working_y_train = y_train.copy()
    if imbalance_strategy == "oversample":
        working_X_train, working_y_train = apply_random_oversampling(X_train, y_train, random_state=random_state)

    model_spec = build_model(run["model"], random_state=random_state, class_weight=class_weight)
    prepared = build_preprocessor(working_X_train, model_family=model_spec.family)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", prepared.preprocessor),
            ("model", model_spec.model),
        ]
    )

    start = time.perf_counter()
    fit_kwargs: dict[str, Any] = {}
    if run["model"] == "gradient_boosting" and imbalance_strategy == "class_weight":
        fit_kwargs["model__sample_weight"] = make_sample_weight(working_y_train.to_numpy())
    pipeline.fit(working_X_train, working_y_train, **fit_kwargs)
    train_seconds = time.perf_counter() - start

    valid_score = _predict_score(pipeline, X_valid)
    valid_pred = (valid_score >= 0.5).astype(int)
    test_score = _predict_score(pipeline, X_test)
    test_pred = (test_score >= 0.5).astype(int)

    valid_metrics = compute_metrics(y_valid.to_numpy(), valid_pred, valid_score)
    test_metrics = compute_metrics(y_test.to_numpy(), test_pred, test_score)
    save_metrics(valid_metrics, artifact_dir / "validation_metrics.json")
    save_metrics(test_metrics, artifact_dir / "metrics.json")
    save_curves(y_test.to_numpy(), test_score, artifact_dir)
    save_confusion_matrix(y_test.to_numpy(), test_pred, artifact_dir)
    save_feature_importance(pipeline, X_test, y_test.to_numpy(), artifact_dir)

    result_row = {
        "run_id": run["run_id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": run["dataset"],
        "model": run["model"],
        "split_seed": random_state,
        "preprocessing_variant": run.get("preprocessing_variant", "default"),
        "imbalance_strategy": imbalance_strategy,
        "sample_size": run.get("sample_size"),
        "train_seconds": round(train_seconds, 4),
        **{f"valid_{key}": value for key, value in valid_metrics.items() if key != "confusion_matrix"},
        **{f"test_{key}": value for key, value in test_metrics.items() if key != "confusion_matrix"},
        "output_dir": str(artifact_dir),
    }
    save_metrics(result_row, artifact_dir / "run_summary.json")
    return result_row


def _predict_score(pipeline: Pipeline, X: pd.DataFrame):
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    return pipeline.decision_function(X)
