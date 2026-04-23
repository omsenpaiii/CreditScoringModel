from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .preprocessing import get_feature_names


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "precision": float(report["1"]["precision"]),
        "recall": float(report["1"]["recall"]),
        "f1": float(report["1"]["f1-score"]),
        "accuracy": float(report["accuracy"]),
        "support_positive": int(report["1"]["support"]),
        "support_negative": int(report["0"]["support"]),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_metrics(metrics: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_curves(y_true: np.ndarray, y_score: np.ndarray, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label="PR curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=180)
    plt.close()


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def save_feature_importance(pipeline, X_eval: pd.DataFrame, y_eval: np.ndarray, output_dir: str | Path, top_k: int = 15) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = list(get_feature_names(preprocessor))
    transformed = preprocessor.transform(X_eval)

    if hasattr(model, "feature_importances_"):
        scores = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        scores = np.abs(np.asarray(model.coef_)).mean(axis=0)
    else:
        importance = permutation_importance(model, transformed, y_eval, scoring="roc_auc", n_repeats=5, random_state=42)
        scores = importance.importances_mean

    frame = (
        pd.DataFrame({"feature": feature_names, "importance": scores})
        .sort_values("importance", ascending=False)
        .head(top_k)
    )
    csv_path = output_dir / "feature_importance.csv"
    frame.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.barh(frame["feature"][::-1], frame["importance"][::-1])
    plt.xlabel("Importance")
    plt.title("Top Feature Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=180)
    plt.close()
    return csv_path
