from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


@dataclass
class ModelSpec:
    model: Any
    family: str
    supports_class_weight: bool


def build_model(model_name: str, random_state: int, class_weight: str | None = None) -> ModelSpec:
    if model_name == "logistic_regression":
        model = LogisticRegression(
            max_iter=1500,
            solver="lbfgs",
            class_weight=class_weight,
            random_state=random_state,
        )
        return ModelSpec(model=model, family="linear", supports_class_weight=True)

    if model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
            class_weight=class_weight,
        )
        return ModelSpec(model=model, family="tree", supports_class_weight=True)

    if model_name == "gradient_boosting":
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=250,
            random_state=random_state,
        )
        return ModelSpec(model=model, family="tree", supports_class_weight=False)

    if model_name == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            batch_size=64,
            max_iter=300,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=random_state,
        )
        return ModelSpec(model=model, family="neural", supports_class_weight=False)

    raise ValueError(f"Unsupported model: {model_name}")


def make_sample_weight(y: np.ndarray) -> np.ndarray:
    values, counts = np.unique(y, return_counts=True)
    if len(values) < 2:
        return np.ones_like(y, dtype=float)
    total = counts.sum()
    weights = {value: total / (len(values) * count) for value, count in zip(values, counts)}
    return np.array([weights[value] for value in y], dtype=float)
