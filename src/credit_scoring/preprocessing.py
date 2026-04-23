from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreparedFeatures:
    preprocessor: ColumnTransformer
    numeric_features: list[str]
    categorical_features: list[str]


def build_preprocessor(df: pd.DataFrame, model_family: str) -> PreparedFeatures:
    numeric_features = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in df.columns if column not in numeric_features]

    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if model_family in {"linear", "neural"}:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=numeric_steps), numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return PreparedFeatures(
        preprocessor=preprocessor,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )


def apply_random_oversampling(X: pd.DataFrame, y: pd.Series, random_state: int) -> tuple[pd.DataFrame, pd.Series]:
    target_counts = y.value_counts()
    if len(target_counts) < 2 or target_counts.nunique() == 1:
        return X, y

    majority_class = target_counts.idxmax()
    minority_class = target_counts.idxmin()
    majority_count = int(target_counts.loc[majority_class])
    minority_indices = y[y == minority_class].index.to_numpy()
    rng = np.random.default_rng(random_state)
    sampled_indices = rng.choice(minority_indices, size=majority_count - len(minority_indices), replace=True)
    oversampled_idx = np.concatenate([y.index.to_numpy(), sampled_indices])
    oversampled_X = X.loc[oversampled_idx].reset_index(drop=True)
    oversampled_y = y.loc[oversampled_idx].reset_index(drop=True)
    return oversampled_X, oversampled_y


def get_feature_names(preprocessor: ColumnTransformer) -> Sequence[str]:
    names: list[str] = []
    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer_name == "remainder":
            continue
        if hasattr(transformer, "named_steps") and "encoder" in transformer.named_steps:
            encoder = transformer.named_steps["encoder"]
            names.extend(encoder.get_feature_names_out(columns).tolist())
        else:
            names.extend(list(columns))
    return names
