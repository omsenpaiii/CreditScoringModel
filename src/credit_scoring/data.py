from __future__ import annotations

import json
import ssl
import subprocess
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


GERMAN_COLUMNS = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "present_residence_since",
    "property_type",
    "age_years",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_liable_people",
    "telephone",
    "foreign_worker",
    "target",
]


GERMAN_URL = "https://cdn.uci-ics-mlr-prod.aws.uci.edu/144/statlog%2Bgerman%2Bcredit%2Bdata.zip"
LENDING_CLUB_URL = "https://raw.githubusercontent.com/h2oai/app-consumer-loan/master/data/loan.csv"


def _download_bytes(url: str) -> bytes:
    context = ssl.create_default_context()
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
        },
    )
    try:
        with urlopen(request, context=context, timeout=60) as response:
            return response.read()
    except Exception:
        result = subprocess.run(
            ["curl", "-L", "--fail", "-A", "Mozilla/5.0", "-H", "Accept: */*", url],
            check=True,
            capture_output=True,
        )
        return result.stdout


def ensure_dataset(dataset_name: str, raw_dir: str | Path) -> Path:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if dataset_name == "german_credit":
        target_path = raw_dir / "german_credit.data"
        if not target_path.exists():
            archive_bytes = _download_bytes(GERMAN_URL)
            with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
                with archive.open("german.data") as source:
                    target_path.write_bytes(source.read())
        return target_path
    elif dataset_name == "lending_club_sample":
        target_path = raw_dir / "lending_club_sample.csv"
        url = LENDING_CLUB_URL
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if not target_path.exists():
        target_path.write_bytes(_download_bytes(url))
    return target_path


def load_dataset(dataset_name: str, raw_dir: str | Path, sample_size: int | None = None, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    dataset_path = ensure_dataset(dataset_name, raw_dir)
    if dataset_name == "german_credit":
        df = pd.read_csv(dataset_path, sep=r"\s+", header=None, names=GERMAN_COLUMNS)
        target = (df.pop("target") == 2).astype(int)
        metadata = {
            "dataset": "german_credit",
            "source_path": str(dataset_path),
            "records": int(len(df)),
            "target_name": "bad_credit_risk",
            "positive_label": 1,
            "notes": "Positive class represents bad credit risk (original target value 2).",
        }
        return df, target, metadata

    df = pd.read_csv(dataset_path)
    target = _map_lending_target(df)
    if sample_size and sample_size < len(df):
        working = df.assign(target=target)
        strata = working["target"]
        working = working.groupby(strata, group_keys=False).apply(
            lambda group: group.sample(
                n=min(len(group), round(sample_size * len(group) / len(working))),
                random_state=random_state,
            )
        )
        working = working.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        target = working.pop("target")
        df = working
    else:
        target = target.reset_index(drop=True)
        df = df.reset_index(drop=True)

    metadata = {
        "dataset": "lending_club_sample",
        "source_path": str(dataset_path),
        "records": int(len(df)),
        "target_name": "default_like_outcome",
        "positive_label": 1,
        "notes": "Public cleaned Lending Club CSV mirrored on GitHub. Pilot run may use a stratified subset for feasibility.",
    }
    return df, target, metadata


def _map_lending_target(df: pd.DataFrame) -> pd.Series:
    if "bad_loan" in df.columns:
        target = df["bad_loan"].astype(int).reset_index(drop=True)
        df.drop(columns=["bad_loan"], inplace=True)
        return target

    if "loan_status" not in df.columns:
        raise ValueError("Expected 'bad_loan' or 'loan_status' column in Lending Club sample.")

    risky_statuses = {
        "Charged Off",
        "Default",
        "Late (31-120 days)",
        "Late (16-30 days)",
        "In Grace Period",
    }
    safe_statuses = {"Fully Paid", "Current"}

    target = df["loan_status"].isin(risky_statuses).astype(int)
    known_mask = df["loan_status"].isin(risky_statuses.union(safe_statuses))
    if known_mask.sum() == 0:
        raise ValueError("No known loan status values were found for binary mapping.")
    filtered = df.loc[known_mask].copy()
    target = target.loc[known_mask].reset_index(drop=True)

    df.drop(df.index.difference(filtered.index), inplace=True)
    df.reset_index(drop=True, inplace=True)
    return target


def save_dataset_profile(metadata: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def summarize_class_balance(target: pd.Series) -> dict[str, Any]:
    counts = target.value_counts().sort_index().to_dict()
    total = int(target.shape[0])
    positive_rate = float(target.mean())
    return {
        "class_counts": {str(int(k)): int(v) for k, v in counts.items()},
        "records": total,
        "positive_rate": positive_rate,
    }
