from __future__ import annotations

from pathlib import Path

from credit_scoring.data import load_dataset


def main() -> None:
    root = Path("/Users/macbook/Downloads/Credit Scoring Model")
    raw_dir = root / "data/raw"
    german_X, german_y, german_meta = load_dataset("german_credit", raw_dir=raw_dir)
    assert len(german_X) == len(german_y) == german_meta["records"]
    print("German Credit loader OK")


if __name__ == "__main__":
    main()
