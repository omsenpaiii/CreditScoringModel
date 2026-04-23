from __future__ import annotations

import argparse

from credit_scoring.runner import run_credit_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SIT723 credit-scoring pilot benchmark.")
    parser.add_argument("--config", required=True, help="Path to the benchmark JSON config.")
    args = parser.parse_args()
    results = run_credit_benchmark(args.config)
    print(f"Completed {len(results)} benchmark runs.")


if __name__ == "__main__":
    main()
