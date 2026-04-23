from __future__ import annotations

import argparse
import json

from document_ocr.runner import run_ocr_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PaddleOCR pilot experiment.")
    parser.add_argument("--config", required=True, help="Path to the OCR JSON config.")
    args = parser.parse_args()
    result = run_ocr_experiment(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
