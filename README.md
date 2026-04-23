# SIT723 Credit Scoring and PaddleOCR Pilot

This workspace implements a supervisor-ready SIT723 progress package with:

- credit-scoring pilot experiments on German Credit and a public Lending Club sample
- a PaddleOCR-oriented OCR pilot workflow for document-intake experimentation
- report and email generation driven by saved experiment artifacts

## Quick start

Use the bundled Python runtime plus the local dependency folder:

```bash
PYTHONPATH="/Users/macbook/Downloads/Credit Scoring Model/.deps:/Users/macbook/Downloads/Credit Scoring Model/src" \
/Users/macbook/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 run_credit_benchmark.py --config configs/credit/benchmark_suite.json
```

Then generate the report package:

```bash
PYTHONPATH="/Users/macbook/Downloads/Credit Scoring Model/.deps:/Users/macbook/Downloads/Credit Scoring Model/src" \
/Users/macbook/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/generate_report.py
```

## Structure

- `src/credit_scoring`: data download, preprocessing, training, evaluation, explainability
- `src/document_ocr`: synthetic OCR data generation and PaddleOCR experiment orchestration
- `configs`: JSON experiment configs
- `artifacts`: generated metrics, plots, and sample predictions
- `report`: generated progress report and supervisor email draft

## Notes

- Credit-scoring results are intended to be real pilot results only.
- The Lending Club pilot uses a manageable public sample rather than the full paper-scale dataset.
- The OCR runner is designed for a tiny synthetic fine-tuning pilot so results remain clearly preliminary.
