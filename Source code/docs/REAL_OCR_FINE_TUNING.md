# Real PaddleOCR Fine-Tuning Pipeline

This project uses real public scanned-document datasets for OCR preparation and evaluation:

- **ICDAR2019 SROIE**: scanned receipts with OCR transcripts, bounding boxes, and key fields such as company, address, date, and total.
- **FUNSD**: noisy scanned forms with word-level annotations for document understanding research.

The conversion script crops annotated text regions and writes PaddleOCR `SimpleDataSet` recognition labels in this format:

```text
images/train/example.png<TAB>recognized text
```

## Setup

```bash
cd "Source code"
python3 -m pip install -r requirements.txt
```

If you use the bundled local dependency folder during evaluation:

```bash
PYTHONPATH="../.deps:src" python3 run_ocr_experiment.py --config configs/ocr/real_ocr_finetune.json
```

## CPU-Friendly Real-Data Run

The default config prepares a bounded real-data sample from SROIE and FUNSD, runs a pretrained PaddleOCR baseline on the held-out test label file, and saves measured metrics.

```bash
cd "Source code"
PYTHONPATH="src" python3 run_ocr_experiment.py --config configs/ocr/real_ocr_finetune.json
```

Outputs are saved under:

```text
artifacts/ocr/real_ocr_finetune/
```

Key files:

- `real_ocr_dataset/train_labels.txt`
- `real_ocr_dataset/val_labels.txt`
- `real_ocr_dataset/test_labels.txt`
- `real_ocr_dataset/dataset_manifest.json`
- `baseline/baseline_eval_metrics.json`
- `ocr_run_summary.json`

## Fine-Tuning

Fine-tuning is disabled by default to keep the submission runnable on a local Mac/CPU. To start a short CPU fine-tuning run, set `run_training` to `true` in `configs/ocr/real_ocr_finetune.json`.

For a smaller training smoke run, use:

```bash
cd "Source code"
PYTHONPATH="src" python3 run_ocr_experiment.py --config configs/ocr/real_ocr_finetune_train_cpu.json
```

The generated PaddleOCR config is:

```text
artifacts/ocr/real_ocr_finetune/real_rec_config.yml
```

The training command is orchestrated through `run_ocr_experiment.py` and uses the vendored PaddleOCR `tools/train.py` entry point. Pretrained PaddleOCR training weights are downloaded into `Source code/.cache/pretrain_models/` so large model archives are not committed to GitHub.

If training completes, the pipeline exports the best checkpoint and runs a post-training OCR evaluation on the same held-out test label file, saving results under `artifacts/ocr/real_ocr_finetune_train_cpu/fine_tuned/`.

## Evaluation Policy

Only metrics saved by `baseline_eval_metrics.json` or a completed post-training evaluation should be reported in final documents. Do not manually type OCR scores into the report or presentation unless they are traceable to these artifacts.

## Dataset And Copyright Notes

SROIE and FUNSD are third-party public research datasets. The team's copyright claim should cover original source code, configuration, diagrams, documentation, report text, integration workflow, and generated package structure, not ownership of third-party datasets or PaddleOCR itself.
