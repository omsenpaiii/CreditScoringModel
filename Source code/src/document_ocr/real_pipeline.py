from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evaluation import evaluate_with_paddleocr
from .paddle_workflow import (
    paddle_available,
    prepare_paddleocr_workspace,
    run_subprocess,
    summarize_ocr_run,
    write_real_rec_config,
)
from .real_data import PreparedOCRDataset, prepare_real_ocr_dataset


def _dataset_payload(dataset: PreparedOCRDataset) -> dict[str, Any]:
    return {
        "data_dir": str(dataset.data_dir),
        "label_files": {key: str(value) for key, value in dataset.label_files.items()},
        "manifest_path": str(dataset.manifest_path),
        "field_labels_path": str(dataset.field_labels_path) if dataset.field_labels_path else None,
        "counts": dataset.counts,
        "sources": dataset.sources,
    }


def run_real_ocr_experiment(config_path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        dataset = prepare_real_ocr_dataset(config)
    except Exception as exc:
        summary_path = summarize_ocr_run(
            output_dir,
            status="blocked_dataset_preparation",
            details={
                "message": str(exc),
                "datasets": config.get("datasets", []),
                "next_step": "Install dataset dependencies and rerun dataset preparation.",
            },
        )
        return json.loads(summary_path.read_text(encoding="utf-8"))

    summary_details: dict[str, Any] = {"dataset_info": _dataset_payload(dataset)}

    baseline_limit = config.get("baseline_sample_limit", 25)
    if config.get("run_baseline", True):
        baseline = evaluate_with_paddleocr(
            label_file=dataset.label_files["test"],
            data_dir=dataset.data_dir,
            output_dir=output_dir / "baseline",
            sample_limit=baseline_limit,
        )
        summary_details["baseline"] = baseline

    fallback_pretrained_dir = Path(config["workspace_root"]) / ".cache" / "pretrain_models" / "en_PP-OCRv3_rec_train"
    config_file = write_real_rec_config(
        data_dir=dataset.data_dir,
        output_dir=output_dir,
        pretrained_model_dir=fallback_pretrained_dir,
        epochs=int(config.get("epochs", 2)),
        batch_size=int(config.get("batch_size", 4)),
    )
    summary_details["fine_tuning_config"] = str(config_file)

    if not config.get("run_training", False):
        summary_path = summarize_ocr_run(
            output_dir,
            status="prepared_real_dataset",
            details={
                **summary_details,
                "message": "Real SROIE/FUNSD data prepared and baseline evaluation attempted. Fine-tuning is disabled in this config for CPU-friendly dry runs.",
            },
        )
        return json.loads(summary_path.read_text(encoding="utf-8"))

    if not paddle_available():
        summary_path = summarize_ocr_run(
            output_dir,
            status="blocked_missing_paddle",
            details={
                **summary_details,
                "message": "PaddleOCR/PaddlePaddle is required for fine-tuning. Dataset preparation completed.",
            },
        )
        return json.loads(summary_path.read_text(encoding="utf-8"))

    try:
        workspace = prepare_paddleocr_workspace(config["workspace_root"])
        config_file = write_real_rec_config(
            data_dir=dataset.data_dir,
            output_dir=output_dir,
            pretrained_model_dir=workspace["pretrained_model_dir"],
            epochs=int(config.get("epochs", 2)),
            batch_size=int(config.get("batch_size", 4)),
        )
        summary_details["fine_tuning_config"] = str(config_file)
        summary_details["paddleocr_repo"] = workspace["repo_dir"]
    except Exception as exc:
        summary_path = summarize_ocr_run(
            output_dir,
            status="blocked_pretrained_model_setup",
            details={
                **summary_details,
                "message": f"Dataset and baseline completed, but pretrained model setup failed: {exc}",
            },
        )
        return json.loads(summary_path.read_text(encoding="utf-8"))

    train_command = [config["python_executable"], "tools/train.py", "-c", str(config_file)]
    eval_command = [
        config["python_executable"],
        "tools/eval.py",
        "-c",
        str(config_file),
        "-o",
        f"Global.checkpoints={output_dir.as_posix()}/model/best_accuracy",
    ]
    export_dir = output_dir / "exported_rec_model"
    export_command = [
        config["python_executable"],
        "tools/export_model.py",
        "-c",
        str(config_file),
        "-o",
        f"Global.pretrained_model={output_dir.as_posix()}/model/best_accuracy",
        f"Global.save_inference_dir={export_dir.as_posix()}",
    ]

    try:
        train_result = run_subprocess(train_command, cwd=workspace["repo_dir"])
        eval_result = run_subprocess(eval_command, cwd=workspace["repo_dir"])
        export_result = run_subprocess(export_command, cwd=workspace["repo_dir"])
        fine_tuned_eval = evaluate_with_paddleocr(
            label_file=dataset.label_files["test"],
            data_dir=dataset.data_dir,
            output_dir=output_dir / "fine_tuned",
            sample_limit=config.get("post_training_sample_limit", config.get("baseline_sample_limit", 25)),
            text_recognition_model_dir=str(export_dir),
        )
        summary_path = summarize_ocr_run(
            output_dir,
            status="completed",
            details={
                **summary_details,
                "config_file": str(config_file),
                "export_dir": str(export_dir),
                "train_stdout_tail": train_result.stdout[-2500:],
                "eval_stdout_tail": eval_result.stdout[-2500:],
                "export_stdout_tail": export_result.stdout[-2500:],
                "fine_tuned": fine_tuned_eval,
            },
        )
    except Exception as exc:
        summary_path = summarize_ocr_run(
            output_dir,
            status="failed_training_or_eval",
            details={**summary_details, "config_file": str(config_file), "message": str(exc)},
        )

    return json.loads(summary_path.read_text(encoding="utf-8"))
