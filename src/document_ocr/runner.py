from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paddle_workflow import (
    paddle_available,
    prepare_paddleocr_workspace,
    run_subprocess,
    summarize_ocr_run,
    write_tiny_rec_config,
)
from .synthetic_data import generate_synthetic_ocr_dataset


def run_ocr_experiment(config_path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = generate_synthetic_ocr_dataset(
        output_dir=output_dir / "synthetic_dataset",
        train_count=config.get("train_count", 40),
        val_count=config.get("val_count", 10),
        seed=config.get("seed", 42),
    )

    if not paddle_available():
        summary_path = summarize_ocr_run(
            output_dir,
            status="blocked_missing_paddle",
            details={
                "dataset_info": dataset_info,
                "message": "PaddleOCR or PaddlePaddle is not installed. Workspace and synthetic data were prepared, but the fine-tuning pilot could not execute yet.",
            },
        )
        return json.loads(summary_path.read_text(encoding="utf-8"))

    workspace = prepare_paddleocr_workspace(config["workspace_root"])
    config_file = write_tiny_rec_config(
        repo_dir=workspace["repo_dir"],
        data_dir=output_dir / "synthetic_dataset",
        output_dir=output_dir,
        pretrained_model_dir=workspace["pretrained_model_dir"],
    )

    train_command = [
        config["python_executable"],
        "tools/train.py",
        "-c",
        str(config_file),
    ]
    eval_command = [
        config["python_executable"],
        "tools/eval.py",
        "-c",
        str(config_file),
        "-o",
        f"Global.checkpoints={output_dir.as_posix()}/model/best_accuracy",
    ]

    try:
        train_result = run_subprocess(train_command, cwd=workspace["repo_dir"])
        eval_result = run_subprocess(eval_command, cwd=workspace["repo_dir"])
        summary_path = summarize_ocr_run(
            output_dir,
            status="completed",
            details={
                "dataset_info": dataset_info,
                "config_file": str(config_file),
                "train_stdout_tail": train_result.stdout[-2000:],
                "eval_stdout_tail": eval_result.stdout[-2000:],
            },
        )
    except Exception as exc:
        summary_path = summarize_ocr_run(
            output_dir,
            status="failed",
            details={
                "dataset_info": dataset_info,
                "config_file": str(config_file),
                "message": str(exc),
            },
        )

    return json.loads(summary_path.read_text(encoding="utf-8"))
