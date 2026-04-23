from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve


PADDLEOCR_REPO = "https://github.com/PaddlePaddle/PaddleOCR.git"
PPOCR_PRETRAIN_URL = "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar"


def paddle_available() -> bool:
    try:
        import paddle  # noqa: F401
        import paddleocr  # noqa: F401
    except Exception:
        return False
    return True


def prepare_paddleocr_workspace(base_dir: str | Path) -> dict[str, str]:
    base_dir = Path(base_dir)
    repo_dir = base_dir / "third_party" / "PaddleOCR"
    pretrain_dir = base_dir / "third_party" / "pretrain_models"
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    pretrain_dir.mkdir(parents=True, exist_ok=True)

    if not repo_dir.exists():
        subprocess.run(["git", "clone", "--depth", "1", PADDLEOCR_REPO, str(repo_dir)], check=True)

    tar_path = pretrain_dir / "en_PP-OCRv3_rec_train.tar"
    model_dir = pretrain_dir / "en_PP-OCRv3_rec_train"
    if not model_dir.exists():
        if not tar_path.exists():
            urlretrieve(PPOCR_PRETRAIN_URL, tar_path)
        with tarfile.open(tar_path) as archive:
            archive.extractall(pretrain_dir)

    return {
        "repo_dir": str(repo_dir),
        "pretrain_dir": str(pretrain_dir),
        "pretrained_model_dir": str(model_dir),
    }


def write_tiny_rec_config(repo_dir: str | Path, data_dir: str | Path, output_dir: str | Path, pretrained_model_dir: str | Path) -> Path:
    repo_dir = Path(repo_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "tiny_rec_config.yml"
    config_text = f"""Global:
  use_gpu: False
  epoch_num: 1
  log_smooth_window: 10
  print_batch_step: 2
  save_model_dir: {output_dir.as_posix()}/model
  save_epoch_step: 1
  eval_batch_step: [0, 2]
  cal_metric_during_train: True
  pretrained_model: {Path(pretrained_model_dir).as_posix()}/best_accuracy
  character_dict_path: ppocr/utils/en_dict.txt
  max_text_length: 40
  infer_mode: False
  use_space_char: True
Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    learning_rate: 0.00005
  regularizer:
    name: L2
    factor: 0.0
Architecture:
  model_type: rec
  algorithm: CRNN
  Backbone:
    name: MobileNetV1Enhance
    scale: 0.5
    last_conv_stride: [1, 2]
    last_pool_type: avg
  Neck:
    name: SequenceEncoder
    encoder_type: rnn
    hidden_size: 48
  Head:
    name: CTCHead
    fc_decay: 0.00004
Loss:
  name: CTCLoss
PostProcess:
  name: CTCLabelDecode
Metric:
  name: RecMetric
  main_indicator: acc
Train:
  dataset:
    name: SimpleDataSet
    data_dir: {Path(data_dir).as_posix()}
    label_file_list:
      - {Path(data_dir).joinpath("train_labels.txt").as_posix()}
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: False
      - RecConAug: {{}}
      - RecAug: {{}}
      - MultiLabelEncode: {{}}
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys: ['image', 'label_ctc', 'length', 'label']
  loader:
    shuffle: True
    batch_size_per_card: 8
    drop_last: False
    num_workers: 0
Eval:
  dataset:
    name: SimpleDataSet
    data_dir: {Path(data_dir).as_posix()}
    label_file_list:
      - {Path(data_dir).joinpath("val_labels.txt").as_posix()}
    transforms:
      - DecodeImage:
          img_mode: BGR
          channel_first: False
      - MultiLabelEncode: {{}}
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys: ['image', 'label_ctc', 'length', 'label']
  loader:
    shuffle: False
    drop_last: False
    batch_size_per_card: 8
    num_workers: 0
"""
    config_path.write_text(config_text, encoding="utf-8")
    return config_path


def run_subprocess(command: list[str], cwd: str | Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(command, cwd=str(cwd), env=merged_env, check=True, capture_output=True, text=True)


def summarize_ocr_run(output_dir: str | Path, status: str, details: dict[str, Any]) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "ocr_run_summary.json"
    summary_path.write_text(json.dumps({"status": status, **details}, indent=2), encoding="utf-8")
    return summary_path
