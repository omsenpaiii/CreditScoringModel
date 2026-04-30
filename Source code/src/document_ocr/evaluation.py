from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if left_char == right_char else 1),
                )
            )
        previous = current
    return previous[-1]


def read_label_file(label_file: str | Path, data_dir: str | Path) -> list[dict[str, str]]:
    data_dir = Path(data_dir)
    records: list[dict[str, str]] = []
    for line in Path(label_file).read_text(encoding="utf-8").splitlines():
        if not line.strip() or "\t" not in line:
            continue
        rel_path, text = line.split("\t", 1)
        records.append({"image_path": str(data_dir / rel_path), "relative_path": rel_path, "ground_truth": text})
    return records


def extract_paddle_text(result: Any) -> tuple[str, float | None]:
    """Extract concatenated text from PaddleOCR v3 pipeline result objects."""

    payload = result
    if hasattr(result, "json"):
        payload = result.json
    if isinstance(payload, dict) and "res" in payload:
        payload = payload["res"]
    if isinstance(payload, dict):
        texts = payload.get("rec_texts") or []
        scores = payload.get("rec_scores") or []
        text = " ".join(str(item) for item in texts if str(item).strip()).strip()
        score = None
        if scores:
            score = float(sum(float(item) for item in scores) / len(scores))
        return text, score
    return "", None


def compute_recognition_metrics(records: list[dict[str, Any]]) -> dict[str, float | int]:
    total_chars = 0
    total_char_errors = 0
    total_words = 0
    correct_words = 0
    exact_matches = 0

    for record in records:
        truth = normalize_text(record.get("ground_truth", ""))
        prediction = normalize_text(record.get("prediction", ""))
        total_chars += max(1, len(truth))
        total_char_errors += edit_distance(truth, prediction)
        truth_words = truth.split()
        pred_words = prediction.split()
        total_words += max(1, len(truth_words))
        correct_words += sum(1 for left, right in zip(truth_words, pred_words) if left == right)
        exact_matches += int(truth == prediction)

    count = len(records)
    return {
        "sample_count": count,
        "character_error_rate": total_char_errors / total_chars if total_chars else 0.0,
        "word_accuracy": correct_words / total_words if total_words else 0.0,
        "exact_match_accuracy": exact_matches / count if count else 0.0,
    }


def evaluate_predictions(label_file: str | Path, data_dir: str | Path, predictions_jsonl: str | Path, output_path: str | Path) -> dict[str, Any]:
    labels = {record["relative_path"]: record for record in read_label_file(label_file, data_dir)}
    predictions: list[dict[str, Any]] = []
    for line in Path(predictions_jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        label = labels.get(row["relative_path"])
        if not label:
            continue
        predictions.append({**label, **row})
    metrics = compute_recognition_metrics(predictions)
    result = {"status": "completed", "metrics": metrics, "predictions": predictions[:25]}
    Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def evaluate_with_paddleocr(
    label_file: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    sample_limit: int | None = 25,
    text_recognition_model_dir: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_records = read_label_file(label_file, data_dir)
    if sample_limit is not None:
        label_records = label_records[:sample_limit]

    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency guard
        result = {
            "status": "blocked_missing_paddleocr",
            "message": "Install paddlepaddle and paddleocr to run measured OCR evaluation.",
            "error": str(exc),
        }
        (output_dir / "baseline_eval_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    ocr_kwargs: dict[str, Any] = {
        "lang": "en",
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }
    if text_recognition_model_dir:
        ocr_kwargs["text_recognition_model_dir"] = text_recognition_model_dir
    ocr = PaddleOCR(**ocr_kwargs)

    prediction_path = output_dir / "paddleocr_predictions.jsonl"
    evaluated: list[dict[str, Any]] = []
    with prediction_path.open("w", encoding="utf-8") as handle:
        for record in label_records:
            result_items = ocr.predict(record["image_path"])
            texts: list[str] = []
            scores: list[float] = []
            for result_item in result_items:
                text, score = extract_paddle_text(result_item)
                if text:
                    texts.append(text)
                if score is not None:
                    scores.append(score)
            prediction = " ".join(texts).strip()
            output_record = {
                **record,
                "prediction": prediction,
                "confidence": sum(scores) / len(scores) if scores else None,
            }
            evaluated.append(output_record)
            handle.write(json.dumps({k: output_record[k] for k in ["relative_path", "prediction", "confidence"]}) + "\n")

    metrics = compute_recognition_metrics(evaluated)
    result = {
        "status": "completed",
        "engine": "PaddleOCR pretrained English OCR",
        "metrics": metrics,
        "prediction_file": str(prediction_path),
        "samples": evaluated[:15],
    }
    (output_dir / "baseline_eval_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
