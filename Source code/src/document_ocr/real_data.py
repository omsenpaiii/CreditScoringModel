from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


DEFAULT_DATASETS = {
    "sroie": "jsdnrs/ICDAR2019-SROIE",
    "funsd": "nielsr/funsd",
}


@dataclass(frozen=True)
class PreparedOCRDataset:
    data_dir: Path
    label_files: dict[str, Path]
    manifest_path: Path
    field_labels_path: Path | None
    counts: dict[str, int]
    sources: list[str]


def _clean_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _safe_name(text: str, max_len: int = 48) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text[:max_len] or "sample"


def _require_datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "The Hugging Face datasets package is required for real OCR data preparation. "
            "Install with: python3 -m pip install datasets pyarrow"
        ) from exc
    return load_dataset


def _as_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and "bytes" in value:
        from io import BytesIO

        return Image.open(BytesIO(value["bytes"])).convert("RGB")
    raise TypeError(f"Unsupported image value type: {type(value)!r}")


def _bbox_to_pixels(bbox: Iterable[Any], width: int, height: int) -> tuple[int, int, int, int] | None:
    values = [float(v) for v in bbox]
    if len(values) < 4:
        return None
    x0, y0, x1, y1 = values[:4]

    # FUNSD-style boxes are commonly normalized to a 0-1000 coordinate frame.
    if max(abs(v) for v in values[:4]) <= 1000 and (x1 > width or y1 > height):
        x0, x1 = x0 * width / 1000, x1 * width / 1000
        y0, y1 = y0 * height / 1000, y1 * height / 1000

    left = max(0, min(width - 1, int(round(min(x0, x1)))))
    top = max(0, min(height - 1, int(round(min(y0, y1)))))
    right = max(1, min(width, int(round(max(x0, x1)))))
    bottom = max(1, min(height, int(round(max(y0, y1)))))
    if right - left < 2 or bottom - top < 2:
        return None
    return left, top, right, bottom


def _record_words_and_boxes(record: dict[str, Any]) -> tuple[list[str], list[Any]]:
    words = record.get("words") or record.get("tokens") or []
    boxes = record.get("bboxes") or record.get("boxes") or []
    return list(words), list(boxes)


def _record_entities(record: dict[str, Any]) -> dict[str, str]:
    entities = record.get("entities")
    if isinstance(entities, dict):
        return {str(k): _clean_text(v) for k, v in entities.items() if _clean_text(v)}
    return {}


def _iter_split(dataset: Any, limit: int | None, seed: int) -> list[dict[str, Any]]:
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if limit is not None:
        indices = indices[:limit]
    return [dict(dataset[int(idx)]) for idx in indices]


def _write_label(handle, rel_path: str, text: str) -> None:
    handle.write(f"{rel_path}\t{text}\n")


def prepare_real_ocr_dataset(config: dict[str, Any]) -> PreparedOCRDataset:
    """Prepare SROIE/FUNSD crops in PaddleOCR recognition-label format.

    Output label lines follow PaddleOCR SimpleDataSet convention:
    `relative/image/path<TAB>transcript`.
    """

    load_dataset = _require_datasets()
    output_dir = Path(config["output_dir"])
    data_dir = output_dir / "real_ocr_dataset"
    images_dir = data_dir / "images"
    doc_dir = data_dir / "documents"
    images_dir.mkdir(parents=True, exist_ok=True)
    doc_dir.mkdir(parents=True, exist_ok=True)

    selected_sources = config.get("datasets", ["sroie", "funsd"])
    split_limits = config.get("split_limits", {"train": 120, "val": 40, "test": 40})
    max_regions = config.get("max_text_regions_per_split", {})
    seed = int(config.get("seed", 42))

    label_files = {
        "train": data_dir / "train_labels.txt",
        "val": data_dir / "val_labels.txt",
        "test": data_dir / "test_labels.txt",
    }
    counts = {split: 0 for split in label_files}
    manifest_records: list[dict[str, Any]] = []
    field_records: list[dict[str, Any]] = []

    handles = {split: path.open("w", encoding="utf-8") for split, path in label_files.items()}
    try:
        for source_name in selected_sources:
            dataset_id = DEFAULT_DATASETS.get(source_name, source_name)
            hf_train = load_dataset(dataset_id, split="train")
            hf_test = load_dataset(dataset_id, split="test")
            split_sources = {
                "train": _iter_split(hf_train, split_limits.get("train"), seed),
                "val": _iter_split(hf_test, split_limits.get("val"), seed + 1),
                "test": _iter_split(hf_test, split_limits.get("test"), seed + 2),
            }

            for split, records in split_sources.items():
                for doc_index, record in enumerate(records):
                    image = _as_image(record["image"])
                    width, height = image.size
                    words, boxes = _record_words_and_boxes(record)
                    if not words or not boxes:
                        continue

                    doc_stem = f"{source_name}_{split}_{doc_index:04d}"
                    document_path = doc_dir / split / f"{doc_stem}.jpg"
                    document_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(document_path, quality=90)

                    entities = _record_entities(record)
                    if entities:
                        field_records.append(
                            {
                                "source": source_name,
                                "split": split,
                                "document": str(document_path.relative_to(data_dir)),
                                "entities": entities,
                            }
                        )

                    for word_index, (word, box) in enumerate(zip(words, boxes)):
                        max_for_split = max_regions.get(split)
                        if max_for_split is not None and counts[split] >= int(max_for_split):
                            break
                        text = _clean_text(word)
                        if not text:
                            continue
                        bbox = _bbox_to_pixels(box, width, height)
                        if bbox is None:
                            continue
                        crop = image.crop(bbox)
                        rel_path = Path("images") / split / f"{doc_stem}_{word_index:04d}_{_safe_name(text)}.png"
                        crop_path = data_dir / rel_path
                        crop_path.parent.mkdir(parents=True, exist_ok=True)
                        crop.save(crop_path)
                        _write_label(handles[split], rel_path.as_posix(), text)
                        counts[split] += 1

                    manifest_records.append(
                        {
                            "source": source_name,
                            "split": split,
                            "document": str(document_path.relative_to(data_dir)),
                            "word_regions": len(words),
                            "image_size": {"width": width, "height": height},
                        }
                    )
    finally:
        for handle in handles.values():
            handle.close()

    manifest_path = data_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "PaddleOCR SimpleDataSet recognition labels",
                "sources": selected_sources,
                "counts": counts,
                "records": manifest_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    field_labels_path = data_dir / "field_labels.jsonl"
    with field_labels_path.open("w", encoding="utf-8") as handle:
        for record in field_records:
            handle.write(json.dumps(record) + "\n")

    return PreparedOCRDataset(
        data_dir=data_dir,
        label_files=label_files,
        manifest_path=manifest_path,
        field_labels_path=field_labels_path if field_records else None,
        counts=counts,
        sources=list(selected_sources),
    )
