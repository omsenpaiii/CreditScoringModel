from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WORDS = [
    "Applicant",
    "Income",
    "Loan",
    "Risk",
    "Verified",
    "Pending",
    "Statement",
    "Borrower",
    "Salary",
    "Score",
]

NAMES = [
    "Rudraksh Dhamija",
    "Ava Carter",
    "Noah Singh",
    "Mia Taylor",
    "Liam Joseph",
]


def generate_synthetic_ocr_dataset(output_dir: str | Path, train_count: int = 40, val_count: int = 10, seed: int = 42) -> dict[str, str]:
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    font = ImageFont.load_default()

    manifests = {}
    for split_name, count in [("train", train_count), ("val", val_count)]:
        lines = []
        split_dir = image_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            text = _make_sample_text(rng)
            image_path = split_dir / f"{split_name}_{index:03d}.png"
            _render_text_image(text, image_path, font)
            rel_path = image_path.relative_to(output_dir)
            lines.append(f"{rel_path}\t{text}")
        manifest_path = output_dir / f"{split_name}_labels.txt"
        manifest_path.write_text("\n".join(lines), encoding="utf-8")
        manifests[split_name] = str(manifest_path)

    metadata_path = output_dir / "dataset_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "train_count": train_count,
                "val_count": val_count,
                "note": "Synthetic OCR snippets used only for a preliminary PaddleOCR fine-tuning pilot.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifests["metadata"] = str(metadata_path)
    return manifests


def _make_sample_text(rng: random.Random) -> str:
    return f"{rng.choice(WORDS)} {rng.choice(NAMES)} {rng.randint(1200, 9800)}"


def _render_text_image(text: str, output_path: Path, font: ImageFont.ImageFont) -> None:
    image = Image.new("RGB", (520, 64), color="white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([(0, 0), (519, 63)], outline="black", width=1)
    draw.text((12, 22), text, font=font, fill="black")
    image.save(output_path)
