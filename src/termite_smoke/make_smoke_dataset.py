from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


SPLITS = ("train", "val", "test")


def _ensure_split_dirs(root: Path) -> None:
    for split in SPLITS:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def _draw_termite(draw: ImageDraw.ImageDraw, rng: random.Random, size: int) -> tuple[float, float, float, float]:
    length = rng.randint(max(22, size // 11), max(30, size // 6))
    width = rng.randint(max(8, size // 42), max(12, size // 23))
    x1 = rng.randint(8, size - length - 8)
    y1 = rng.randint(8, size - width - 8)
    x2 = x1 + length
    y2 = y1 + width

    body = rng.choice([(230, 205, 150), (216, 190, 135), (238, 220, 170)])
    head = tuple(max(0, c - 22) for c in body)
    outline = (120, 93, 55)

    # A deliberately simple axis-aligned "termite": enough to test labels/training.
    draw.ellipse([x1, y1, x1 + width, y2], fill=head, outline=outline, width=1)
    draw.ellipse([x1 + width * 0.45, y1 - 1, x2 - width * 0.35, y2 + 1], fill=body, outline=outline, width=1)
    draw.ellipse([x2 - width, y1, x2, y2], fill=body, outline=outline, width=1)

    for leg_x in (x1 + length * 0.32, x1 + length * 0.52, x1 + length * 0.72):
        leg_x = int(leg_x)
        draw.line([leg_x, y1 + 2, leg_x - 5, y1 - 4], fill=outline, width=1)
        draw.line([leg_x, y2 - 2, leg_x - 5, y2 + 4], fill=outline, width=1)

    pad = 4
    return max(0, x1 - pad), max(0, y1 - pad), min(size, x2 + pad), min(size, y2 + pad)


def _make_background(rng: random.Random, size: int) -> Image.Image:
    base = Image.effect_noise((size, size), rng.uniform(35, 55)).convert("L")
    low = rng.choice([(92, 67, 44), (104, 78, 48), (117, 88, 55)])
    high = rng.choice([(188, 158, 110), (171, 139, 93), (198, 171, 126)])
    img = Image.merge(
        "RGB",
        tuple(base.point(lambda p, a=lo, b=hi: int(a + (p / 255.0) * (b - a))) for lo, hi in zip(low, high)),
    )
    img = img.filter(ImageFilter.GaussianBlur(radius=0.35))
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(10, 22)):
        x = rng.randint(0, size - 1)
        y = rng.randint(0, size - 1)
        r = rng.randint(1, 4)
        color = rng.choice([(70, 50, 32), (134, 100, 62), (154, 125, 80)])
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    return img


def _write_label(path: Path, boxes: list[tuple[float, float, float, float]], size: int) -> None:
    lines = []
    for x1, y1, x2, y2 in boxes:
        cx = ((x1 + x2) / 2) / size
        cy = ((y1 + y2) / 2) / size
        w = (x2 - x1) / size
        h = (y2 - y1) / size
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _generate_split(root: Path, split: str, count: int, size: int, rng: random.Random) -> None:
    for idx in range(count):
        image = _make_background(rng, size)
        draw = ImageDraw.Draw(image)
        boxes = []
        for _ in range(rng.randint(1, 4)):
            boxes.append(_draw_termite(draw, rng, size))

        stem = f"{split}_{idx:04d}"
        image.save(root / "images" / split / f"{stem}.jpg", quality=92)
        _write_label(root / "labels" / split / f"{stem}.txt", boxes, size)


def write_dataset_yaml(root: Path) -> None:
    yaml_text = (
        f"path: {root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: termite\n"
    )
    (root / "dataset.yaml").write_text(yaml_text, encoding="utf-8")


def build_dataset(output: Path, train: int, val: int, test: int, size: int, seed: int) -> Path:
    output = output.resolve()
    _ensure_split_dirs(output)
    rng = random.Random(seed)
    counts = {"train": train, "val": val, "test": test}
    for split in SPLITS:
        _generate_split(output, split, counts[split], size, rng)
    write_dataset_yaml(output)
    return output / "dataset.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a tiny YOLO termite smoke dataset.")
    parser.add_argument("--output", type=Path, default=Path("data/smoke_yolo"))
    parser.add_argument("--train", type=int, default=16)
    parser.add_argument("--val", type=int, default=4)
    parser.add_argument("--test", type=int, default=4)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = build_dataset(args.output, args.train, args.val, args.test, args.size, args.seed)
    print(f"Wrote smoke dataset: {data_yaml}")


if __name__ == "__main__":
    main()
