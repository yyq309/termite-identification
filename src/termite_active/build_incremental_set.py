from __future__ import annotations

"""Build a YOLO incremental dataset from reviewed active-learning samples."""

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import yaml
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
POS_LABELS = {"termite", "positive", "pos", "1", "yes"}
NEG_LABELS = {"non_termite", "negative", "neg", "0", "no", "background"}
IGNORE_LABELS = {"", "ignore", "skip", "bad", "ambiguous", "unknown"}


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _read_review(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _norm_label(label: str) -> str:
    v = str(label or "").strip().lower()
    if v in POS_LABELS:
        return "termite"
    if v in NEG_LABELS:
        return "non_termite"
    if v in IGNORE_LABELS:
        return "ignore"
    raise ValueError(f"Unsupported human_label={label!r}; use termite, non_termite, or ignore.")


def _image_files(root: Path, split: str) -> list[Path]:
    d = root / "images" / split
    if not d.exists():
        return []
    return [p for p in sorted(d.rglob("*")) if p.suffix.lower() in IMG_EXTS]


def _copy_base(base: Path, out: Path) -> int:
    n = 0
    for split in ("train", "val", "test"):
        for img in _image_files(base, split):
            stem = f"base_{split}_{img.stem}"
            dst_img = out / "images" / split / f"{stem}{img.suffix.lower()}"
            dst_lab = out / "labels" / split / f"{stem}.txt"
            _copy_file(img, dst_img)
            lab = base / "labels" / split / f"{img.stem}.txt"
            dst_lab.parent.mkdir(parents=True, exist_ok=True)
            dst_lab.write_text(lab.read_text(encoding="utf-8") if lab.exists() else "", encoding="utf-8")
            n += 1
    return n


def _yolo_line(row: dict, img_path: Path) -> str:
    with Image.open(img_path) as im:
        width, height = im.size
    x1 = float(row.get("x1") or 0)
    y1 = float(row.get("y1") or 0)
    x2 = float(row.get("x2") or width)
    y2 = float(row.get("y2") or height)
    x1 = max(0.0, min(width - 1.0, x1)); y1 = max(0.0, min(height - 1.0, y1))
    x2 = max(x1 + 1.0, min(float(width), x2)); y2 = max(y1 + 1.0, min(float(height), y2))
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"


def _write_data_yaml(out: Path) -> None:
    data = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "termite"},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a reviewed incremental YOLO dataset.")
    ap.add_argument("--review-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-data", type=Path, default=None, help="Optional existing YOLO data root to copy first.")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-crops", action="store_true",
                    help="If a positive row has no full image, train on its crop as a whole-image box.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    base_count = _copy_base(args.base_data, args.out) if args.base_data else 0
    rows = _read_review(args.review_csv)
    rng.shuffle(rows)
    manifest = []
    added = {"termite": 0, "non_termite": 0, "ignore": 0}

    for row in rows:
        label = _norm_label(row.get("human_label", ""))
        if label == "ignore":
            added["ignore"] += 1
            continue
        split = "val" if rng.random() < args.val_ratio else "train"
        sid = row.get("sample_id") or f"row_{len(manifest):06d}"
        img_path = Path(row.get("image_path") or "")
        if not img_path.exists() and args.include_crops:
            img_path = Path(row.get("crop_path") or "")
            row = dict(row)
            row.update({"x1": 0, "y1": 0, "x2": "", "y2": ""})
        if not img_path.exists():
            print(f"skip missing image for {sid}: {row.get('image_path')}")
            continue

        dst_img = args.out / "images" / split / f"active_{sid}{img_path.suffix.lower()}"
        dst_lab = args.out / "labels" / split / f"active_{sid}.txt"
        _copy_file(img_path, dst_img)
        if label == "termite":
            if not (row.get("x2") and row.get("y2")):
                with Image.open(dst_img) as im:
                    row = dict(row)
                    row.update({"x1": 0, "y1": 0, "x2": im.width, "y2": im.height})
            dst_lab.write_text(_yolo_line(row, dst_img), encoding="utf-8")
        else:
            dst_lab.write_text("", encoding="utf-8")
        added[label] += 1
        manifest.append({
            "sample_id": sid,
            "split": split,
            "label": label,
            "source_path": row.get("source_path"),
            "image": str(dst_img),
            "label_file": str(dst_lab),
        })

    _write_data_yaml(args.out)
    (args.out / "manifest.json").write_text(json.dumps({
        "base_images_copied": base_count,
        "active_added": added,
        "samples": manifest,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Built {args.out}")
    print(f"base copied: {base_count}; active added: {added}")
    print(f"data yaml: {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
