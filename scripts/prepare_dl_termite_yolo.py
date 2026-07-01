from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


SPLIT_FILES = {
    "train": "BENCHMARK-LARGE_Ind_trainfile_1_sub200.txt",
    "val": "BENCHMARK-LARGE_Ind_valfile_1_sub200.txt",
    "test": "BENCHMARK-LARGE_Ind_testfile_1.txt",
}


def _read_split_file(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            rows.append((Path(parts[0]).name, parts[1]))
    return rows


def _index_images(images_root: Path) -> dict[str, Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    index: dict[str, Path] = {}
    for path in images_root.rglob("*"):
        if path.suffix.lower() in exts:
            index.setdefault(path.name, path)
    return index


def _write_dataset_yaml(root: Path) -> None:
    text = (
        f"path: {root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: termite\n"
    )
    (root / "dataset.yaml").write_text(text, encoding="utf-8")


def prepare_dataset(source_images: Path, split_dir: Path, output: Path, limit: int, seed: int) -> Path:
    rng = random.Random(seed)
    image_index = _index_images(source_images)
    if not image_index:
        raise FileNotFoundError(f"No images found under {source_images}")

    for split, filename in SPLIT_FILES.items():
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
        rows = _read_split_file(split_dir / filename)
        rng.shuffle(rows)
        copied = 0
        for image_name, _class_id in rows:
            src = image_index.get(image_name)
            if src is None:
                continue
            stem = Path(image_name).stem
            dst = output / "images" / split / src.name
            shutil.copy2(src, dst)
            # DL-termite-identification is a classification dataset. This full-frame
            # box is only a pipeline bridge, not a real detection annotation.
            (output / "labels" / split / f"{stem}.txt").write_text("0 0.5 0.5 1.0 1.0\n", encoding="utf-8")
            copied += 1
            if copied >= limit:
                break
        print(f"{split}: copied {copied} images")

    _write_dataset_yaml(output)
    return output / "dataset.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a tiny DL-termite-identification sample to YOLO format.")
    parser.add_argument("--source-images", type=Path, required=True, help="Extracted image directory from images.tgz.")
    parser.add_argument("--split-dir", type=Path, required=True, help="DL-termite-identification/src/training_dataset.")
    parser.add_argument("--output", type=Path, default=Path("data/dl_termite_yolo"))
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = prepare_dataset(args.source_images, args.split_dir, args.output, args.limit, args.seed)
    print(f"Wrote YOLO dataset: {data_yaml}")


if __name__ == "__main__":
    main()
