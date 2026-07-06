from __future__ import annotations

"""Collect review candidates from image folders for offline continual learning.

This script is the "online" part of the loop, but it does not update model
weights. It runs the current detector on incoming images, stores uncertain or
high-value samples, and writes a human-review CSV. Today the source can be a
plain folder of photos; later it can be the folder where the robot dog uploads
frames.
"""

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from termite_detect.sliced_infer import detect_sliced  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CSV_FIELDS = [
    "sample_id", "kind", "source_path", "image_path", "crop_path",
    "detector_score", "classifier_termite_prob", "x1", "y1", "x2", "y2",
    "image_width", "image_height", "human_label", "notes",
]


def _images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return [p for p in sorted(source.rglob("*")) if p.suffix.lower() in IMG_EXTS]


def _sample_id(path: Path, box: list[float] | None, kind: str) -> str:
    h = hashlib.sha1()
    h.update(str(path.resolve()).encode("utf-8", "ignore"))
    try:
        st = path.stat()
        h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
    except OSError:
        pass
    if box:
        h.update(",".join(f"{v:.1f}" for v in box).encode())
    h.update(kind.encode())
    return h.hexdigest()[:16]


def _existing_ids(metadata_path: Path) -> set[str]:
    ids: set[str] = set()
    if not metadata_path.exists():
        return ids
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            ids.add(json.loads(line)["sample_id"])
        except Exception:
            continue
    return ids


def _termite_index(names: dict[int, str]) -> int:
    for i, name in names.items():
        if str(name).lower() == "termite":
            return int(i)
    raise ValueError(f"'termite' class not found in classifier names: {names}")


def _classify_crop(model: YOLO | None, crop: Image.Image, imgsz: int, device: str) -> float | None:
    if model is None:
        return None
    ti = _termite_index(model.names)
    result = model.predict(crop, imgsz=imgsz, device=device, verbose=False)[0]
    return float(result.probs.data[ti].item())


def _write_review_header(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def _append_review(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    _write_review_header(path)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def _save_record(out: Path, record: dict, image: Image.Image, source: Path,
                 crop_box: list[float] | None, copy_full: bool) -> dict:
    sid = record["sample_id"]
    images_dir = out / "images"
    crops_dir = out / "crops"
    images_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    if copy_full:
        dst_img = images_dir / f"{sid}{source.suffix.lower() or '.jpg'}"
        shutil.copy2(source, dst_img)
    else:
        dst_img = source
    record["image_path"] = str(dst_img)

    if crop_box:
        x1, y1, x2, y2 = [int(round(v)) for v in crop_box]
        x1 = max(0, min(image.width - 1, x1)); y1 = max(0, min(image.height - 1, y1))
        x2 = max(x1 + 1, min(image.width, x2)); y2 = max(y1 + 1, min(image.height, y2))
        crop = image.crop((x1, y1, x2, y2))
        crop_path = crops_dir / f"{sid}.jpg"
        crop.save(crop_path, quality=92)
        record["crop_path"] = str(crop_path)
    else:
        record["crop_path"] = ""
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect active-learning candidates from images.")
    ap.add_argument("--source", type=Path, required=True, help="Image file or directory.")
    ap.add_argument("--detector", type=Path, required=True, help="YOLO detection checkpoint.")
    ap.add_argument("--classifier", type=Path, default=None, help="Optional termite/non-termite classifier.")
    ap.add_argument("--out", type=Path, default=Path("active_pool"))
    ap.add_argument("--model-version", default=None)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--low", type=float, default=0.25, help="Lower bound for uncertain detections.")
    ap.add_argument("--high", type=float, default=0.70, help="High-confidence positive candidate threshold.")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--sahi", action="store_true", help="Use sliced inference for large images.")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--classifier-imgsz", type=int, default=320)
    ap.add_argument("--copy-images", action="store_true", help="Copy full images into the pool.")
    ap.add_argument("--save-empty", action="store_true", help="Sample images with no detections as negatives.")
    ap.add_argument("--negative-every", type=int, default=20, help="Keep every Nth empty frame.")
    ap.add_argument("--allow-duplicates", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out / "metadata.jsonl"
    review_path = args.out / "review_queue.csv"
    seen = set() if args.allow_duplicates else _existing_ids(metadata_path)

    detector = YOLO(str(args.detector))
    classifier = YOLO(str(args.classifier)) if args.classifier else None
    model_version = args.model_version or args.detector.stem
    imgs = _images(args.source)
    if args.max_images:
        imgs = imgs[: args.max_images]
    if not imgs:
        raise SystemExit(f"No images found under {args.source}")

    rows: list[dict] = []
    now = datetime.now(timezone.utc).isoformat()
    kept = 0
    for idx, path in enumerate(imgs, start=1):
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            print(f"skip unreadable {path}: {exc}")
            continue

        if args.sahi:
            dets = detect_sliced(detector, image, args.tile, args.overlap, args.conf, args.iou, args.device)
        else:
            result = detector.predict(image, imgsz=args.tile, conf=args.conf, device=args.device, verbose=False)[0]
            dets = [(box.tolist(), float(score)) for box, score in zip(result.boxes.xyxy.cpu(), result.boxes.conf.cpu())]

        candidate_rows: list[dict] = []
        for box, score in dets:
            if score >= args.high:
                kind = "high_conf_positive"
            elif args.low <= score < args.high:
                kind = "uncertain_detection"
            else:
                continue
            sid = _sample_id(path, box, kind)
            if sid in seen:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            crop = image.crop((max(0, int(x1)), max(0, int(y1)), min(image.width, int(x2)), min(image.height, int(y2))))
            cls_prob = _classify_crop(classifier, crop, args.classifier_imgsz, args.device)
            record = {
                "sample_id": sid,
                "kind": kind,
                "source_path": str(path),
                "model_version": model_version,
                "created_at": now,
                "detector_score": round(score, 6),
                "classifier_termite_prob": "" if cls_prob is None else round(cls_prob, 6),
                "x1": round(x1, 2), "y1": round(y1, 2), "x2": round(x2, 2), "y2": round(y2, 2),
                "image_width": image.width, "image_height": image.height,
                "human_label": "", "notes": "",
            }
            candidate_rows.append(_save_record(args.out, record, image, path, [x1, y1, x2, y2], args.copy_images))
            seen.add(sid)

        if not dets and args.save_empty and args.negative_every > 0 and idx % args.negative_every == 0:
            kind = "empty_negative"
            sid = _sample_id(path, None, kind)
            if sid not in seen:
                record = {
                    "sample_id": sid,
                    "kind": kind,
                    "source_path": str(path),
                    "model_version": model_version,
                    "created_at": now,
                    "detector_score": "",
                    "classifier_termite_prob": "",
                    "x1": "", "y1": "", "x2": "", "y2": "",
                    "image_width": image.width, "image_height": image.height,
                    "human_label": "", "notes": "",
                }
                candidate_rows.append(_save_record(args.out, record, image, path, None, args.copy_images))
                seen.add(sid)

        rows.extend(candidate_rows)
        kept += len(candidate_rows)
        if candidate_rows:
            print(f"{path.name}: queued {len(candidate_rows)} samples")

    with metadata_path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    _append_review(review_path, rows)
    print(f"\nQueued {kept} samples -> {review_path}")
    print("Fill human_label with termite, non_termite, or ignore before building an incremental dataset.")


if __name__ == "__main__":
    main()
