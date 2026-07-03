from __future__ import annotations

"""Quantify SAHI-style slicing on the synthetic large-image test set: compare how
many tiny termites full-image inference finds vs sliced inference (recall/precision),
matching predictions to ground-truth boxes by IoU. This is the key number showing
slicing recovers small termites a single-pass detector misses.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

from sliced_infer import detect_sliced  # same package

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _load_gt(label: Path, W, H):
    boxes = []
    if label.exists():
        for ln in label.read_text().splitlines():
            p = ln.split()
            if len(p) == 5:
                cx, cy, w, h = map(float, p[1:])
                boxes.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
    return np.array(boxes) if boxes else np.zeros((0, 4))


def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (ua + 1e-9)


def _score(preds, gts, iou_thr=0.3):
    matched = set(); tp = 0
    for pb in sorted(preds, key=lambda x: -x[1]):
        best, bj = 0, -1
        for j, g in enumerate(gts):
            if j in matched:
                continue
            v = _iou(pb[0], g)
            if v > best:
                best, bj = v, j
        if best >= iou_thr and bj >= 0:
            matched.add(bj); tp += 1
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare full-image vs sliced detection on labelled test set.")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/termite_detect"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--full-imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    model = YOLO(str(a.weights))
    img_dir = a.data / "images" / a.split
    lbl_dir = a.data / "labels" / a.split
    imgs = [p for p in sorted(img_dir.rglob("*")) if p.suffix.lower() in IMG_EXTS]

    agg = {"full": [0, 0, 0], "sliced": [0, 0, 0]}
    for p in imgs:
        img = Image.open(p).convert("RGB"); W, H = img.size
        gts = _load_gt(lbl_dir / f"{p.stem}.txt", W, H)
        # full-image
        r = model.predict(img, imgsz=a.full_imgsz, conf=a.conf, device=a.device, verbose=False)[0]
        full = [([b[0], b[1], b[2], b[3]], float(s)) for b, s in
                zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
        # sliced
        sliced = detect_sliced(model, img, a.tile, a.overlap, a.conf, 0.5, a.device)
        for mode, preds in [("full", full), ("sliced", sliced)]:
            tp, fp, fn = _score(preds, gts)
            agg[mode][0] += tp; agg[mode][1] += fp; agg[mode][2] += fn

    print(f"\n== {a.split}: {len(imgs)} large images ==")
    print(f"{'mode':8} {'recall':>8} {'precision':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    for mode in ("full", "sliced"):
        tp, fp, fn = agg[mode]
        rec = tp / (tp + fn + 1e-9); prec = tp / (tp + fp + 1e-9)
        print(f"{mode:8} {rec:8.3f} {prec:10.3f} {tp:6d} {fp:6d} {fn:6d}")


if __name__ == "__main__":
    main()
