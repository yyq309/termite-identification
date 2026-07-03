from __future__ import annotations

"""SAHI-style sliced inference for detecting SMALL termites in LARGE images.

Splits the image into overlapping tiles, runs the detector on each tile at native
resolution (so tiny termites are big enough to detect), maps boxes back to full-image
coordinates, and merges duplicates with global NMS. Self-contained (ultralytics +
numpy only) so it runs on the air-gapped server without the extra `sahi` package.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _tiles(W, H, tile, overlap):
    step = int(tile * (1 - overlap))
    xs = list(range(0, max(1, W - tile) + 1, step)) or [0]
    ys = list(range(0, max(1, H - tile) + 1, step)) or [0]
    if xs[-1] != W - tile and W > tile:
        xs.append(W - tile)
    if ys[-1] != H - tile and H > tile:
        ys.append(H - tile)
    return [(x, y) for y in ys for x in xs]


def _nms(boxes, scores, iou_thr=0.5):
    if not len(boxes):
        return []
    b = np.array(boxes, dtype=float); s = np.array(scores)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    area = (x2 - x1) * (y2 - y1)
    order = s.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1); h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def detect_sliced(model, img: Image.Image, tile, overlap, conf, iou, device):
    W, H = img.size
    boxes, scores = [], []
    # full-image pass too (catches large/obvious ones)
    passes = [(0, 0, img)]
    for (x, y) in _tiles(W, H, tile, overlap):
        passes.append((x, y, img.crop((x, y, min(x + tile, W), min(y + tile, H)))))
    for x, y, patch in passes:
        r = model.predict(patch, imgsz=tile, conf=conf, device=device, verbose=False)[0]
        for bb, sc in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            boxes.append([bb[0] + x, bb[1] + y, bb[2] + x, bb[3] + y]); scores.append(float(sc))
    keep = _nms(boxes, scores, iou)
    return [(boxes[i], scores[i]) for i in keep]


def main() -> None:
    ap = argparse.ArgumentParser(description="SAHI-style sliced termite detection on large images.")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True, help="Image file or directory.")
    ap.add_argument("--out", type=Path, default=Path("runs/sliced"))
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(a.weights))

    imgs = [a.source] if a.source.is_file() else [p for p in sorted(a.source.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    for p in imgs:
        img = Image.open(p).convert("RGB")
        dets = detect_sliced(model, img, a.tile, a.overlap, a.conf, a.iou, a.device)
        vis = img.copy(); d = ImageDraw.Draw(vis)
        for (b, s) in dets:
            d.rectangle(list(b), outline=(255, 0, 0), width=max(2, img.size[0] // 400))
            d.text((b[0], max(0, b[1] - 12)), f"{s:.2f}", fill=(255, 0, 0))
        vis.save(a.out / f"{p.stem}_sliced.jpg", quality=88)
        print(f"{p.name}: {len(dets)} termites detected -> {a.out / (p.stem + '_sliced.jpg')}")


if __name__ == "__main__":
    main()
