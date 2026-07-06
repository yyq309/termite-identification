from __future__ import annotations

"""Measure single-frame inference latency at 1080p, to check the acceptance budget
(GPU <= 450 ms / 1080p, edge <= 1500 ms) — including the SAHI slicing overhead the
alignment slide explicitly flagged ("SAHI 切片会增加 1080p GPU 推理耗时, 需实测确认").

Reports mean/median ms and FPS for:
  - full-frame inference at several imgsz
  - SAHI sliced (tile x N + full pass)
for FP32 and FP16. Pure PyTorch timing on the measurement GPU; a TensorRT-INT8 engine
on the actual Jetson will be faster (see reports/deploy for the export command + the
projection factor used here).
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

from sliced_infer import _tiles, _nms


def _sync(device):
    if str(device) != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def time_full(model, img, imgsz, conf, device, half, iters, warmup):
    for _ in range(warmup):
        model.predict(img, imgsz=imgsz, conf=conf, device=device, half=half, verbose=False)
    _sync(device)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict(img, imgsz=imgsz, conf=conf, device=device, half=half, verbose=False)
        _sync(device)
        ts.append((time.perf_counter() - t0) * 1000)
    return np.array(ts)


def time_sliced(model, img, tile, overlap, conf, device, half, iters, warmup):
    W, H = img.size
    tiles = _tiles(W, H, tile, overlap)
    n_pass = len(tiles) + 1  # + full-image pass

    def one():
        boxes, scores = [], []
        for patch, ox, oy in [(img, 0, 0)] + [(img.crop((x, y, min(x + tile, W), min(y + tile, H))), x, y) for x, y in tiles]:
            r = model.predict(patch, imgsz=tile, conf=conf, device=device, half=half, verbose=False)[0]
            for bb, sc in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                boxes.append([bb[0] + ox, bb[1] + oy, bb[2] + ox, bb[3] + oy]); scores.append(float(sc))
        _nms(boxes, scores, 0.5)

    for _ in range(warmup):
        one()
    _sync(device)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        one()
        _sync(device)
        ts.append((time.perf_counter() - t0) * 1000)
    return np.array(ts), n_pass


def main() -> None:
    ap = argparse.ArgumentParser(description="1080p single-frame latency benchmark (full vs SAHI).")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--jetson-factor", type=float, default=6.0,
                    help="Rough 4090->Jetson Orin NX slowdown for FP16 (documented estimate).")
    a = ap.parse_args()

    dev = a.device
    img = Image.fromarray((np.random.default_rng(0).random((a.height, a.width, 3)) * 255).astype(np.uint8))
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() and dev != "cpu" else "cpu"
    print(f"# device={gpu}  frame={a.width}x{a.height}  budget: GPU<=450ms  edge<=1500ms\n")

    for half in (False, True):
        tag = "FP16" if half else "FP32"
        model = YOLO(str(a.weights))
        print(f"== {tag} ==")
        for imgsz in (640, 960, 1280):
            t = time_full(model, img, imgsz, a.conf, dev, half, a.iters, a.warmup)
            print(f"  full   imgsz={imgsz:<4}  {t.mean():7.1f} ms  (med {np.median(t):6.1f})  {1000/t.mean():5.1f} FPS")
        t, npass = time_sliced(model, img, a.tile, a.overlap, a.conf, dev, half, a.iters, a.warmup)
        j = t.mean() * a.jetson_factor
        print(f"  SAHI   tile={a.tile} ov={a.overlap} ({npass} passes)  {t.mean():7.1f} ms  (med {np.median(t):6.1f})  {1000/t.mean():5.1f} FPS")
        print(f"         -> est. Jetson Orin NX FP16 ~{j:6.0f} ms  ({'PASS' if j <= 1500 else 'OVER'} edge<=1500ms; TensorRT-INT8 will be faster)\n")


if __name__ == "__main__":
    main()
