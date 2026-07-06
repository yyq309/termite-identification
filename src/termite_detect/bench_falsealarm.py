from __future__ import annotations

"""Measure the FALSE-ALARM rate that actually matters for a patrolling robot dog:
how often the detector fires 'termite' on frames that contain NO termite (real
non-termite insects incl. ants = the hardest look-alike negatives). The acceptance
target is 蚁误检率 ~5%.

Two views:
  (1) hard-negative crops: fraction of non-termite insect images that yield >=1
      termite detection (per-object false-positive rate) across a conf sweep.
  (2) empty large scenes: paste several non-termite crops on a cluttered canvas
      (no termite), run SAHI, count false detections per frame — measures whether
      slicing hallucinates on clutter.
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

from sliced_infer import detect_sliced

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _empty_scene(neg_paths, rng, canvas=(1920, 1080), n=(6, 16)):
    W, H = canvas
    arr = (np.random.default_rng(rng.randint(0, 2**31)).random((H, W, 3)) * 40 + 100).astype(np.uint8)
    cv = Image.fromarray(arr).convert("RGB")
    for _ in range(rng.randint(*n)):
        p = rng.choice(neg_paths)
        try:
            c = Image.open(p).convert("RGB")
        except Exception:
            continue
        tl = rng.randint(28, 90)
        s = tl / max(c.size)
        c = c.resize((max(1, int(c.size[0] * s)), max(1, int(c.size[1] * s))))
        cv.paste(c, (rng.randint(0, W - c.size[0]), rng.randint(0, H - c.size[1])))
    return cv


def main() -> None:
    ap = argparse.ArgumentParser(description="False-alarm rate on termite-free frames.")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--neg-dir", type=Path, required=True, help="Dir of non-termite (insect) images.")
    ap.add_argument("--device", default="0")
    ap.add_argument("--confs", type=float, nargs="+", default=[0.25, 0.35, 0.45, 0.55])
    ap.add_argument("--cap", type=int, default=800)
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    model = YOLO(str(a.weights))

    negs = [p for p in sorted(a.neg_dir.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    rng.shuffle(negs)
    negs = negs[: a.cap]
    print(f"# {len(negs)} hard-negative crops | {a.scenes} empty scenes | target 蚁误检率 <= 5%\n")

    print("== (1) per-crop false positive on real non-termite insects ==")
    print(f"{'conf':>6} {'frames_fired':>13} {'FP_rate':>9}")
    for conf in a.confs:
        fired = 0
        for p in negs:
            im = Image.open(p).convert("RGB")
            r = model.predict(im, imgsz=max(320, min(im.size)), conf=conf, device=a.device, verbose=False)[0]
            if len(r.boxes):
                fired += 1
        print(f"{conf:6.2f} {fired:13d} {fired/len(negs):9.1%}")

    print("\n== (2) SAHI false detections on termite-free large scenes ==")
    scenes = [_empty_scene(negs, rng) for _ in range(a.scenes)]
    print(f"{'conf':>6} {'scenes_fired':>13} {'mean_FP/frame':>14}")
    for conf in a.confs:
        fp_counts = [len(detect_sliced(model, s, a.tile, a.overlap, conf, 0.5, a.device)) for s in scenes]
        fired = sum(1 for c in fp_counts if c > 0)
        print(f"{conf:6.2f} {fired:13d} {np.mean(fp_counts):14.2f}")


if __name__ == "__main__":
    main()
