from __future__ import annotations

"""Harden a YOLO-detection dataset against REAL robot-dog capture conditions.

The detector is trained on clean copy-paste composites, but a walking robot dog
produces frames with motion blur, defocus, uneven outdoor lighting, sensor noise
and JPEG artefacts. This script applies randomised *photometric + blur* degradations
(NO geometric change, so boxes/labels stay valid) to narrow the clean->field gap
without needing any real field images.

Usage:
  python harden_capture.py --src <dataset>/images/train --dst <hard>/images/train \
      --labels-src <dataset>/labels/train --labels-dst <hard>/labels/train \
      --prob 0.8 --seed 42
"""

import argparse
import io
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _motion_blur(im: Image.Image, length: int, angle_deg: float) -> Image.Image:
    """Directional (linear) motion blur via a line kernel — mimics dog gait / panning.

    Uses cv2.filter2D (fast, memory-light) so it scales to multi-megapixel frames;
    falls back to a plain Gaussian blur if OpenCV is unavailable.
    """
    length = max(3, length | 1)  # odd
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    try:
        import cv2
        rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
        k = cv2.warpAffine(k, rot, (length, length))
        s = k.sum()
        if s <= 0:
            return im
        k /= s
        arr = np.asarray(im.convert("RGB"))
        out = cv2.filter2D(arr, -1, k, borderType=cv2.BORDER_REFLECT)
        return Image.fromarray(out)
    except Exception:
        return im.filter(ImageFilter.GaussianBlur(length / 6.0))


def degrade(im: Image.Image, rng: random.Random) -> Image.Image:
    im = im.convert("RGB")
    W, H = im.size
    long_side = max(W, H)
    # --- blur: motion OR defocus (mutually exclusive, most frames get one) ---
    r = rng.random()
    if r < 0.45:
        length = int(rng.uniform(0.004, 0.014) * long_side)
        im = _motion_blur(im, length, rng.uniform(0, 180))
    elif r < 0.75:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 2.2)))
    # --- lighting: brightness / contrast / gamma (outdoor sun & shade) ---
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.55, 1.45))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.7, 1.35))
    im = ImageEnhance.Color(im).enhance(rng.uniform(0.7, 1.3))
    if rng.random() < 0.5:
        g = rng.uniform(0.7, 1.6)
        lut = [min(255, int((i / 255.0) ** (1.0 / g) * 255)) for i in range(256)] * 3
        im = im.point(lut)
    # --- distance / sensor: downscale-upscale ---
    if rng.random() < 0.5:
        f = rng.uniform(0.5, 0.85)
        im = im.resize((max(1, int(W * f)), max(1, int(H * f))), Image.BILINEAR).resize((W, H), Image.BILINEAR)
    # --- sensor noise ---
    if rng.random() < 0.6:
        arr = np.asarray(im, np.float32)
        arr += rng.uniform(2, 12) * np.random.default_rng(rng.randint(0, 2**31)).standard_normal(arr.shape)
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    # --- JPEG recompression artefacts ---
    if rng.random() < 0.7:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=rng.randint(28, 72))
        buf.seek(0)
        im = Image.open(buf).convert("RGB")
    return im


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply capture-condition degradations to a detection split.")
    ap.add_argument("--src", type=Path, required=True, help="images/<split> dir")
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--labels-src", type=Path, default=None)
    ap.add_argument("--labels-dst", type=Path, default=None)
    ap.add_argument("--prob", type=float, default=0.8, help="Per-image probability of degradation.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quality", type=int, default=92)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    a.dst.mkdir(parents=True, exist_ok=True)
    if a.labels_dst:
        a.labels_dst.mkdir(parents=True, exist_ok=True)
    imgs = [p for p in sorted(a.src.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    n_deg = 0
    for p in imgs:
        im = Image.open(p).convert("RGB")
        if rng.random() < a.prob:
            im = degrade(im, rng)
            n_deg += 1
        im.save(a.dst / f"{p.stem}.jpg", quality=a.quality)
        if a.labels_src and a.labels_dst:
            lab = a.labels_src / f"{p.stem}.txt"
            (a.labels_dst / f"{p.stem}.txt").write_text(lab.read_text() if lab.exists() else "", encoding="utf-8")
    print(f"hardened {len(imgs)} images ({n_deg} degraded, {len(imgs)-n_deg} clean) -> {a.dst}")


if __name__ == "__main__":
    main()
