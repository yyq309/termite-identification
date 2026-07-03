from __future__ import annotations

"""Synthesise a YOLO-detection dataset for SMALL-termite detection by copy-paste:
composite real termite cutouts (small scales) + ant distractors onto varied
backgrounds, recording tight bounding boxes for termites only.

Output (Ultralytics detect layout):
  <out>/images/{train,val,test}/*.jpg
  <out>/labels/{train,val,test}/*.txt      # class cx cy w h  (normalised), class 0 = termite
  <out>/data.yaml
The `test` split uses LARGE canvases with tiny termites - the scenario SAHI slicing targets.
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _cutout(path: Path, feather: float = 0.12) -> Image.Image | None:
    """Return an RGBA cutout. If the file already has alpha (a rembg-segmented PNG),
    use it directly (trimmed); otherwise fall back to a feathered elliptical mask."""
    try:
        im = Image.open(path)
    except Exception:
        return None
    if im.mode == "RGBA":
        bb = im.split()[-1].getbbox()
        return im.crop(bb) if bb else None
    im = im.convert("RGB")
    w, h = im.size
    if min(w, h) < 24:
        return None
    mask = Image.new("L", (w, h), 0)
    m = 0.05
    ImageDraw.Draw(mask).ellipse([w * m, h * m, w * (1 - m), h * (1 - m)], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(1, int(min(w, h) * feather))))
    im.putalpha(mask)
    return im


def _prep(cutout: Image.Image, target_long: int, rng: random.Random) -> Image.Image:
    w, h = cutout.size
    s = target_long / max(w, h)
    cut = cutout.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    cut = cut.rotate(rng.uniform(0, 360), expand=True, resample=Image.BICUBIC)
    if rng.random() < 0.5:
        cut = cut.transpose(Image.FLIP_LEFT_RIGHT)
    # light photometric jitter so pastes vary and blend
    cut_rgb = cut.convert("RGB")
    cut_rgb = ImageEnhance.Brightness(cut_rgb).enhance(rng.uniform(0.75, 1.15))
    cut_rgb = ImageEnhance.Color(cut_rgb).enhance(rng.uniform(0.8, 1.2))
    cut_rgb.putalpha(cut.split()[-1])
    return cut_rgb


def _alpha_bbox(cut: Image.Image, thr: int = 40):
    a = np.array(cut.split()[-1])
    ys, xs = np.where(a > thr)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _paste(canvas: Image.Image, cut: Image.Image, x: int, y: int):
    canvas.alpha_composite(cut, (x, y))


def _compose(bg: Image.Image, canvas_wh, termites, ants, size_range, n_term, n_ant, rng):
    W, H = canvas_wh
    canvas = bg.convert("RGB").resize((W, H)).convert("RGBA")
    boxes = []
    for _ in range(n_term):
        base = rng.choice(termites)
        tl = int(rng.triangular(size_range[0], size_range[1], size_range[0]))  # bias small
        cut = _prep(base, tl, rng)
        cw, ch = cut.size
        if cw >= W or ch >= H:
            continue
        x, y = rng.randint(0, W - cw), rng.randint(0, H - ch)
        bb = _alpha_bbox(cut)
        if bb is None:
            continue
        _paste(canvas, cut, x, y)
        x1, y1, x2, y2 = x + bb[0], y + bb[1], x + bb[2], y + bb[3]
        boxes.append(((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
    for _ in range(n_ant):  # ant distractors: pasted, NOT labelled
        base = rng.choice(ants)
        cut = _prep(base, int(rng.triangular(size_range[0], size_range[1], size_range[0])), rng)
        cw, ch = cut.size
        if cw >= W or ch >= H:
            continue
        _paste(canvas, cut, rng.randint(0, W - cw), rng.randint(0, H - ch))
    return canvas.convert("RGB"), boxes


def _load_cutouts(d: Path, glob: str, cap: int, rng: random.Random):
    paths = [p for p in d.glob(glob) if p.suffix.lower() in IMG_EXTS]
    rng.shuffle(paths)
    outs = []
    for p in paths:
        c = _cutout(p)
        if c is not None:
            outs.append(c)
        if len(outs) >= cap:
            break
    return outs


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthesise small-termite YOLO detection data.")
    ap.add_argument("--termite-dir", type=Path, default=Path("data/raw/termite"))
    ap.add_argument("--neg-dir", type=Path, default=Path("data/raw/non_termite"))
    ap.add_argument("--bg-dir", type=Path, default=Path("data/backgrounds"))
    ap.add_argument("--out", type=Path, default=Path("data/termite_detect"))
    ap.add_argument("--n-train", type=int, default=2500)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=60, help="Large SAHI-style test scenes.")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    print("loading cutouts...")
    termites = _load_cutouts(a.termite_dir, "*", 995, rng)
    ants = _load_cutouts(a.neg_dir, "ant_*", 600, rng)
    bgs = [p for p in a.bg_dir.glob("*") if p.suffix.lower() in IMG_EXTS]
    print(f"termite cutouts={len(termites)} ant distractors={len(ants)} backgrounds={len(bgs)}")
    if not termites or not bgs:
        raise SystemExit("need termite cutouts and backgrounds")

    def gen(split, n, canvas_rng, size_range, term_rng, ant_rng):
        (a.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (a.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            W = rng.randrange(*canvas_rng, 32); H = rng.randrange(*canvas_rng, 32)
            bg = Image.open(rng.choice(bgs)).convert("RGB")
            img, boxes = _compose(bg, (W, H), termites, ants, size_range,
                                  rng.randint(*term_rng), rng.randint(*ant_rng), rng)
            stem = f"{split}_{i:05d}"
            img.save(a.out / "images" / split / f"{stem}.jpg", quality=90)
            with open(a.out / "labels" / split / f"{stem}.txt", "w") as f:
                for cx, cy, w, h in boxes:
                    f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        print(f"  {split}: {n} images")

    print("generating...")
    gen("train", a.n_train, (640, 1024), (14, 90), (1, 8), (0, 5))
    gen("val", a.n_val, (640, 1024), (14, 90), (1, 8), (0, 5))
    gen("test", a.n_test, (2048, 3200), (22, 70), (3, 12), (2, 10))  # large canvas, tiny termites

    yaml = (f"path: {a.out.resolve().as_posix()}\n"
            "train: images/train\nval: images/val\ntest: images/test\n"
            "names:\n  0: termite\n")
    (a.out / "data.yaml").write_text(yaml, encoding="utf-8")
    print(f"Wrote detection dataset: {a.out}")


if __name__ == "__main__":
    main()
