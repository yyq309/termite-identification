from __future__ import annotations

"""Segment termite / ant crops into clean transparent RGBA cutouts with rembg
(U2Net), trimmed to the object. These replace the crude elliptical masks so pasted
insects look real (no colored-background halos) and the detector transfers better.
"""

import argparse
from pathlib import Path

from PIL import Image
from rembg import new_session, remove

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Make transparent RGBA cutouts via rembg.")
    ap.add_argument("--src-dir", type=Path, required=True)
    ap.add_argument("--glob", default="*")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cap", type=int, default=100000)
    ap.add_argument("--model", default="u2netp", help="u2netp (fast) or u2net (better).")
    ap.add_argument("--min-alpha-frac", type=float, default=0.02,
                    help="Drop cutouts whose foreground covers < this fraction (bad segmentation).")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    session = new_session(a.model)

    paths = [p for p in sorted(a.src_dir.glob(a.glob)) if p.suffix.lower() in IMG_EXTS][:a.cap]
    print(f"segmenting {len(paths)} crops with {a.model} -> {a.out_dir}")
    kept = 0
    for i, p in enumerate(paths):
        try:
            rgba = remove(Image.open(p).convert("RGB"), session=session)  # RGBA
            alpha = rgba.split()[-1]
            bb = alpha.getbbox()
            if bb is None:
                continue
            cut = rgba.crop(bb)
            w, h = cut.size
            import numpy as np
            frac = (np.array(cut.split()[-1]) > 40).mean()
            if w < 16 or h < 16 or frac < a.min_alpha_frac:
                continue
            cut.save(a.out_dir / f"{p.stem}.png")
            kept += 1
        except Exception:
            continue
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(paths)} (kept {kept})", flush=True)
    print(f"done: {kept} cutouts -> {a.out_dir}")


if __name__ == "__main__":
    main()
