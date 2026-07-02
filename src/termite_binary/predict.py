from __future__ import annotations

"""Run the trained binary classifier on new images: is it a termite?

Prints, per image, the predicted label and the termite probability, applying a
decision threshold (default 0.5; tune it from evaluate.py's threshold sweep).
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _termite_index(names: dict[int, str]) -> int:
    for i, n in names.items():
        if str(n).lower() == "termite":
            return int(i)
    raise ValueError(f"'termite' class not found in model names: {names}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict termite / non-termite for images.")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True, help="Image file or directory.")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--threshold", type=float, default=0.5, help="P(termite) >= threshold -> termite.")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    model = YOLO(str(args.weights))
    ti = _termite_index(model.names)

    if args.source.is_dir():
        imgs = [str(p) for p in sorted(args.source.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    else:
        imgs = [str(args.source)]
    if not imgs:
        raise SystemExit(f"No images under {args.source}")

    n_termite = 0
    for i in range(0, len(imgs), 256):
        for r in model.predict(imgs[i:i + 256], imgsz=args.imgsz, device=args.device, verbose=False):
            p = float(r.probs.data[ti].item())
            is_termite = p >= args.threshold
            n_termite += int(is_termite)
            label = "termite (白蚁)" if is_termite else "non_termite (非白蚁)"
            print(f"{Path(r.path).name:40s}  P(termite)={p:.3f}  ->  {label}")
    print(f"\n{n_termite}/{len(imgs)} predicted termite (threshold={args.threshold})")


if __name__ == "__main__":
    main()
