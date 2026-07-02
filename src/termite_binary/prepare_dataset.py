from __future__ import annotations

"""Build a balanced binary (termite / non_termite) image-classification dataset in
Ultralytics YOLOv8-cls ImageFolder layout:

    <out>/train/termite/*.jpg      <out>/train/non_termite/*.jpg
    <out>/val/termite/*.jpg        <out>/val/non_termite/*.jpg
    <out>/test/termite/*.jpg       <out>/test/non_termite/*.jpg

Input is two pools of raw images (positives = termites, negatives = anything else).
Images are validated (corrupt files dropped), de-duplicated by content hash,
optionally capped/balanced, then split deterministically.
"""

import argparse
import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def _iter_images(roots: list[Path]):
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.suffix.lower() in IMG_EXTS and p.is_file():
                yield p


def _valid_and_hash(path: Path, min_side: int) -> str | None:
    """Return a content hash if the image is decodable and large enough, else None."""
    try:
        with Image.open(path) as im:
            im.verify()  # cheap integrity check
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            if min(w, h) < min_side:
                return None
            # hash on downscaled pixels so near-identical re-encodes collapse together
            small = im.resize((64, 64))
            return hashlib.md5(small.tobytes()).hexdigest()
    except Exception:
        return None


def _collect(roots: list[Path], min_side: int, cap: int, rng: random.Random) -> list[Path]:
    seen: set[str] = set()
    kept: list[Path] = []
    files = list(_iter_images(roots))
    rng.shuffle(files)
    for p in files:
        h = _valid_and_hash(p, min_side)
        if h is None or h in seen:
            continue
        seen.add(h)
        kept.append(p)
        if cap and len(kept) >= cap:
            break
    return kept


def _split(items: list[Path], val_frac: float, test_frac: float, rng: random.Random):
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test = items[:n_test]
    val = items[n_test:n_test + n_val]
    train = items[n_test + n_val:]
    return {"train": train, "val": val, "test": test}


def _copy(items: list[Path], dst_dir: Path, prefix: str) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, src in enumerate(items):
        # normalise everything to .jpg RGB so training never chokes on odd modes
        dst = dst_dir / f"{prefix}_{i:06d}.jpg"
        try:
            with Image.open(src) as im:
                im.convert("RGB").save(dst, "JPEG", quality=92)
            n += 1
        except Exception:
            continue
    return n


def build(pos_dirs, neg_dirs, out: Path, val_frac, test_frac, cap, min_side, balance, seed):
    rng = random.Random(seed)
    out = out.resolve()
    if out.exists():
        shutil.rmtree(out)

    pos = _collect([Path(p) for p in pos_dirs], min_side, cap, rng)
    neg = _collect([Path(p) for p in neg_dirs], min_side, cap, rng)
    print(f"collected: termite={len(pos)}  non_termite={len(neg)}")
    if not pos or not neg:
        raise SystemExit("Need images in BOTH the positive and negative pools.")

    if balance:
        k = min(len(pos), len(neg))
        pos, neg = pos[:k], neg[:k]
        print(f"balanced to {k} per class")

    splits = {"termite": _split(pos, val_frac, test_frac, rng),
              "non_termite": _split(neg, val_frac, test_frac, rng)}

    counts: dict[str, dict[str, int]] = defaultdict(dict)
    for cls, sp in splits.items():
        for split in SPLITS:
            n = _copy(sp[split], out / split / cls, f"{split}_{cls}")
            counts[split][cls] = n
    print("final counts:")
    for split in SPLITS:
        print(f"  {split}: {dict(counts[split])}")
    (out / "classes.txt").write_text("termite\nnon_termite\n", encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a balanced binary termite/non_termite ImageFolder dataset.")
    ap.add_argument("--pos-dirs", nargs="+", required=True, help="Directories of termite images.")
    ap.add_argument("--neg-dirs", nargs="+", required=True, help="Directories of non-termite images.")
    ap.add_argument("--out", type=Path, default=Path("data/termite_binary"))
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--cap", type=int, default=0, help="Max images collected per class before balancing (0 = no cap).")
    ap.add_argument("--min-side", type=int, default=64, help="Drop images whose shorter side is below this.")
    ap.add_argument("--no-balance", dest="balance", action="store_false", default=True)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    out = build(a.pos_dirs, a.neg_dirs, a.out, a.val_frac, a.test_frac, a.cap, a.min_side, a.balance, a.seed)
    print(f"Wrote dataset: {out}")


if __name__ == "__main__":
    main()
