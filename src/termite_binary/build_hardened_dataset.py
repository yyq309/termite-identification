from __future__ import annotations

"""Build a HARDENED binary dataset that stresses the termite/ant boundary.

- Negatives are ANT-HEAVY (ants are the classic termite look-alike).
- A held-out ANTS-ONLY bucket (`hard_ants/`) is reserved to measure the model's
  false-positive rate on the hardest distractor - a real robustness stress test.

Input: a raw termite dir + a raw non_termite dir whose files are prefixed
`ant_*` (Formicidae) and `ins_*` (other insects), as produced by fetch_data.py.
Output layout (YOLOv8-cls ImageFolder + an extra flat hard bucket):
  <out>/train|val|test/{termite,non_termite}/*.jpg
  <out>/hard_ants/*.jpg          # ants only, never seen in training
"""

import argparse
import hashlib
import random
import shutil
from pathlib import Path

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def _valid_hash(path: Path, min_side: int):
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im = im.convert("RGB")
            if min(im.size) < min_side:
                return None
            return hashlib.md5(im.resize((64, 64)).tobytes()).hexdigest()
    except Exception:
        return None


def _collect(paths, min_side, rng):
    seen, kept = set(), []
    paths = list(paths); rng.shuffle(paths)
    for p in paths:
        h = _valid_hash(p, min_side)
        if h and h not in seen:
            seen.add(h); kept.append(p)
    return kept


def _copy(items, dst, prefix):
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, src in enumerate(items):
        try:
            with Image.open(src) as im:
                im.convert("RGB").save(dst / f"{prefix}_{i:06d}.jpg", "JPEG", quality=92)
            n += 1
        except Exception:
            pass
    return n


def _split3(items, val_frac, test_frac, rng):
    items = list(items); rng.shuffle(items)
    n = len(items); nt = int(n * test_frac); nv = int(n * val_frac)
    return {"test": items[:nt], "val": items[nt:nt + nv], "train": items[nt + nv:]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build ant-heavy hardened termite dataset + ants-only hard bucket.")
    ap.add_argument("--termite-dir", type=Path, default=Path("data/raw/termite"))
    ap.add_argument("--neg-dir", type=Path, default=Path("data/raw/non_termite"))
    ap.add_argument("--out", type=Path, default=Path("data/termite_hard"))
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--ant-frac", type=float, default=0.6, help="Ant share of the negative class.")
    ap.add_argument("--hard-ants", type=int, default=250, help="Ants held out for the hard bucket.")
    ap.add_argument("--min-side", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = a.out.resolve()
    if out.exists():
        shutil.rmtree(out)

    term = _collect(a.termite_dir.rglob("*"), a.min_side, rng)
    ants = _collect([p for p in a.neg_dir.glob("ant_*") if p.suffix.lower() in IMG_EXTS], a.min_side, rng)
    ins = _collect([p for p in a.neg_dir.glob("ins_*") if p.suffix.lower() in IMG_EXTS], a.min_side, rng)
    print(f"collected: termite={len(term)} ant={len(ants)} insect={len(ins)}")

    # reserve ants-only hard bucket (held out from everything else)
    hard = ants[:a.hard_ants]; ants = ants[a.hard_ants:]
    _copy(hard, out / "hard_ants", "hardant")
    print(f"hard_ants bucket: {len(hard)}")

    # balance negatives to #termite, ant-heavy
    k = len(term)
    n_ant = min(len(ants), int(k * a.ant_frac))
    n_ins = min(len(ins), k - n_ant)
    neg = ants[:n_ant] + ins[:n_ins]; rng.shuffle(neg)
    print(f"negatives: {len(neg)} ({n_ant} ant + {n_ins} insect) vs termite {k}")

    ts = _split3(term, a.val_frac, a.test_frac, rng)
    ns = _split3(neg, a.val_frac, a.test_frac, rng)
    for sp in SPLITS:
        nt = _copy(ts[sp], out / sp / "termite", f"{sp}_t")
        nn = _copy(ns[sp], out / sp / "non_termite", f"{sp}_n")
        print(f"  {sp}: termite={nt} non_termite={nn}")
    print(f"Wrote hardened dataset: {out}")


if __name__ == "__main__":
    main()
