from __future__ import annotations

"""Fetch termite (positive) and non-termite insect (negative) images for the binary
classifier, from open datasets reachable via HuggingFace.

Positives (termite): all ImageNet-21K Isoptera synsets (995 imgs), which live in just
two parquet shards of `gmongaras/Imagenet21K`. Filtered by the WordNet-ID prefix of the
row `id` field (the `class` gloss string is unreliable).

Negatives (non_termite):
  * ANTS (Formicidae) - the classic termite look-alike - from the adjacent Imagenet21K
    shards (same source, WNID-filtered).
  * A spread of other insects (beetles, flies, bees, orthoptera, roaches, odonata,
    butterflies...) from the ungated `benjamin-paine/imagenet-1k` (label 300-326).

This runs on a machine WITH internet (uses the working proxy). Output raw pools are then
fed to prepare_dataset.py. Requires: pyarrow, pillow.
"""

import argparse
import io
import os
import ssl
import urllib.request
from pathlib import Path

import certifi
import pyarrow.parquet as pq
from PIL import Image

CTX = ssl.create_default_context(cafile=certifi.where())

# ImageNet-21K WordNet IDs (verified counts in parentheses)
TERMITE_WNIDS = {
    "n02223266",  # termite, white ant (493)
    "n02223520",  # dry-wood termite (143)
    "n02224023",  # Reticulitermes lucifugus (129)
    "n02224713",  # Mastotermes darwiniensis (2)
    "n02225081",  # Mastotermes electrodominicus (5)
    "n02225798",  # powder-post termite (223)
}
ANT_WNIDS = {
    "n02219486", "n02220055", "n02220225", "n02220518", "n02220804", "n02221083",
    "n02221414", "n02221571", "n02221715", "n02221820", "n02222035", "n02222321",
    "n02222582",
}
# ImageNet-1K insect class indices (300-326), excluding 310=ant (we take ants from 21K)
INSECT_1K_LABELS = set(range(300, 327)) - {310}

I21K = "https://huggingface.co/datasets/gmongaras/Imagenet21K/resolve/main/data/train-{:05d}-of-07760.parquet"
I1K = "https://huggingface.co/datasets/benjamin-paine/imagenet-1k/resolve/main/data/train-{:05d}-of-00310.parquet"


def _download(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached {dest.name}"); return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "hf"}), context=CTX, timeout=600) as r, open(tmp, "wb") as f:
        total = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b); total += len(b)
    tmp.replace(dest)
    print(f"  got {dest.name}  {total/1e6:.0f}MB")


def _save(img_bytes: bytes, dst: Path) -> bool:
    try:
        Image.open(io.BytesIO(img_bytes)).convert("RGB").save(dst, "JPEG", quality=92)
        return True
    except Exception:
        return False


def _img_bytes(cell):
    if isinstance(cell, dict):
        return cell.get("bytes")
    return cell


def harvest_i21k(shards, cache: Path, out_termite: Path, out_ant: Path, ant_cap: int):
    n_term = n_ant = 0
    for s in shards:
        p = cache / f"i21k_{s:05d}.parquet"
        _download(I21K.format(s), p)
        pf = pq.ParquetFile(p)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["image", "id"])
            ids = t.column("id").to_pylist()
            imgs = t.column("image").to_pylist()
            for cell, idv in zip(imgs, ids):
                wnid = str(idv).split("_")[0]
                if wnid in TERMITE_WNIDS:
                    if _save(_img_bytes(cell), out_termite / f"{idv}.jpg"):
                        n_term += 1
                elif wnid in ANT_WNIDS and n_ant < ant_cap:
                    if _save(_img_bytes(cell), out_ant / f"ant_{idv}.jpg"):
                        n_ant += 1
        print(f"  shard {s}: termite={n_term} ant={n_ant}")
    return n_term, n_ant


def harvest_i1k(shards, cache: Path, out_ins: Path, cap: int):
    n = 0
    for s in shards:
        if n >= cap:
            break
        p = cache / f"i1k_{s:05d}.parquet"
        _download(I1K.format(s), p)
        pf = pq.ParquetFile(p)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["image", "label"])
            labels = t.column("label").to_pylist()
            imgs = t.column("image").to_pylist()
            for cell, lab in zip(imgs, labels):
                if int(lab) in INSECT_1K_LABELS and n < cap:
                    if _save(_img_bytes(cell), out_ins / f"ins_{s}_{lab}_{n}.jpg"):
                        n += 1
        print(f"  shard {s}: other_insects={n}")
    return n


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fetch termite + non-termite insect images from HF open datasets.")
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--cache", type=Path, default=Path("data/_parquet_cache"))
    ap.add_argument("--i21k-shards", type=int, nargs="+", default=[1279, 1280, 1281, 1282])
    ap.add_argument("--i1k-shards", type=int, nargs="+", default=[94, 97, 101])
    ap.add_argument("--ant-cap", type=int, default=700)
    ap.add_argument("--other-cap", type=int, default=700)
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    os.makedirs(a.cache, exist_ok=True)
    out_termite = a.out / "termite"
    out_neg = a.out / "non_termite"
    for d in (out_termite, out_neg):
        d.mkdir(parents=True, exist_ok=True)

    print("== ImageNet-21K: termite (positives) + ants (hard negatives) ==")
    n_term, n_ant = harvest_i21k(a.i21k_shards, a.cache, out_termite, out_neg, a.ant_cap)
    print("== ImageNet-1K: diverse insect negatives ==")
    n_other = harvest_i1k(a.i1k_shards, a.cache, out_neg, a.other_cap)

    print(f"\nDONE: termite={n_term}  non_termite(ant={n_ant} + other={n_other})={n_ant + n_other}")
    print(f"  positives -> {out_termite}")
    print(f"  negatives -> {out_neg}")


if __name__ == "__main__":
    main()
