from __future__ import annotations

"""Download a diverse set of surface/texture background images (DTD) for
synthesising small-termite detection scenes. Samples N per texture category so
backgrounds span wood/cracked/grooved/porous/... surfaces a termite might sit on.
"""

import argparse
import json
import ssl
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import certifi

CTX = ssl.create_default_context(cafile=certifi.where())
REPO = "cansa/Describable-Textures-Dataset-DTD"
TREE = f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=1"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"


def _get(url, timeout=30):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "x"}), context=CTX, timeout=timeout)


def _all_files():
    """Follow HF tree API pagination (Link: rel=next) to list every file."""
    url, files = TREE, []
    while url:
        r = _get(url)
        files += [f["path"] for f in json.load(r) if f["path"].lower().endswith((".jpg", ".jpeg", ".png"))]
        nxt, link = None, r.headers.get("Link", "")
        for part in link.split(","):
            if 'rel="next"' in part and "<" in part:
                nxt = part[part.find("<") + 1:part.find(">")]
        url = nxt
    return files


def _download(path_out):
    path, out = path_out
    try:
        data = _get(RESOLVE + path, timeout=40).read()
        out.write_bytes(data)
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Download DTD texture backgrounds.")
    ap.add_argument("--out", type=Path, default=Path("data/backgrounds"))
    ap.add_argument("--per-category", type=int, default=32)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    files = _all_files()
    by_cat = defaultdict(list)
    for p in files:
        parts = p.split("/")
        cat = parts[1] if len(parts) > 2 else "misc"
        by_cat[cat].append(p)
    import random
    rng = random.Random(a.seed)
    picked = []
    for cat, ps in by_cat.items():
        rng.shuffle(ps)
        picked += ps[:a.per_category]
    print(f"{len(by_cat)} categories, downloading {len(picked)} backgrounds -> {a.out}")

    jobs = [(p, a.out / f"bg_{i:05d}.jpg") for i, p in enumerate(picked)]
    ok = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(_download, jobs):
            ok += int(r)
    print(f"downloaded {ok}/{len(jobs)} backgrounds")


if __name__ == "__main__":
    main()
