from __future__ import annotations

"""Extract real termite (positive) and ant (hard-negative) images from the
iNaturalist HF dataset `sxj1215/inaturalist`, where each row carries the
scientific species name in `messages[1].content`. Filters by genus.

Adds in-the-wild, domain-diverse images on top of the ImageNet-21K pool to lift
the binary termite classifier's accuracy and robustness. Requires: fsspec, aiohttp,
pyarrow, pillow.
"""

import argparse
import io
import os
from pathlib import Path

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("CURL_CA_BUNDLE", certifi.where())

import fsspec
import pyarrow.parquet as pq
from PIL import Image

BASE = "https://huggingface.co/datasets/sxj1215/inaturalist/resolve/main/data/train-{:05d}-of-00007.parquet"

TERMITE_GENERA = {g.lower() for g in [
    "Reticulitermes", "Coptotermes", "Heterotermes", "Prorhinotermes", "Schedorhinotermes", "Rhinotermes",
    "Nasutitermes", "Trinervitermes", "Tenuirostritermes", "Cornitermes", "Procornitermes", "Syntermes",
    "Microcerotermes", "Amitermes", "Gnathamitermes", "Termes", "Odontotermes", "Macrotermes", "Microtermes",
    "Ancistrotermes", "Pseudacanthotermes", "Globitermes", "Bulbitermes", "Cryptotermes", "Incisitermes",
    "Kalotermes", "Neotermes", "Marginitermes", "Pterotermes", "Paraneotermes", "Glyptotermes", "Zootermopsis",
    "Porotermes", "Stolotermes", "Mastotermes", "Hodotermes", "Microhodotermes", "Anacanthotermes",
    "Hodotermopsis", "Archotermopsis", "Constrictotermes", "Cubitermes", "Velocitermes", "Armitermes",
    "Rhynchotermes", "Ruptitermes", "Anoplotermes", "Labiotermes", "Embiratermes", "Silvestritermes",
    "Drepanotermes", "Occasitermes", "Ephelotermes", "Australitermes", "Tumulitermes", "Sphaerotermes",
    "Apicotermes", "Psammotermes", "Serritermes", "Dolichorhinotermes", "Parrhinotermes", "Termitogeton",
]}
ANT_GENERA = {g.lower() for g in [
    "Camponotus", "Formica", "Lasius", "Solenopsis", "Pheidole", "Crematogaster", "Myrmica", "Tapinoma",
    "Linepithema", "Monomorium", "Tetramorium", "Pogonomyrmex", "Atta", "Acromyrmex", "Odontomachus",
    "Pachycondyla", "Ectatomma", "Dorymyrmex", "Aphaenogaster", "Temnothorax", "Messor", "Cataglyphis",
    "Iridomyrmex", "Nylanderia", "Brachymyrmex", "Prenolepis", "Myrmecocystus", "Dolichoderus", "Azteca",
    "Cephalotes", "Neivamyrmex", "Eciton", "Polyergus", "Formicoxenus", "Ponera", "Hypoponera", "Strumigenys",
    "Wasmannia", "Anoplolepis", "Paratrechina", "Technomyrmex", "Liometopum", "Forelius", "Myrmecia",
    "Oecophylla", "Polyrhachis", "Rhytidoponera", "Iridomyrmex", "Notoncus", "Melophorus",
]}


def _genus(msg) -> str:
    try:
        return msg[1]["content"].strip().split(" ")[0].lower()
    except Exception:
        return ""


def harvest(out_termite: Path, out_ant: Path, term_cap: int, ant_cap: int):
    fs = fsspec.filesystem("https")
    n_t = n_a = 0
    for s in range(7):
        if n_t >= term_cap and n_a >= ant_cap:
            break
        try:
            pf = pq.ParquetFile(fs.open(BASE.format(s)))
        except Exception as e:
            print(f"open shard {s} failed: {e}"); continue
        for rg in range(pf.num_row_groups):
            msgs = pf.read_row_group(rg, columns=["messages"]).column("messages").to_pylist()
            want = {i: _genus(m) for i, m in enumerate(msgs)}
            want = {i: g for i, g in want.items()
                    if (g in TERMITE_GENERA and n_t < term_cap) or (g in ANT_GENERA and n_a < ant_cap)}
            if not want:
                continue
            imgs = pf.read_row_group(rg, columns=["images"]).column("images").to_pylist()
            for i, g in want.items():
                cell = imgs[i]
                b = cell[0]["bytes"] if isinstance(cell, list) else (cell.get("bytes") if isinstance(cell, dict) else cell)
                is_t = g in TERMITE_GENERA
                dst = (out_termite if is_t else out_ant) / f"inat_{g}_{s}_{rg}_{i}.jpg"
                try:
                    Image.open(io.BytesIO(b)).convert("RGB").save(dst, "JPEG", quality=92)
                    if is_t:
                        n_t += 1
                    else:
                        n_a += 1
                except Exception:
                    pass
        print(f"shard {s}: termite={n_t} ant={n_a}", flush=True)
    return n_t, n_a


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract iNat termites + ants by genus.")
    ap.add_argument("--out", type=Path, default=Path("data/raw_inat"))
    ap.add_argument("--term-cap", type=int, default=100000)
    ap.add_argument("--ant-cap", type=int, default=1500)
    a = ap.parse_args()
    ot = a.out / "termite"; oa = a.out / "ant"
    ot.mkdir(parents=True, exist_ok=True); oa.mkdir(parents=True, exist_ok=True)
    n_t, n_a = harvest(ot, oa, a.term_cap, a.ant_cap)
    print(f"\nDONE: iNat termite={n_t}  iNat ant={n_a}\n  {ot}\n  {oa}")


if __name__ == "__main__":
    main()
