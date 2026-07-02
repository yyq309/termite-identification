from __future__ import annotations

"""Compare all trained termite classifiers (singles + ensembles) on a split.
Scores each model once, caches per-image P(termite), then reports metrics for
every single model and several ensemble combinations. Picks the best by F1.
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def termite_index(names):
    for i, n in names.items():
        if str(n).lower() == "termite":
            return int(i)
    raise ValueError(names)


def list_split(split_dir: Path):
    y, paths = [], []
    for d, lab in [(split_dir / "termite", 1), (split_dir / "non_termite", 0)]:
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                paths.append(str(p)); y.append(lab)
    return np.array(y), paths


def score(model, paths, ti, imgsz, device):
    out = []
    for i in range(0, len(paths), 256):
        for r in model.predict(paths[i:i + 256], imgsz=imgsz, device=device, verbose=False):
            out.append(float(r.probs.data[ti].item()))
    return np.array(out)


def metrics(y, s, thr=0.5):
    p = (s >= thr).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    acc = (tp + tn) / max(1, len(y))
    return acc, pr, rc, f1, fp, fn


def best_f1(y, s):
    best = (-1, 0.5)
    for t in np.linspace(0.01, 0.99, 99):
        _, _, _, f1, _, _ = metrics(y, s, t)
        if f1 > best[0]:
            best = (f1, round(float(t), 3))
    return best


def auc(y, s):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs/classify"))
    ap.add_argument("--data", type=Path, default=Path("data/termite_binary"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="0")
    ap.add_argument("--models", nargs="+", default=["termite_s224:224", "termite_m320:320",
                                                    "termite_m320_v2:320", "termite_l320:320"])
    args = ap.parse_args()

    y, paths = list_split((args.data / args.split).resolve())
    probs = {}
    for spec in args.models:
        name, sz = spec.split(":")
        w = args.runs / name / "weights" / "best.pt"
        if not w.exists():
            print(f"skip {name} (no weights)"); continue
        m = YOLO(str(w))
        probs[name] = score(m, paths, termite_index(m.names), int(sz), args.device)

    print(f"\n== {args.split} set: {len(y)} imgs ({int(y.sum())} termite) ==")
    print(f"{'model/ensemble':42s} {'acc':>6} {'prec':>6} {'rec':>6} {'F1':>6} {'AUC':>6}  {'bestF1(thr)':>14}")
    rows = []
    # singles
    for name, s in probs.items():
        acc, pr, rc, f1, fp, fn = metrics(y, s)
        bf1, bt = best_f1(y, s)
        rows.append((f1, name, acc, pr, rc, f1, auc(y, s), bf1, bt))
    # ensembles: all pairs, all triples, and all
    names = list(probs)
    combos = []
    for r in (2, 3, len(names)):
        combos += list(itertools.combinations(names, r))
    for c in combos:
        s = np.mean([probs[n] for n in c], axis=0)
        acc, pr, rc, f1, fp, fn = metrics(y, s)
        bf1, bt = best_f1(y, s)
        rows.append((f1, "ENS[" + "+".join(x.replace("termite_", "") for x in c) + "]",
                     acc, pr, rc, f1, auc(y, s), bf1, bt))

    for f1, name, acc, pr, rc, _f1, au, bf1, bt in sorted(rows, key=lambda x: -x[0]):
        print(f"{name:42s} {acc:6.3f} {pr:6.3f} {rc:6.3f} {_f1:6.3f} {au:6.3f}  {bf1:6.3f}@{bt:<5}")


if __name__ == "__main__":
    main()
