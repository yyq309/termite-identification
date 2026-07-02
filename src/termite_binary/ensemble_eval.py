from __future__ import annotations

"""Evaluate an ENSEMBLE of trained termite classifiers by averaging P(termite)
across models. Same metrics as evaluate.py (accuracy/precision/recall/F1/AUC +
threshold sweep). Often lifts both precision and recall over any single model.

Example:
  python src/termite_binary/ensemble_eval.py \
    --weights runs/classify/termite_m320/weights/best.pt runs/classify/termite_s224/weights/best.pt \
    --imgsz   320 224 --data data/termite_binary --split test
"""

import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _termite_index(names):
    for i, n in names.items():
        if str(n).lower() == "termite":
            return int(i)
    raise ValueError(f"'termite' not in {names}")


def _list_split(split_dir: Path):
    y, paths = [], []
    for cls_dir, label in [(split_dir / "termite", 1), (split_dir / "non_termite", 0)]:
        for p in sorted(cls_dir.rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                paths.append(str(p)); y.append(label)
    return np.array(y), paths


def _score(model, paths, ti, imgsz, device):
    out = []
    for i in range(0, len(paths), 256):
        for r in model.predict(paths[i:i + 256], imgsz=imgsz, device=device, verbose=False):
            out.append(float(r.probs.data[ti].item()))
    return np.array(out)


def _metrics(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum()); tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum()); fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    return dict(accuracy=acc, precision=prec, recall=rec, specificity=spec, f1=f1, tp=tp, tn=tn, fp=fp, fn=fn)


def _sweep(y_true, y_score):
    best = {"f1": -1}
    for t in np.linspace(0.01, 0.99, 99):
        m = _metrics(y_true, (y_score >= t).astype(int)); m["threshold"] = round(float(t), 3)
        if m["f1"] > best["f1"]:
            best = m
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Ensemble evaluation of termite classifiers.")
    ap.add_argument("--weights", type=Path, nargs="+", required=True)
    ap.add_argument("--imgsz", type=int, nargs="+", required=True)
    ap.add_argument("--data", type=Path, default=Path("data/termite_binary"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    assert len(args.weights) == len(args.imgsz), "one --imgsz per --weights"

    y_true, paths = _list_split((args.data / args.split).resolve())
    scores = []
    for w, sz in zip(args.weights, args.imgsz):
        m = YOLO(str(w))
        scores.append(_score(m, paths, _termite_index(m.names), sz, args.device))
    y_score = np.mean(scores, axis=0)

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        roc, pr = float(roc_auc_score(y_true, y_score)), float(average_precision_score(y_true, y_score))
    except Exception:
        roc = pr = None

    report = {
        "ensemble": [f"{w.parent.parent.name}@{sz}" for w, sz in zip(args.weights, args.imgsz)],
        "split": args.split, "n_images": int(len(y_true)),
        "roc_auc": roc, "pr_auc": pr,
        "at_threshold_0.5": _metrics(y_true, (y_score >= 0.5).astype(int)),
        "best_f1_operating_point": _sweep(y_true, y_score),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
