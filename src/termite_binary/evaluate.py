from __future__ import annotations

"""Evaluate a trained binary termite classifier.

Runs the model over an ImageFolder split (default: test) and reports the metrics
that matter for an 'is it a termite?' decision: accuracy, precision, recall, F1,
specificity, ROC-AUC, PR-AUC, a confusion matrix, and a threshold sweep so we can
pick an operating point (e.g. recall >= 0.95). Writes a JSON report + plots.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _termite_index(names: dict[int, str]) -> int:
    for i, n in names.items():
        if str(n).lower() == "termite":
            return int(i)
    raise ValueError(f"'termite' class not found in model names: {names}")


def _score_split(model: YOLO, split_dir: Path, termite_idx: int, imgsz: int, device: str):
    """Return (y_true, y_score) where y_true=1 for termite, y_score=P(termite)."""
    y_true, y_score, paths = [], [], []
    for cls_dir, label in [(split_dir / "termite", 1), (split_dir / "non_termite", 0)]:
        imgs = [p for p in sorted(cls_dir.rglob("*")) if p.suffix.lower() in IMG_EXTS]
        for i in range(0, len(imgs), 256):
            batch = [str(p) for p in imgs[i:i + 256]]
            for r in model.predict(batch, imgsz=imgsz, device=device, verbose=False):
                y_score.append(float(r.probs.data[termite_idx].item()))
                y_true.append(label)
            paths.extend(batch)
    return np.array(y_true), np.array(y_score), paths


def _metrics(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    return dict(accuracy=acc, precision=prec, recall=rec, specificity=spec, f1=f1,
                tp=tp, tn=tn, fp=fp, fn=fn)


def _sweep(y_true, y_score):
    best = {"f1": -1}
    rec95 = None
    for t in np.linspace(0.01, 0.99, 99):
        m = _metrics(y_true, (y_score >= t).astype(int))
        m["threshold"] = round(float(t), 3)
        if m["f1"] > best["f1"]:
            best = m
        if m["recall"] >= 0.95 and (rec95 is None or m["precision"] > rec95["precision"]):
            rec95 = m
    return best, rec95


def _auc(y_true, y_score):
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        return float(roc_auc_score(y_true, y_score)), float(average_precision_score(y_true, y_score))
    except Exception:
        return None, None


def _plots(y_true, y_score, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, precision_recall_curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "--", c="gray")
        ax[0].set_title("ROC"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
        ax[1].plot(rec, prec); ax[1].set_title("Precision-Recall")
        ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
        fig.tight_layout(); fig.savefig(out / "eval_curves.png", dpi=120); plt.close(fig)
    except Exception as e:
        print("plot skipped:", e)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate binary termite classifier.")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/termite_binary"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    model = YOLO(str(args.weights))
    termite_idx = _termite_index(model.names)
    split_dir = (args.data / args.split).resolve()
    y_true, y_score, paths = _score_split(model, split_dir, termite_idx, args.imgsz, args.device)
    if len(y_true) == 0:
        raise SystemExit(f"No images found in {split_dir}")

    at50 = _metrics(y_true, (y_score >= 0.5).astype(int))
    best_f1, rec95 = _sweep(y_true, y_score)
    roc_auc, pr_auc = _auc(y_true, y_score)

    report = {
        "split": args.split,
        "n_images": int(len(y_true)),
        "n_termite": int(y_true.sum()),
        "n_non_termite": int((1 - y_true).sum()),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "at_threshold_0.5": at50,
        "best_f1_operating_point": best_f1,
        "recall>=0.95_operating_point": rec95,
    }
    out = args.out or args.weights.parent.parent
    out.mkdir(parents=True, exist_ok=True)
    (out / f"eval_{args.split}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _plots(y_true, y_score, out)

    print(json.dumps(report, indent=2))
    print(f"\nSaved: {out / f'eval_{args.split}.json'}")


if __name__ == "__main__":
    main()
