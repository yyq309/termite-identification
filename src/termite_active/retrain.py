from __future__ import annotations

"""Offline retraining entry point for reviewed active-learning data."""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune a detector on an incremental YOLO dataset.")
    ap.add_argument("--weights", type=Path, required=True, help="Current promoted detector checkpoint.")
    ap.add_argument("--data", type=Path, required=True, help="Incremental data.yaml.")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr0", type=float, default=0.002)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--freeze", type=int, default=0)
    ap.add_argument("--name", default="termite_active_finetune")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    model = YOLO(str(args.weights))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        freeze=args.freeze,
        name=args.name,
        seed=args.seed,
        amp=False,
        mosaic=0.5,
        close_mosaic=10,
        scale=0.4,
        translate=0.1,
        fliplr=0.5,
        flipud=0.2,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=8.0,
        plots=True,
        verbose=True,
    )
    print("Retrain done. Evaluate on fixed clean/field/false-alarm gates before promotion.")


if __name__ == "__main__":
    main()
