from __future__ import annotations

"""Data engine: one-command DOMAIN fine-tune on real field images.

The synthetic-trained nano is the cold-start model. The single biggest accuracy
lever for real deployment is fine-tuning on real robot-dog frames. This script is
the plug: point it at a small folder of real, labelled images (standard YOLO
layout) and it fine-tunes the synthetic checkpoint at a low LR, warm-starting from
everything the model already learned.

Real-data layout expected (same as Ultralytics):
  real/images/{train,val}/*.jpg
  real/labels/{train,val}/*.txt   # class cx cy w h  (class 0 = termite; add signs as new classes)
  real/data.yaml

Recommended first batch: 300-800 labelled frames captured by the dog's own camera
at the real working distance. See reports/deploy/DATA_COLLECTION.md for the protocol.

  python finetune.py --weights <synthetic_nano>/best.pt --data real/data.yaml \
      --epochs 60 --imgsz 640 --freeze 0 --name termite_nano_real
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune the synthetic nano on real field images.")
    ap.add_argument("--weights", type=Path, required=True, help="Synthetic-pretrained nano best.pt")
    ap.add_argument("--data", type=Path, required=True, help="Real-data data.yaml")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr0", type=float, default=0.003, help="Low LR: warm-start, don't wipe synthetic priors.")
    ap.add_argument("--freeze", type=int, default=0, help="Freeze first N layers (e.g. 10) if data is scarce.")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--name", default="termite_nano_real")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    model = YOLO(str(a.weights))
    model.train(
        data=str(a.data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        lr0=a.lr0, freeze=a.freeze, device=a.device, workers=a.workers, amp=False,
        patience=a.patience, name=a.name, seed=a.seed,
        # keep capture-condition robustness on; real data is already noisy so ease geometric aug
        mosaic=0.5, close_mosaic=10, scale=0.4, translate=0.1, fliplr=0.5, flipud=0.2,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, degrees=8.0, verbose=True, plots=True,
    )
    print("Domain fine-tune done. Re-run bench_latency / bench_falsealarm / eval_sliced on the new best.pt.")


if __name__ == "__main__":
    main()
