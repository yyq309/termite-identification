from __future__ import annotations

"""Train a binary 'is it a termite?' image classifier with Ultralytics YOLOv8-cls.

Expects the dataset produced by prepare_dataset.py (train/ val/ [test/] each with
`termite/` and `non_termite/` subfolders). Uses an ImageNet-pretrained backbone by
default for a strong baseline. Designed to run on the RTX 4090 server.
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train YOLOv8-cls binary termite classifier.")
    ap.add_argument("--data", type=Path, default=Path("data/termite_binary"))
    ap.add_argument("--model", default="yolov8s-cls.pt",
                    help="Pretrained weights (.pt) or arch (.yaml). Path to a local .pt works offline.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="0", help="'0', '0,1' for DDP, or 'cpu'.")
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers. 0 avoids crashes on containers with a tiny /dev/shm (64MB).")
    ap.add_argument("--patience", type=int, default=25, help="Early-stopping patience (epochs).")
    ap.add_argument("--amp", action="store_true", default=False,
                    help="Enable AMP. Off by default: the AMP self-check needs to download a model and "
                         "hangs on an air-gapped host.")
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--dropout", type=float, default=0.1)
    # augmentation knobs that help robustness on real-world photos
    ap.add_argument("--fliplr", type=float, default=0.5)
    ap.add_argument("--flipud", type=float, default=0.2)
    ap.add_argument("--degrees", type=float, default=15.0)
    ap.add_argument("--translate", type=float, default=0.1)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--erasing", type=float, default=0.2)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--cutmix", type=float, default=0.0)
    ap.add_argument("--cos-lr", dest="cos_lr", action="store_true", default=False)
    ap.add_argument("--hsv_h", type=float, default=0.015)
    ap.add_argument("--hsv_s", type=float, default=0.5)
    ap.add_argument("--hsv_v", type=float, default=0.4)
    ap.add_argument("--project", default=None, help="Ultralytics project dir; None -> runs/classify.")
    ap.add_argument("--name", default="termite_yolov8s")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    data = a.data.resolve()
    if not (data / "train").exists() or not (data / "val").exists():
        raise FileNotFoundError(f"{data} must contain train/ and val/ (run prepare_dataset.py first)")

    model = YOLO(a.model)
    results = model.train(
        data=str(data),
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        workers=a.workers,
        amp=a.amp,
        patience=a.patience,
        lr0=a.lr0,
        dropout=a.dropout,
        fliplr=a.fliplr,
        flipud=a.flipud,
        degrees=a.degrees,
        translate=a.translate,
        scale=a.scale,
        erasing=a.erasing,
        mixup=a.mixup,
        cutmix=a.cutmix,
        cos_lr=a.cos_lr,
        hsv_h=a.hsv_h,
        hsv_s=a.hsv_s,
        hsv_v=a.hsv_v,
        project=a.project,
        name=a.name,
        seed=a.seed,
        resume=a.resume,
        verbose=True,
        plots=True,
    )
    print(f"Training finished. Best weights under: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
