from __future__ import annotations

"""Train a YOLOv8 DETECTOR for small termites on the synthetic dataset.
Runs on the RTX 4090 (offline: workers=0 for tiny /dev/shm, amp off)."""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train YOLOv8-detect for small termites.")
    ap.add_argument("--data", type=Path, default=Path("data/termite_detect/data.yaml"))
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=768, help="Match ~ the SAHI tile size.")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--amp", action="store_true", default=False)
    ap.add_argument("--name", default="termite_detect_s")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    model = YOLO(a.model)
    results = model.train(
        data=str(a.data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        device=a.device, workers=a.workers, amp=a.amp, patience=a.patience,
        name=a.name, seed=a.seed,
        # small-object friendly augmentation
        mosaic=1.0, close_mosaic=15, scale=0.5, translate=0.1, fliplr=0.5, flipud=0.3,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, degrees=10.0,
        verbose=True, plots=True,
    )
    print(f"Detector trained. Best weights: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
