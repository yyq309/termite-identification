from __future__ import annotations

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small YOLO training loop.")
    parser.add_argument("--data", type=Path, default=Path("data/smoke_yolo/dataset.yaml"))
    parser.add_argument("--model", default="yolov8n.yaml", help="Use yolov8n.yaml for no download, yolov8n.pt for pretrained.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default=None, help="Optional Ultralytics project directory.")
    parser.add_argument("--name", default="smoke_yolov8n")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exist-ok", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {args.data}. Run make_smoke_dataset.py first.")

    model = YOLO(args.model)
    train_kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name=args.name,
        exist_ok=args.exist_ok,
        seed=args.seed,
        verbose=True,
    )
    if args.project:
        train_kwargs["project"] = str(Path(args.project).resolve())

    results = model.train(**train_kwargs)
    print(f"Training finished. Results directory: {results.save_dir}")


if __name__ == "__main__":
    main()
