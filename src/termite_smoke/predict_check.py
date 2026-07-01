from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def _resolve_weights(path: Path) -> Path:
    candidates = [
        path,
        Path("runs/detect/smoke_yolov8n/weights/best.pt"),
        Path("runs/detect/runs/smoke_yolov8n/weights/best.pt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Weights not found: {path}. Run train_yolo.py first.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prediction on the smoke validation images.")
    parser.add_argument("--weights", type=Path, default=Path("runs/detect/smoke_yolov8n/weights/best.pt"))
    parser.add_argument("--source", type=Path, default=Path("data/smoke_yolo/images/val"))
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", default=None, help="Optional Ultralytics project directory.")
    parser.add_argument("--name", default="smoke_predict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = _resolve_weights(args.weights)
    model = YOLO(str(weights))
    predict_kwargs = dict(
        source=str(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        name=args.name,
        save=True,
        exist_ok=True,
    )
    if args.project:
        predict_kwargs["project"] = str(Path(args.project).resolve())

    results = model.predict(**predict_kwargs)
    if results:
        print(f"Prediction check finished. Output directory: {results[0].save_dir}")


if __name__ == "__main__":
    main()
