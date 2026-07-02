#!/usr/bin/env bash
# Train + evaluate the binary termite classifier on the 4090 (run after server_setup.sh).
# Usage: bash scripts/server_train_eval.sh [ROOT] [MODEL] [NAME] [IMGSZ] [DEVICE]
set -euo pipefail

ROOT="${1:-/root/termite}"
MODEL="${2:-weights/yolov8s-cls.pt}"
NAME="${3:-termite_yolov8s}"
IMGSZ="${4:-224}"
DEVICE="${5:-0}"
cd "$ROOT"
DATA="data/termite_binary"

echo "== train ($MODEL @ ${IMGSZ}px on cuda:$DEVICE) =="
python3 src/termite_binary/train.py \
    --data "$DATA" --model "$MODEL" --name "$NAME" \
    --epochs 100 --imgsz "$IMGSZ" --batch 128 --device "$DEVICE"

BEST="runs/classify/$NAME/weights/best.pt"
echo "== evaluate on val =="
python3 src/termite_binary/evaluate.py --weights "$BEST" --data "$DATA" --split val  --imgsz "$IMGSZ" --device "$DEVICE"
echo "== evaluate on test =="
python3 src/termite_binary/evaluate.py --weights "$BEST" --data "$DATA" --split test --imgsz "$IMGSZ" --device "$DEVICE"
echo "== done: weights at $BEST =="
