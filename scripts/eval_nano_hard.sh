#!/usr/bin/env bash
set -euo pipefail

# Evaluate the lightweight nano detector after capture-condition hardening.
# The default paths match the 4090 training box; override env vars as needed.
ROOT="${ROOT:-/root/termite}"
PY="${PY:-$ROOT/python/bin/python3}"
WEIGHTS="${WEIGHTS:-$ROOT/src/termite_detect/runs/detect/termite_detect_n_hard/weights/best.pt}"
CLEAN_DATA="${CLEAN_DATA:-$ROOT/data/termite_detect}"
HARD_DATA="${HARD_DATA:-$ROOT/data/termite_detect_hard}"
NEG_DIR="${NEG_DIR:-$ROOT/data/termite_binary/train/non_termite}"
DEVICE="${DEVICE:-1}"
ITERS="${ITERS:-25}"
WARMUP="${WARMUP:-6}"
CAP="${CAP:-600}"
SCENES="${SCENES:-30}"

cd "$ROOT/src/termite_detect"
echo "############ NANO best.pt: $WEIGHTS ############"

echo
echo "########## (1) LATENCY @1080p (full vs SAHI, FP32/FP16) ##########"
"$PY" bench_latency.py --weights "$WEIGHTS" --device "$DEVICE" --iters "$ITERS" --warmup "$WARMUP" 2>&1 \
  | grep -avE "^(WARNING|Ultralytics|YOLO|Fusing|Model summary|Speed|image|0:|requirements)" || true

echo
echo "########## (2) SAHI vs FULL on CLEAN test (conf sweep) ##########"
for c in 0.25 0.35 0.45 0.55; do
  echo "---- conf $c ----"
  "$PY" eval_sliced.py --weights "$WEIGHTS" --data "$CLEAN_DATA" --split test --conf "$c" --full-imgsz 1280 --device "$DEVICE" 2>&1 \
    | grep -aE "recall|full|sliced|==" || true
done

echo
echo "########## (3) ROBUSTNESS: SAHI on DEGRADED test ##########"
for c in 0.25 0.45; do
  echo "---- conf $c (degraded) ----"
  "$PY" eval_sliced.py --weights "$WEIGHTS" --data "$HARD_DATA" --split test_hard --conf "$c" --full-imgsz 1280 --device "$DEVICE" 2>&1 \
    | grep -aE "recall|full|sliced|==" || true
done

echo
echo "########## (4) FALSE-ALARM on real non-termite insects ##########"
"$PY" bench_falsealarm.py --weights "$WEIGHTS" --neg-dir "$NEG_DIR" --device "$DEVICE" --cap "$CAP" --scenes "$SCENES" 2>&1 \
  | grep -avE "^(WARNING|Ultralytics|YOLO|Fusing|Model summary|Speed|image|0:|requirements)" || true

echo
echo "DONE_EVAL_NANO"
