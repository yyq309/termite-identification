#!/usr/bin/env bash
set -euo pipefail

# Build a capture-degraded copy of the synthetic detection dataset on the 4090 box.
# Override ROOT, PY, SRC_DATA, HARD_DATA, PROB, or TEST_PROB when running elsewhere.
ROOT="${ROOT:-/root/termite}"
PY="${PY:-$ROOT/python/bin/python3}"
SRC_DATA="${SRC_DATA:-$ROOT/data/termite_detect}"
HARD_DATA="${HARD_DATA:-$ROOT/data/termite_detect_hard}"
PROB="${PROB:-0.8}"
TEST_PROB="${TEST_PROB:-1.0}"

cd "$ROOT/src/termite_detect"
mkdir -p "$HARD_DATA/images" "$HARD_DATA/labels"
rm -rf "$HARD_DATA/images"/* "$HARD_DATA/labels"/*

cp -r "$SRC_DATA/images/val" "$HARD_DATA/images/val"
cp -r "$SRC_DATA/labels/val" "$HARD_DATA/labels/val"
cp -r "$SRC_DATA/images/test" "$HARD_DATA/images/test"
cp -r "$SRC_DATA/labels/test" "$HARD_DATA/labels/test"

echo "[1/3] hardening train (prob $PROB)..."
"$PY" harden_capture.py \
  --src "$SRC_DATA/images/train" \
  --dst "$HARD_DATA/images/train" \
  --labels-src "$SRC_DATA/labels/train" \
  --labels-dst "$HARD_DATA/labels/train" \
  --prob "$PROB" \
  --seed 42

echo "[2/3] degraded test probe (prob $TEST_PROB)..."
"$PY" harden_capture.py \
  --src "$SRC_DATA/images/test" \
  --dst "$HARD_DATA/images/test_hard" \
  --labels-src "$SRC_DATA/labels/test" \
  --labels-dst "$HARD_DATA/labels/test_hard" \
  --prob "$TEST_PROB" \
  --seed 7

cat > "$HARD_DATA/data.yaml" << YAML
path: $HARD_DATA
train: images/train
val: images/val
test: images/test
names:
  0: termite
YAML

echo "[3/3] DONE_HARD"
find "$HARD_DATA/images" -maxdepth 1 -type d -printf "%f\n" | sort
