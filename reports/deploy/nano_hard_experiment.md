# Nano Hardening Experiment

This is a deployment-oriented follow-up to the main YOLOv8m detector. It does not replace `models/termite_detect_yolov8m.pt`; it tests a smaller nano model plus robot-dog capture degradation for edge deployment.

## Goal

- Reduce runtime cost for edge inference.
- Make training data closer to robot-dog video frames: motion blur, defocus, uneven lighting, noise and JPEG artifacts.
- Add tools to measure the acceptance risks that matter in deployment: 1080p latency, false alarm rate and temporal alert gating.

## 4090 Run

Source dataset:

- Clean synthetic detection data: `/root/termite/data/termite_detect`
- Hardened dataset: `/root/termite/data/termite_detect_hard`

Hardened data build:

- train: 2500 images, 1987 degraded and 513 kept clean
- val: copied from clean validation split
- test: copied from clean test split
- test_hard: 60 test images degraded with probability 1.0

Training:

```bash
python train_detect.py \
  --data /root/termite/data/termite_detect_hard/data.yaml \
  --model /root/termite/yolov8n.pt \
  --epochs 120 --imgsz 640 --batch 64 --device 1 --workers 0 \
  --name termite_detect_n_hard
```

Final validation of the best checkpoint:

| model | split | precision | recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| YOLOv8n hard @640 | val | 0.609 | 0.684 | 0.624 | 0.365 |

The checkpoint remains on the 4090 server at:

```text
/root/termite/src/termite_detect/runs/detect/termite_detect_n_hard/weights/best.pt
```

It is intentionally not committed because model artifacts and Ultralytics runs are reproducible and large.

## Added Tools

- `src/termite_detect/harden_capture.py`: creates capture-degraded training and test splits while preserving YOLO boxes.
- `src/termite_detect/bench_latency.py`: measures 1080p full-frame and sliced inference latency, including FP32/FP16 timing.
- `src/termite_detect/bench_falsealarm.py`: measures false detections on termite-free hard-negative insect crops and empty cluttered scenes.
- `src/termite_detect/temporal_confirm.py`: K-of-N temporal gate for video streams, intended to suppress single-frame false positives.
- `src/termite_detect/finetune.py`: warm-start fine-tuning entry point for real labelled robot-dog frames.

## Reproduction

```bash
bash scripts/build_hard_dataset.sh

python src/termite_detect/train_detect.py \
  --data /root/termite/data/termite_detect_hard/data.yaml \
  --model /root/termite/yolov8n.pt \
  --epochs 120 --imgsz 640 --batch 64 --device 1 --workers 0 \
  --name termite_detect_n_hard

bash scripts/eval_nano_hard.sh
```

## Interpretation

The nano hard model is smaller and more deployment-friendly, but its current synthetic validation score is below the main YOLOv8m detector. The practical path is:

1. Keep YOLOv8m + SAHI as the current high-recall baseline.
2. Use nano hardening to explore edge latency and robustness.
3. Fine-tune on 300-800 real robot-dog frames.
4. Promote the lighter model only after it passes recall, false-alarm and 1080p latency checks.
