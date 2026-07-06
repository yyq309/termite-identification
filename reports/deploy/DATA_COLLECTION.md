# Real Robot-Dog Data Collection Protocol

This project currently has a strong synthetic-data cold start. The next accuracy jump should come from real frames captured by the robot dog's own camera.

## First Batch Target

- 300-800 labelled frames from the real camera and mounting height.
- Include both termites and hard negatives: ants, beetles, soil texture, wood grain, cracks, shadows, leaves and debris.
- Capture at the actual working distance, with standing, walking and turning motion.
- Cover normal daylight, shade, low light, motion blur, defocus and JPEG compression.

## Annotation Format

Use standard YOLO detection layout:

```text
real_termite/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
  data.yaml
```

Each label line:

```text
class cx cy w h
```

For the current detector, `class 0 = termite`. If termite signs, nests or mud tubes are added later, add them as new classes instead of mixing them into class 0.

## Split And QA

- Split by scene or capture session, not by adjacent frames, to avoid leakage.
- Keep at least 20 percent for validation.
- Review boxes at full resolution; tiny targets are easy to mis-box after resizing.
- Mark ambiguous frames separately rather than forcing labels.
- Keep raw data out of Git. Commit only code, reports, aggregate metrics and a few sanitized demo images.

## Fine-Tune Command

```bash
python src/termite_detect/finetune.py \
  --weights runs/detect/termite_detect_n_hard/weights/best.pt \
  --data data/real_termite/data.yaml \
  --epochs 60 --imgsz 640 --batch 32 --device 0
```

After fine-tuning, rerun `scripts/eval_nano_hard.sh` on clean, degraded and false-alarm tests before promoting the checkpoint.
