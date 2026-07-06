# Offline Continual Learning Loop

This project should not update model weights directly on the robot dog. The safer loop is:

1. Run the current detector on incoming photos or frames.
2. Save high-value samples: high-confidence detections, uncertain detections and sampled empty negatives.
3. Human-review the queue and fill `human_label`.
4. Build an incremental YOLO dataset.
5. Retrain or fine-tune offline on the 4090 server.
6. Promote a checkpoint only if fixed recall, precision, false-alarm and latency gates pass.

The current implementation works without a robot dog. Use any folder of images as the source. Later, the robot only needs to upload frames into the same input folder.

## 1. Collect Candidates

```bash
python src/termite_active/collect_candidates.py \
  --source data/incoming_photos \
  --detector models/termite_detect_yolov8m.pt \
  --classifier models/termite_binary_yolov8m.pt \
  --out active_pool \
  --sahi --conf 0.25 --low 0.25 --high 0.70 \
  --save-empty --negative-every 20 --copy-images
```

Outputs:

- `active_pool/metadata.jsonl`
- `active_pool/review_queue.csv`
- `active_pool/images/`
- `active_pool/crops/`

Fill `human_label` in the CSV with:

- `termite`
- `non_termite`
- `ignore`

## 2. Build Incremental Data

```bash
python src/termite_active/build_incremental_set.py \
  --base-data data/termite_detect \
  --review-csv active_pool/review_queue.csv \
  --out data/termite_detect_incremental \
  --val-ratio 0.2
```

Positive rows use the candidate box as the YOLO box. Negative rows create empty label files, which is important for suppressing false alarms on ants, dirt, wood grain and other look-alikes.

## 3. Offline Retraining

```bash
python src/termite_active/retrain.py \
  --weights models/termite_detect_yolov8m.pt \
  --data data/termite_detect_incremental/data.yaml \
  --epochs 50 --imgsz 640 --batch 32 --device 0 \
  --name termite_active_round01
```

## 4. Promotion Gate

Create a metrics JSON after evaluation:

```json
{
  "recall": 0.83,
  "precision": 0.74,
  "false_alarm_rate": 0.04,
  "gpu_ms": 380,
  "edge_ms": 1200
}
```

Then promote only if the candidate passes:

```bash
python src/termite_active/promote_model.py \
  --weights runs/detect/termite_active_round01/weights/best.pt \
  --metrics reports/active_round01_metrics.json \
  --name termite_active_round01 \
  --min-recall 0.80 --min-precision 0.70 \
  --max-false-alarm-rate 0.05 --max-gpu-ms 450 --max-edge-ms 1500
```

The registry is written to `models/model_registry.json`.

## Notes

- Keep raw photos, active pools and incremental datasets out of Git.
- Commit only code, aggregate reports and a few sanitized demo images.
- Always evaluate against a fixed holdout set before promoting a model.
- If model quality drops, roll back by selecting the previous entry in `model_registry.json`.
