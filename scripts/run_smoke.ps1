python src/termite_smoke/make_smoke_dataset.py --output data/smoke_yolo --train 16 --val 4 --test 4 --size 320
python src/termite_smoke/train_yolo.py --data data/smoke_yolo/dataset.yaml --model yolov8n.yaml --epochs 1 --imgsz 320 --batch 2 --device cpu --workers 0 --name smoke_yolov8n
python src/termite_smoke/predict_check.py --weights runs/detect/smoke_yolov8n/weights/best.pt --source data/smoke_yolo/images/val --device cpu
