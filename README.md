# Termite Identification

白蚁识别训练工程的最小闭环。当前目标不是在本地训出最终模型，而是先确认数据格式、YOLO 训练调用、验证和预测链路都能跑通；正式训练再迁移到 RTX 4090 服务器。

## 当前闭环

默认 smoke 流程会自动生成一个极小的合成白蚁检测数据集：

- `data/smoke_yolo/images/{train,val,test}`
- `data/smoke_yolo/labels/{train,val,test}`
- `data/smoke_yolo/dataset.yaml`

然后使用 Ultralytics YOLO 的 `yolov8n.yaml` 从零初始化，CPU 训练 1 个 epoch，并保存一次预测可视化。

```powershell
pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File scripts/run_smoke.ps1
```

也可以分步运行：

```powershell
python src/termite_smoke/make_smoke_dataset.py --output data/smoke_yolo
python src/termite_smoke/train_yolo.py --data data/smoke_yolo/dataset.yaml --model yolov8n.yaml --epochs 1 --device cpu
python src/termite_smoke/predict_check.py --weights runs/detect/smoke_yolov8n/weights/best.pt
```

训练输出在 `runs/detect/smoke_yolov8n`，预测图在 `runs/detect/smoke_predict`。

## 数据来源策略

项目资料里提到的主数据源是 `DL-termite-identification`，约 2.4 万张白蚁个体图。这个数据源更接近分类数据，不是标准检测框数据，因此本仓库提供了一个桥接脚本，把整张图作为一个白蚁框，只用于验证训练管道：

```powershell
python scripts/prepare_dl_termite_yolo.py `
  --source-images E:\path\to\extracted\images `
  --split-dir E:\path\to\DL-termite-identification\src\training_dataset `
  --output data/dl_termite_yolo `
  --limit 40
```

正式 4090 训练建议使用真实框标注数据，或者把野外采集图用 CVAT/Label Studio/Roboflow 标成 YOLO 检测格式。

## 4090 服务器训练建议

本地 smoke 通过后，在服务器上建议从预训练权重开始：

```bash
pip install -r requirements.txt
python src/termite_smoke/train_yolo.py \
  --data data/real_termite_yolo/dataset.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 960 \
  --batch 32 \
  --device 0 \
  --workers 8 \
  --name yolov8n_termite_960
```

后续再做 `imgsz=1280`、SAHI 切片推理、负样本强化和 TensorRT 导出实验。

## 重要说明

- `data/` 和 `runs/` 默认不提交 Git，避免把训练数据和模型权重推到仓库。
- 当前合成数据只用于工程闭环，不代表真实识别效果。
- `DL-termite-identification` 的全图框转换也只是管道验证方式，不应作为最终检测标注。
