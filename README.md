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

---

# 白蚁二分类识别（是不是白蚁）

在 smoke 闭环之外，本仓库提供一个真正训练用的**二分类**管道：给一张图，判断**是不是白蚁**（termite / non_termite），
在 RTX 4090 服务器上训练，追求高准确率、高召回率与鲁棒性。

## 为什么是分类而不是检测

论文数据源 `DL-termite-identification`（4 物种 × 2 品级 × 3 组 × 1000 = 2.4 万张个体图）本质是**分类**数据，
且其 GitHub Git-LFS 对象已失效（`images.tgz` 返回 404），无法直接获取。因此：

- 任务改为"是不是白蚁"的二分类，正类=白蚁，负类=**蚂蚁**（最易混淆的 look-alike）+ 其他昆虫。
- 数据从可达的开放源（HuggingFace / GitHub）抓取，见 `src/termite_binary/fetch_data.py`。

## 目录布局（Ultralytics YOLOv8-cls ImageFolder）

```
data/termite_binary/
  train/termite/*.jpg   train/non_termite/*.jpg
  val/termite/*.jpg     val/non_termite/*.jpg
  test/termite/*.jpg    test/non_termite/*.jpg
```

## 本地流程（有网机器）

```bash
# 1) 抓取正/负样本原图到 raw/ 下
python src/termite_binary/fetch_data.py --out data/raw
# 2) 校验、去重、均衡并切分为上面的 ImageFolder 结构
python src/termite_binary/prepare_dataset.py \
  --pos-dirs data/raw/termite --neg-dirs data/raw/non_termite --out data/termite_binary
```

## 断网 4090 服务器：离线装环境

服务器完全断网（只有系统 Python 3.8，无 pip/torch）。在有网机器上把 cp38 / linux_x86_64 / cu121 的
torch + torchvision + ultralytics + 全部依赖打成 wheel 包，scp 过去离线安装：

```bash
# 服务器上（bundle 已 scp 到 /root/termite）
bash scripts/server_setup.sh /root/termite     # get-pip.py 离线引导 + pip 离线安装 + 验证 CUDA
```

## 训练 + 评估（4090）

```bash
python src/termite_binary/train.py \
  --data data/termite_binary --model weights/yolov8s-cls.pt \
  --epochs 100 --imgsz 224 --batch 128 --device 0 --name termite_yolov8s

python src/termite_binary/evaluate.py \
  --weights runs/classify/termite_yolov8s/weights/best.pt \
  --data data/termite_binary --split test --device 0
```

`evaluate.py` 输出 accuracy / precision / recall / F1 / specificity / ROC-AUC / PR-AUC、混淆矩阵，
并做阈值扫描给出"recall ≥ 0.95"等工作点，用于把"是不是白蚁"的判定阈值调到高标准。

## 数据来源（真实、可复现，全部来自可达的开放源）

- **正类 白蚁**：ImageNet-21K 的全部 Isoptera（白蚁）同义词集，共 995 张真实图，托管于 HuggingFace
  `gmongaras/Imagenet21K`，按 `id` 字段的 WordNet-ID 前缀过滤（`n02223266` 等 6 个 synset），只需下载 2 个 parquet 分片。
- **负类 非白蚁**：
  - **蚂蚁 (Formicidae)** —— 最易混淆的 look-alike，取自同源 ImageNet-21K 相邻分片（13 个蚂蚁 WNID）。
  - **其他昆虫** —— 甲虫/苍蝇/蜜蜂/蟑螂/螳螂/蜻蜓/蝴蝶等，取自 `benjamin-paine/imagenet-1k`（类别 300–326）。
- 经去重、去损坏、按最短边过滤后均衡为 **926 : 926**，切分 train/val/test = 650/138/138 每类。
- 原始的 `DL-termite-identification`（2.4 万张）因其 GitHub Git-LFS 对象已失效（404）而无法获取，故改用上述等价开放源。

## 结果（RTX 4090，YOLOv8m-cls @ 320，阈值 0.5）

最优单模型 `models/termite_binary_yolov8m.pt`（在 val 与 test 上一致，稳健）：

| split | accuracy | precision | recall | F1 | ROC-AUC |
|-------|---------:|----------:|-------:|-----:|--------:|
| val   | 0.931 | 0.910 | 0.957 | 0.933 | 0.980 |
| test  | 0.960 | 0.944 | 0.978 | 0.961 | 0.987 |

- 召回率 0.96–0.98：极少漏判白蚁；ROC-AUC 0.98–0.99：类别可分性强、稳健。
- 对比过 yolov8s/m/l-cls、224/320 分辨率、mixup/cutmix/cos-lr 以及多种集成（`compare_models.py`）；
  在这份真实数据上 **yolov8m-cls@320（基础增强，不加 mixup）** 综合最优。集成 `m320+m320_v2` 的 ROC-AUC 最高（0.992），可作为更稳健的备选。
- 训练曲线/混淆矩阵/PR-ROC 曲线见 `reports/`。

## 推理：判断一张图是不是白蚁

```bash
python src/termite_binary/predict.py \
  --weights models/termite_binary_yolov8m.pt --source path/to/img_or_dir --imgsz 320 --threshold 0.5
```

## 断网服务器说明

服务器完全断网（仅系统 Python 3.8，无 pip/torch）。整套 GPU PyTorch 栈通过在有网机器上下载 cp38/linux/cu121
的 wheel 包 + 自带的 python-build-standalone 3.8，scp 过去用 `scripts/server_setup.sh` 离线安装。容器 `/dev/shm`
仅 64MB，故训练用 `--workers 0`；离线环境下 AMP 自检需联网下载模型，故默认 `--amp` 关闭。
