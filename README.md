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

---

# 大图小目标白蚁检测 + SAHI 切片

二分类回答"这张图是不是白蚁"，但真实场景里**白蚁往往是大图中的一个小点**（墙面、木梁、地基照片里只占 1–2%），
整图一次推理容易漏检。这条线做**目标检测 + SAHI 切片推理**（Slicing Aided Hyper Inference）：把大图切成重叠小块分别检测、
再全局 NMS 合并，专门找回微小白蚁。代码在 `src/termite_detect/`。

## 为什么用合成数据

公开源没有**带检测框**的白蚁数据（二分类那批是整图分类图，无框）。因此用 **copy-paste 合成**造带框数据：
把**真实白蚁抠图**贴到**真实纹理背景**上，坐标即真值框。蚂蚁作为**未标注干扰物**贴入，逼模型学会区分 look-alike。

```bash
# 1) 抠图：rembg(U2Net) 把白蚁/蚂蚁分割成干净透明 RGBA（去除 studio 背景光晕）
python src/termite_detect/make_cutouts.py --src-dir data/raw/termite --out-dir data/cutouts/termite --model u2netp
# 2) 背景：下载 DTD 表面纹理（木纹/裂纹/墙面/多孔…）
python src/termite_detect/fetch_backgrounds.py --out data/backgrounds
# 3) 合成 YOLO 检测集（train/val 小画布，test 用大画布+微小白蚁 = SAHI 目标场景）
python src/termite_detect/make_synthetic.py \
  --termite-dir data/cutouts/termite --neg-dir data/cutouts/ant \
  --bg-dir data/backgrounds --out data/termite_detect
```

规模：957 白蚁 + 590 蚂蚁抠图，1503 背景 → train 2500 / val 400 / **test 60 张大图（2048–3200px，白蚁仅 22–70px）**。

## 训练（RTX 4090，YOLOv8-detect @768，120 epochs）

```bash
python src/termite_detect/train_detect.py \
  --data data/termite_detect/data.yaml --model weights/yolov8m.pt \
  --epochs 120 --imgsz 768 --batch 24 --device 0 --workers 0 --name termite_detect_m
```

| 模型 | mAP50 | mAP50-95 | precision | recall |
|------|------:|---------:|----------:|-------:|
| yolov8s @768 | 0.780 | 0.517 | 0.716 | 0.788 |
| **yolov8m @768** | **0.798** | 0.537 | 0.776 | 0.738 |

最终检测器 `models/termite_detect_yolov8m.pt`（best.pt 峰值 mAP50 ≈ 0.80）。

## SAHI 切片 vs 整图（60 张大图，核心证据）

**切片在每个置信度阈值都召回更高，且阈值越高差距越大**——整图把大图缩到 1280 后小白蚁置信度低、一提阈值就漏掉；
切片让每个目标在近原分辨率下被检测，召回稳。

| conf | 整图 recall/prec | 切片 recall/prec | Δrecall | 少漏白蚁 |
|-----:|:---------------:|:----------------:|--------:|:-------:|
| 0.25 | 0.920 / 0.615 | **0.945** / 0.524 | +2.5pp | 35→24 |
| 0.45 | 0.684 / 0.747 | **0.815** / 0.724 | +13.1pp | 138→81 |
| 0.55 | 0.540 / 0.787 | **0.737 / 0.799** | +19.7pp | 201→115 |

conf=0.55 时切片在**召回和精度上同时胜出**。虫害排查优先召回 → 切片 @0.25 召回 **0.945**（437 只只漏 24 只）。
完整四档扫描、训练曲线、demo 可视化见 [`reports/detect/`](reports/detect/)（`detect_eval.md`）。

## 切片推理（自包含，仅需 ultralytics + numpy，无需安装 sahi）

```bash
python src/termite_detect/sliced_infer.py \
  --weights models/termite_detect_yolov8m.pt \
  --source path/to/large_image.jpg --out runs/sliced \
  --tile 640 --overlap 0.25 --conf 0.3 --device 0
```

`eval_sliced.py` 在带标注 test 集上量化"切片 vs 整图"的召回/精度，即上表。

## 局限

- 检测器训练于**合成数据**（真实抠图 + 真实纹理背景），指标为**合成 test 集**上的值；迁移到真实照片存在域差，
  建议用少量真实标注微调。背景用 DTD 纹理，非全部真实栖息场景，可替换以进一步缩小域差。

---

# 部署化增强：误报、时延、真实数据微调

主检测器 `termite_detect_yolov8m.pt` 仍是当前高召回基线；4090 上进一步补了面向机器狗部署的工程化工具：

- `harden_capture.py`：对合成检测数据加入运动模糊、失焦、光照、噪声、JPEG 等退化，模拟机器狗行走采集画面。
- `bench_latency.py`：实测 1080p 整图和 SAHI 切片推理时延，覆盖 FP32/FP16。
- `bench_falsealarm.py`：在无白蚁的蚂蚁/其他昆虫负样本和大图空场景上测误报。
- `temporal_confirm.py`：视频流 K-of-N 时序确认门控，降低单帧误报触发报警的概率。
- `finetune.py`：接入真实机器狗标注图像后的低学习率微调入口。

轻量化探索里还训练了一版 YOLOv8n hard 模型：在退化增强数据上 120 epochs，val `mAP50 ≈ 0.624`、`mAP50-95 ≈ 0.365`。
它用于部署时延和鲁棒性探索，不替代当前 YOLOv8m 主模型。详见：

- [`reports/deploy/nano_hard_experiment.md`](reports/deploy/nano_hard_experiment.md)
- [`reports/deploy/DATA_COLLECTION.md`](reports/deploy/DATA_COLLECTION.md)

复现实验脚本：

```bash
bash scripts/build_hard_dataset.sh
bash scripts/eval_nano_hard.sh
```

---

# 离线持续学习 / 主动学习闭环

目前还没有机器狗时，可以先用普通图片目录模拟机器狗回传照片，跑通“采样 -> 人工复核 -> 增量数据集 -> 离线微调 -> 指标达标后晋级”的闭环。
代码在 `src/termite_active/`，完整流程见 [`reports/active_learning.md`](reports/active_learning.md)。

```bash
# 1) 从图片目录收集高价值候选和空场景负样本
python src/termite_active/collect_candidates.py \
  --source data/incoming_photos \
  --detector models/termite_detect_yolov8m.pt \
  --classifier models/termite_binary_yolov8m.pt \
  --out active_pool --sahi --save-empty --copy-images

# 2) 人工填写 active_pool/review_queue.csv 的 human_label 后，构建增量数据集
python src/termite_active/build_incremental_set.py \
  --base-data data/termite_detect \
  --review-csv active_pool/review_queue.csv \
  --out data/termite_detect_incremental

# 3) 离线微调
python src/termite_active/retrain.py \
  --weights models/termite_detect_yolov8m.pt \
  --data data/termite_detect_incremental/data.yaml \
  --name termite_active_round01
```

现场不建议让机器狗直接在线改权重；机器狗只负责采集和回传，4090 服务器离线重训，新模型通过召回、精度、误报率和时延门槛后再发布。
