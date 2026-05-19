# Bee ReID 预处理

这个仓库用于把当前服务器上已经整理好的蜜蜂 ReID 预处理能力，迁移到另一台服务器、另一套跟踪代码、另一个 ReID 模型中使用。

核心目标：在检测框裁出蜜蜂 crop 之后、送入 ReID 模型之前，独立完成去背景、长轴旋转、头尾方向统一。

[English README](README.md)

## 仓库包含什么

- 训练好的 Unet++ 去背景模型及推理代码
- 训练好的方向分类器，以及长轴旋转、头尾方向统一代码
- 两个预处理子模型的训练脚本和训练数据
- 当前服务器上用于 ReID 训练/评估的图片数据集压缩包
- 给另一台服务器 Codex 看的交接说明：`docs/CODEX_HANDOFF.md`

新部署代码不再依赖 SAM。部分 ReID 数据集压缩包名字里仍然带有 `sam_post`，这是历史实验名称，不代表这个仓库运行时需要 SAM。

## 部署接口

预处理模块不绑定任何 tracker，也不绑定任何 ReID 模型。它接收 OpenCV crop，返回 OpenCV crop。

```python
from bee_reid_preprocessing import BeeReIDCropPreprocessor

preprocessor = BeeReIDCropPreprocessor(mode="foreground_aligned")
crop_for_reid = preprocessor.process(crop_bgr)
```

输入：

- `numpy.ndarray`
- OpenCV `BGR uint8`
- shape 通常是 `(H, W, 3)`
- 来源一般是检测框裁出来的单只蜜蜂 crop

输出：

- 仍然是 OpenCV `BGR uint8` crop
- `foreground` 模式通常保持原尺寸
- `foreground_aligned` 和 `raw_aligned` 会因为旋转对齐改变高宽
- 下游 ReID 模型仍然应该自己做 resize、normalize、tensor 转换

## 预处理模式

- `none`：原样返回，用于调试或关闭预处理
- `foreground`：只去背景
- `foreground_aligned`：去背景、长轴旋转、头尾方向统一
- `raw_aligned`：用去背景 mask 辅助旋转和判头尾，但输出对齐后的原始像素

建议：

- 如果希望 ReID 模型看到干净、方向统一的输入，优先试 `foreground_aligned`
- 如果担心去背景损失纹理细节，只想统一方向，优先试 `raw_aligned`

## 安装

```bash
git clone https://github.com/CHNyebi/bee-reid-preprocessing.git
cd bee-reid-preprocessing
git lfs install
git lfs pull
python -m pip install -r requirements.txt
```

如果在 AutoDL 上访问 GitHub 慢，可以先启用网络加速：

```bash
source /etc/network_turbo
```

安装后先跑 smoke test：

```bash
python scripts/smoke_test_preprocessing.py --device cpu
```

GPU 服务器上可以改成：

```bash
python scripts/smoke_test_preprocessing.py --device cuda
```

## 接入另一套跟踪代码

典型位置是在检测和 ReID 特征提取之间：

```python
preprocessor = BeeReIDCropPreprocessor(
    mode="foreground_aligned",
    foreground_device="cuda",
)

for det in detections:
    crop_bgr = frame_bgr[y1:y2, x1:x2]
    crop_bgr = preprocessor.process(crop_bgr)
    # 这里继续走目标 ReID 模型自己的 transform
```

注意：

- 传入的是 BGR crop，不是整张视频帧，也不是 ReID tensor
- 本仓库不负责 ReID 模型的输入尺寸
- 如果目标 ReID 模型固定输入尺寸，例如 `128x64` 或 `128x128`，应该在 ReID 模型自己的 transform 里处理

## 离线处理 ReID 图片数据集

如果要先把一批 ImageFolder 风格的 crop 全部处理好：

```bash
python scripts/preprocess_bee_reid_crops.py \
  --input-dir data/reid_extracted/train_20260501 \
  --output-dir outputs/train_20260501_foreground_aligned \
  --mode foreground_aligned \
  --foreground-device cuda \
  --batch-size 64 \
  --overwrite
```

脚本会输出：

- 处理后的图片
- `preprocess_report.csv`
- `preprocess_summary.json`

## ReID 数据集

ReID 图片数据集已经打包在：

```text
data/reid_datasets/
```

解压：

```bash
python scripts/extract_reid_datasets.py \
  --archive-dir data/reid_datasets \
  --output-dir data/reid_extracted
```

`data/reid_datasets/manifest.csv` 记录了每个压缩包的文件数、大小和 SHA256。解压脚本默认会校验 SHA256。

当前包含这些数据集：

- `eval.tar.gz`
- `eval_bee_unetpp_v2_sam_post.tar.gz`
- `eval_bee_unetpp_v2_sam_post_aligned.tar.gz`
- `eval_raw_unet_aligned.tar.gz`
- `train_20260501.tar.gz`
- `train_20260501_bee_unetpp_v2_sam_post.tar.gz`
- `train_20260501_bee_unetpp_v2_sam_post_aligned.tar.gz`
- `train_20260501_raw_unet_aligned.tar.gz`
- `val_20260501.tar.gz`
- `val_20260501_bee_unetpp_v2_sam_post.tar.gz`
- `val_20260501_bee_unetpp_v2_sam_post_aligned.tar.gz`
- `val_20260501_raw_unet_aligned.tar.gz`

## 预处理子模型训练数据

方向分类器数据：

- 标签：`data/bee_direction_labeled_batch001_004/labels.csv`
- 图片：`data/bee_direction_labeled_batch001_003/images/`
- 图片：`data/bee_direction_labeled_batch004_200/images/`
- 当前数量：500 张图片，500 条 label

去背景模型数据：

- 数据根目录：`data/bee_foreground_v2/dataset`
- train：211 张 image，211 张 mask
- val：53 张 image，53 张 mask
- 元数据：`data/bee_foreground_v2/*.csv` 和 `data/bee_foreground_v2/dataset_summary.json`

## 重新训练预处理子模型

训练方向分类器：

```bash
python scripts/train_resnet_direction.py \
  --labels-csv data/bee_direction_labeled_batch001_004/labels.csv \
  --output-model models/bee_direction_resnet18_triclass.joblib
```

训练去背景模型：

```bash
python scripts/train_bee_foreground_segmentation.py \
  --dataset-dir data/bee_foreground_v2/dataset \
  --output-dir models/bee_foreground_unetpp_resnet18_v2
```

当前默认部署权重路径是：

```text
models/bee_foreground_unetpp_resnet18_v2/best_model.pt
models/bee_direction_resnet18_triclass.joblib
```

如果重训输出到新目录，部署时需要显式传入新 checkpoint 路径，或者把默认路径更新为新权重。

## 验证清单

另一台服务器部署前建议依次跑：

```bash
python -m py_compile \
  bee_reid_preprocessing/__init__.py \
  bee_reid_preprocessing/preprocessor.py \
  bee_orientation_normalizer/__init__.py \
  bee_orientation_normalizer/orientation.py \
  scripts/*.py

python scripts/smoke_test_preprocessing.py --device cpu

python scripts/extract_reid_datasets.py \
  --archive-dir data/reid_datasets \
  --output-dir /tmp/bee_reid_dataset_check \
  --overwrite
```

如果 `models/*.pt`、`models/*.joblib` 或 `data/reid_datasets/*.tar.gz` 看起来很小，说明 Git LFS 没拉下来，需要执行：

```bash
git lfs pull
```

## 给另一台服务器 Codex 的入口

如果是让另一台服务器上的 Codex 接手，优先让它读：

```text
docs/CODEX_HANDOFF.md
```

那里写了更明确的接入位置、模型路径、数据路径、训练命令和验证步骤。
