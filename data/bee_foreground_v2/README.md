# Bee Foreground v2 Dataset

这个目录是当前训练用的人工修正子集，共 264 张：

- `dataset/train/images`：训练图片，211 张
- `dataset/train/masks`：训练 mask
- `dataset/val/images`：验证图片，53 张
- `dataset/val/masks`：验证 mask
- `annotation_registry.csv`：已标注样本登记表，用于以后从原始数据池继续抽图时去重
- `usable_annotations.csv`：合并后的可用标注明细
- `train.csv` / `val.csv`：训练和验证拆分明细

mask 是单通道二值图，非零像素表示 `bee`，0 表示背景。
