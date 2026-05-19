# Bee Direction Labels Batch001-004

`labels.csv` combines:

- `../bee_direction_labeled_batch001_003/labels.csv`
- `../bee_direction_labeled_batch004_200/labels.csv`

The `image_path` values are repository-relative paths from this directory, so
the training command can be moved to another server without editing absolute
paths:

```bash
python scripts/train_resnet_direction.py \
  --labels-csv data/bee_direction_labeled_batch001_004/labels.csv \
  --output-model models/bee_direction_resnet18_triclass_retrained.joblib \
  --report-dir outputs/bee_direction_retrain \
  --device cuda
```
