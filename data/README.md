# Data Layout

This directory contains all data needed to reproduce or deploy the preprocessing models.

- `bee_direction_labeled_batch001_003/`, `bee_direction_labeled_batch004_200/`, and `bee_direction_labeled_batch001_004/`: direction classifier images and labels
- `bee_foreground_v2/`: foreground segmentation images, masks, metadata, and split files
- `reid_datasets/`: packed ReID image datasets from the source server

The ReID datasets are stored as archives to keep the repository usable. Extract them with `scripts/extract_reid_datasets.py`.
