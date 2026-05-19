# ReID Dataset Archives

This directory stores packed ReID image datasets from the source server. Each archive extracts to a directory with the same name as the archive basename.

Use:

```bash
python scripts/extract_reid_datasets.py \
  --archive-dir data/reid_datasets \
  --output-dir data/reid_extracted
```

`manifest.csv` records archive size, file count, and SHA256 checksum. The names containing `sam_post` are legacy dataset names; the runtime preprocessing code in this repository does not depend on SAM.
