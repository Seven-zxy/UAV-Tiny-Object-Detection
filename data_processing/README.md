# Data Processing Pipeline

Scripts for preparing the UAV aerial detection dataset used in this study.
The dataset itself is not included in this repository due to regulatory
considerations regarding the specific plant species studied.

## Pipeline Overview

```
raw images + raw labels
        |
        v
[clean_empty_annotations.py]  -> remove samples with no objects
        |
        v
[label_consistency_check.py]  -> verify YOLO label format / range
        |
        v
[split_dataset.py]            -> reproducible train/val/test split
        |
        v
[yolo_to_coco.py]             -> (optional) COCO JSON for pycocotools
        |
        v
   ./data/uav/
     ├── train/{images,labels}
     ├── val/{images,labels}
     └── test/{images,labels}
```

## Why Empty-Annotation Cleaning Matters

In tiny-object detection on aerial imagery, dataset cleaning is
disproportionately important. We observed that the raw dataset contained
empty-annotation samples in all three splits (train / val / test) —
silent contamination that depresses recall and introduces false-negative
gradient signal. After cleaning, the canonical partition used in all
reported experiments is:

| Split | Cleaned size |
|-------|--------------|
| train | 523          |
| val   | 149          |
| test  | 76           |

## Usage

```bash
# 1. Identify and remove samples without annotations
python data_processing/clean_empty_annotations.py \
    --images ./data/uav/raw/images \
    --labels ./data/uav/raw/labels \
    --apply

# 2. Verify label format and value ranges
python data_processing/label_consistency_check.py \
    --labels ./data/uav/raw/labels --nc 1

# 3. Reproducible 70/20/10 split (seed=42)
python data_processing/split_dataset.py \
    --images ./data/uav/raw/images \
    --labels ./data/uav/raw/labels \
    --out ./data/uav --seed 42

# 4. (Optional) Convert val/test to COCO JSON for pycocotools eval
python data_processing/yolo_to_coco.py \
    --images ./data/uav/test/images \
    --labels ./data/uav/test/labels \
    --names target \
    --out ./data/uav/test_coco.json
```

## Notes

- Scripts depend only on the Python standard library + Pillow.
- Random seeds are exposed as CLI arguments for full reproducibility.
- `clean_empty_annotations.py` defaults to **dry-run**. Re-run with
  `--apply` to actually delete files.