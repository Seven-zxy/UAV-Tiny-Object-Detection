# UAV-Tiny-Object-Detection
# https://github.com/Seven-zxy/UAV-Tiny-Object-Detection
#
# YOLO -> COCO format converter.
#
# Some evaluation toolkits (notably pycocotools for mAP@[.5:.95]) require
# COCO JSON format. This script converts a YOLO-format split into a single
# COCO JSON file.
#
# Author : Xiaoyi Zheng (Zhejiang Chinese Medical University)
# License: Apache-2.0

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from PIL import Image


def yolo_to_coco_bbox(
    xc: float, yc: float, w: float, h: float, img_w: int, img_h: int,
) -> List[float]:
    """Convert YOLO normalized (xc,yc,w,h) to COCO absolute (x,y,w,h)."""
    abs_w = w * img_w
    abs_h = h * img_h
    abs_x = xc * img_w - abs_w / 2
    abs_y = yc * img_h - abs_h / 2
    return [abs_x, abs_y, abs_w, abs_h]


def build_coco(
    images_dir: Path, labels_dir: Path, class_names: List[str],
) -> dict:
    """Build a COCO-format dict from a YOLO-format split."""
    coco = {
        "info": {"description": "Converted from YOLO format."},
        "licenses": [],
        "categories": [
            {"id": i, "name": name, "supercategory": "object"}
            for i, name in enumerate(class_names)
        ],
        "images": [],
        "annotations": [],
    }

    ann_id = 1
    for img_id, img_path in enumerate(sorted(images_dir.iterdir())):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        with Image.open(img_path) as im:
            img_w, img_h = im.size
        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": img_w,
            "height": img_h,
        })
        lbl = labels_dir / f"{img_path.stem}.txt"
        if not lbl.exists():
            continue
        for line in lbl.read_text().splitlines():
            if not line.strip():
                continue
            cls, xc, yc, w, h = line.split()
            cls = int(cls)
            bbox = yolo_to_coco_bbox(
                float(xc), float(yc), float(w), float(h), img_w, img_h,
            )
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            })
            ann_id += 1
    return coco


def main():
    parser = argparse.ArgumentParser(
        description="Convert a YOLO-format split to COCO JSON."
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--names", nargs="+", default=["target"],
                        help="Class names in id order. Default: ['target'].")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output JSON file path.")
    args = parser.parse_args()

    coco = build_coco(args.images, args.labels, args.names)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(coco, indent=2))
    print(f"[OK] Wrote {len(coco['images'])} images, "
          f"{len(coco['annotations'])} annotations -> {args.out}")


if __name__ == "__main__":
    main()