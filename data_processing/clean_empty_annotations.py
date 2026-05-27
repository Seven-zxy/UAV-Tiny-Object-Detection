# UAV-Tiny-Object-Detection
# https://github.com/Seven-zxy/UAV-Tiny-Object-Detection
#
# Empty annotation cleaner for YOLO-format datasets.
#
# Tiny-object detection datasets are particularly vulnerable to
# empty-annotation contamination: images without any object instance
# silently inflate the training set while contributing only false-negative
# gradient signal. This script identifies and (optionally) removes such
# samples.
#
# Author : Xiaoyi Zheng (Zhejiang Chinese Medical University)
# License: Apache-2.0

import argparse
from pathlib import Path
from typing import List, Tuple


def find_empty_annotations(
    images_dir: Path,
    labels_dir: Path,
    image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> Tuple[List[Path], List[Path]]:
    """Scan a YOLO-format split and return paths with empty / missing labels.

    A label is considered "empty" if:
      - The corresponding .txt file does not exist, OR
      - The .txt file exists but contains zero non-blank lines.

    Args:
        images_dir: Directory containing image files.
        labels_dir: Directory containing YOLO-format .txt label files.
        image_exts: Valid image file extensions.

    Returns:
        (empty_images, empty_labels): paired lists of files to be cleaned.
    """
    empty_images, empty_labels = [], []
    images = [p for p in images_dir.iterdir()
              if p.suffix.lower() in image_exts]

    for img in images:
        lbl = labels_dir / f"{img.stem}.txt"
        if not lbl.exists():
            empty_images.append(img)
            empty_labels.append(lbl)
            continue
        with lbl.open() as f:
            content = [ln for ln in f.read().splitlines() if ln.strip()]
        if not content:
            empty_images.append(img)
            empty_labels.append(lbl)

    return empty_images, empty_labels


def main():
    parser = argparse.ArgumentParser(
        description="Identify empty-annotation samples in a YOLO dataset."
    )
    parser.add_argument("--images", type=Path, required=True,
                        help="Directory containing image files.")
    parser.add_argument("--labels", type=Path, required=True,
                        help="Directory containing YOLO .txt label files.")
    parser.add_argument("--apply", action="store_true",
                        help="Actually delete empty image/label pairs. "
                             "Default is dry-run.")
    args = parser.parse_args()

    empty_imgs, empty_lbls = find_empty_annotations(args.images, args.labels)
    n = len(empty_imgs)
    print(f"[INFO] Scanned: {args.images}")
    print(f"[INFO] Empty-annotation samples found: {n}")
    for p in empty_imgs[:20]:
        print(f"   - {p.name}")
    if n > 20:
        print(f"   ... ({n - 20} more)")

    if args.apply:
        for img, lbl in zip(empty_imgs, empty_lbls):
            if img.exists():
                img.unlink()
            if lbl.exists():
                lbl.unlink()
        print(f"[OK] Removed {n} empty-annotation samples.")
    else:
        print("[DRY-RUN] No files removed. Re-run with --apply to delete.")


if __name__ == "__main__":
    main()