# UAV-Tiny-Object-Detection
# https://github.com/Seven-zxy/UAV-Tiny-Object-Detection
#
# Reproducible random split for YOLO-format datasets.
#
# Re-splits a flat directory of (image, label) pairs into train/val/test
# subdirectories with a fixed random seed. Used after empty-annotation
# cleaning to obtain the canonical 523 / 149 / 76 split used in all
# reported experiments.
#
# Author : Xiaoyi Zheng (Zhejiang Chinese Medical University)
# License: Apache-2.0

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple

PairList = List[Tuple[Path, Path]]


def collect_pairs(images_dir: Path, labels_dir: Path) -> PairList:
    """Collect valid (image, label) pairs from a directory."""
    pairs: PairList = []
    for img in images_dir.iterdir():
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl = labels_dir / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def split_pairs(
    pairs: PairList,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[PairList, PairList, PairList]:
    """Random split with fixed seed for reproducibility."""
    rng = random.Random(seed)
    pairs = pairs.copy()
    rng.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (
        pairs[:n_train],
        pairs[n_train:n_train + n_val],
        pairs[n_train + n_val:],
    )


def copy_split(pairs: PairList, out_root: Path, split: str) -> None:
    """Copy a split's files into out_root/{split}/{images,labels}/."""
    img_out = out_root / split / "images"
    lbl_out = out_root / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for img, lbl in pairs:
        shutil.copy2(img, img_out / img.name)
        shutil.copy2(lbl, lbl_out / lbl.name)


def main():
    parser = argparse.ArgumentParser(
        description="Re-split a YOLO dataset into train/val/test."
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="Output root directory.")
    parser.add_argument("--train", type=float, default=0.7,
                        help="Train ratio (default: 0.7).")
    parser.add_argument("--val", type=float, default=0.2,
                        help="Val ratio (default: 0.2). Test = 1 - train - val.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = collect_pairs(args.images, args.labels)
    print(f"[INFO] Found {len(pairs)} valid (image, label) pairs.")

    train, val, test = split_pairs(pairs, args.train, args.val, seed=args.seed)
    print(f"[INFO] Split: train={len(train)}, val={len(val)}, test={len(test)}")

    for name, split in [("train", train), ("val", val), ("test", test)]:
        copy_split(split, args.out, name)
    print(f"[OK] Splits written to: {args.out}")


if __name__ == "__main__":
    main()
