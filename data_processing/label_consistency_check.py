# UAV-Tiny-Object-Detection
# https://github.com/Seven-zxy/UAV-Tiny-Object-Detection
#
# Label consistency checker for YOLO-format annotations.
#
# Verifies that:
#   - Every label line follows the format: <cls_id> <x_c> <y_c> <w> <h>
#   - cls_id is a non-negative integer within [0, nc)
#   - Bounding box coordinates are normalized to [0, 1]
#   - No duplicated boxes exist within a single image
#
# Author : Xiaoyi Zheng (Zhejiang Chinese Medical University)
# License: Apache-2.0

import argparse
from pathlib import Path
from typing import List


def check_label_file(lbl_path: Path, nc: int) -> List[str]:
    """Return a list of error messages for a single label file."""
    errors = []
    try:
        lines = lbl_path.read_text().splitlines()
    except Exception as e:
        return [f"unreadable file: {e}"]

    seen = set()
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"L{i}: expected 5 values, got {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(f"L{i}: non-numeric value")
            continue
        if cls < 0 or cls >= nc:
            errors.append(f"L{i}: class id {cls} out of [0, {nc})")
        for name, v in zip("xc yc w h".split(), (xc, yc, w, h)):
            if not (0.0 <= v <= 1.0):
                errors.append(f"L{i}: {name}={v:.4f} out of [0,1]")
        key = (cls, round(xc, 4), round(yc, 4), round(w, 4), round(h, 4))
        if key in seen:
            errors.append(f"L{i}: duplicated box")
        seen.add(key)
    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Check YOLO label files for format / range consistency."
    )
    parser.add_argument("--labels", type=Path, required=True,
                        help="Directory containing .txt labels.")
    parser.add_argument("--nc", type=int, default=1,
                        help="Number of classes (default: 1).")
    args = parser.parse_args()

    label_files = sorted(args.labels.glob("*.txt"))
    print(f"[INFO] Checking {len(label_files)} label files (nc={args.nc})...")

    bad = 0
    for lbl in label_files:
        errs = check_label_file(lbl, args.nc)
        if errs:
            bad += 1
            print(f"\n[BAD] {lbl.name}")
            for e in errs:
                print(f"    {e}")

    print(f"\n[DONE] {len(label_files) - bad}/{len(label_files)} files OK.")
    if bad:
        print(f"[WARN] {bad} files contain issues. Fix before training.")
    else:
        print("[OK] All labels passed consistency check.")


if __name__ == "__main__":
    main()
