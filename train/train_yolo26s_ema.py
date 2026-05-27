# UAV-Tiny-Object-Detection
# https://github.com/Seven-zxy/UAV-Tiny-Object-Detection
#
# Training entry point for YOLO26s with EMA (Efficient Multi-scale
# Attention) and a two-stage learning rate schedule.
#
# Reference setup (Phase-1 best YOLO26s variant):
#   - Backbone     : YOLO26s
#   - Modification : EMA attention inserted before the neck FPN
#   - Stage 1 (warm-start)  : 100 ep, lr0 = 0.01,   AMP on
#   - Stage 2 (fine-tuning) : 150 ep, lr0 = 5e-4,   AMP off
#   - Result       : mAP@0.5 = 0.581 -> 0.623 (+4.2 pp), Recall +12.5 pp
#
# Author : Xiaoyi Zheng (Zhejiang Chinese Medical University)
# License: Apache-2.0

import argparse

from ultralytics import YOLO


def train_yolo26s_ema(
    model_cfg: str,
    data_yaml: str,
    epochs: int = 150,
    imgsz: int = 640,
    batch: int = 16,
    lr0: float = 0.0005,
    seed: int = 0,
    project: str = "runs/yolo26s_ema",
    name: str = "exp",
    amp: bool = False,
    pretrained: str = None,
) -> None:
    """Train YOLO26s with EMA attention and fine-tuned learning rate.

    Args:
        model_cfg: Path to model architecture YAML (e.g. yolo26s-ema.yaml).
        data_yaml: Path to dataset YAML (Ultralytics format).
        epochs: Number of training epochs.
        imgsz: Training image size.
        batch: Per-device batch size.
        lr0: Initial learning rate. Default 5e-4 (fine-tuning regime).
        seed: Random seed for reproducibility.
        project: Save directory root.
        name: Experiment name.
        amp: Whether to use automatic mixed precision. Disabled by default
             since EMA attention exhibits NaN issues under fp16 AMP at
             small batch sizes.
        pretrained: Optional path to a warm-start weight (.pt). If provided,
                    weights are loaded before training.
    """
    model = YOLO(model_cfg)
    if pretrained:
        model.load(pretrained)
        print(f"[INFO] Loaded warm-start weights from: {pretrained}")

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        amp=amp,
        seed=seed,
        project=project,
        name=name,
        optimizer="AdamW",
        cos_lr=True,
        warmup_epochs=3,
        # Standard augmentation; aggressive Mosaic/Mixup hurts tiny objects.
        mosaic=0.5,
        mixup=0.0,
        close_mosaic=10,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO26s + EMA on a YOLO-format detection dataset."
    )
    parser.add_argument("--model", type=str, default="yolo26s.yaml",
                        help="Model architecture YAML.")
    parser.add_argument("--data", type=str, required=True,
                        help="Dataset YAML (Ultralytics format).")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr0", type=float, default=0.0005,
                        help="Initial learning rate. Default 5e-4.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=str, default="runs/yolo26s_ema")
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument("--amp", action="store_true",
                        help="Enable AMP (fp16). Off by default due to EMA NaN.")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Optional warm-start weights (.pt).")
    args = parser.parse_args()

    train_yolo26s_ema(
        model_cfg=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        seed=args.seed,
        project=args.project,
        name=args.name,
        amp=args.amp,
        pretrained=args.pretrained,
    )


if __name__ == "__main__":
    main()