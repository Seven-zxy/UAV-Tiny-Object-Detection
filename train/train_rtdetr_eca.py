# UAV-Tiny-Object-Detection
# https://github.com/Seven-zxy/UAV-Tiny-Object-Detection
#
# Training entry point for RT-DETR-L with ECA-HGNetv2 backbone enhancement.
#
# Reference setup (Phase-5, best overall configuration):
#   - Backbone     : HGNetv2 with ECA modules at P3 / P4 / P5 outputs
#   - Detector     : RT-DETR-L (decoder unchanged)
#   - Input size   : 960 (with weak augmentation)
#   - Result       : test mAP@0.5 = 0.728 -> 0.782 (+5.4 pp)
#                  : test best F1 = 0.71  -> 0.78  (+7   pp)
#   - Extra params : +17 learnable parameters (3 x ECA modules,
#                    k_size = 5 / 5 / 7 from adaptive formula)
#
# Author : Xiaoyi Zheng (Zhejiang Chinese Medical University)
# License: Apache-2.0

import argparse

# Custom modules MUST be registered before importing RTDETR / building model.
from train.register_custom_modules import register_eca_modules
register_eca_modules()

from ultralytics import RTDETR  # noqa: E402  (intentional late import)


def train_rtdetr_eca(
    model_cfg: str,
    data_yaml: str,
    epochs: int = 200,
    imgsz: int = 960,
    batch: int = 4,
    lr0: float = 0.0001,
    seed: int = 0,
    project: str = "runs/rtdetr_eca",
    name: str = "exp",
    weak_aug: bool = True,
) -> None:
    """Train RT-DETR-L + ECA-HGNetv2 on a YOLO-format detection dataset.

    Args:
        model_cfg: Path to model YAML, e.g. configs/rtdetr-l-eca.yaml.
        data_yaml: Path to dataset YAML (Ultralytics format).
        epochs: Number of training epochs.
        imgsz: Input image size. 960 was used for the best reported result.
        batch: Per-device batch size. Reduce if OOM at imgsz=960.
        lr0: Initial learning rate. RT-DETR prefers smaller lr than YOLO.
        seed: Random seed.
        project: Save directory root.
        name: Experiment name.
        weak_aug: If True, disable Mosaic / Mixup / color jitter
                  (recommended for tiny objects on ~500-image datasets).
    """
    model = RTDETR(model_cfg)

    aug_kwargs = (
        dict(mosaic=0.0, mixup=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
             degrees=0.0, translate=0.0, scale=0.0, fliplr=0.5)
        if weak_aug else
        dict()
    )

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        seed=seed,
        project=project,
        name=name,
        optimizer="AdamW",
        cos_lr=True,
        amp=True,
        **aug_kwargs,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train RT-DETR-L + ECA-HGNetv2."
    )
    parser.add_argument("--model", type=str,
                        default="configs/rtdetr-l-eca.yaml",
                        help="Model YAML. Default: configs/rtdetr-l-eca.yaml.")
    parser.add_argument("--data", type=str, required=True,
                        help="Dataset YAML (Ultralytics format).")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=960,
                        help="Input image size. 960 for best results.")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr0", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=str, default="runs/rtdetr_eca")
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument("--strong-aug", dest="weak_aug", action="store_false",
                        help="Use default Ultralytics augmentation. "
                             "Default is weak-aug for tiny objects.")
    parser.set_defaults(weak_aug=True)
    args = parser.parse_args()

    train_rtdetr_eca(
        model_cfg=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        seed=args.seed,
        project=args.project,
        name=args.name,
        weak_aug=args.weak_aug,
    )


if __name__ == "__main__":
    main()