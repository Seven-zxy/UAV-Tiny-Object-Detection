# UAV-Tiny-Object-Detection
# https://github.com/Seven-zxy/UAV-Tiny-Object-Detection
#
# [DEPRECATED — NEGATIVE RESULT]
# Pure MPDIoU bounding-box loss (without the Focaler remapping).
#
# OUTCOME ON THIS DATASET:
#   - Training was unstable: 2 of 3 seeds collapsed.
#   - Best surviving seed reached only ~0.70 val mAP@0.5 (below baseline).
#
# Root cause (see results/negative_results.md): MPDIoU can become strongly
# negative when boxes are far apart, making the (1 - MPDIoU) loss unbounded
# above. On certain random initializations this triggered gradient blow-up
# early in training, from which the run never recovered.
#
# Kept for transparency, NOT for use.
#
# Author : Xiaoyi Zheng (Zhejiang Chinese Medical University)
# License: Apache-2.0

import torch

from .focaler_mpdiou_loss import bbox_mpdiou


def pure_mpdiou_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    img_diag_sq: float,
) -> torch.Tensor:
    """Pure MPDIoU loss = 1 - MPDIoU (no Focaler clamping).

    WARNING: deprecated and numerically unstable on small datasets.
    Because MPDIoU is unbounded below, (1 - MPDIoU) is unbounded above,
    which caused gradient explosions on some seeds. Documented here only
    to explain why the Focaler clamp (see focaler_mpdiou_loss.py) was
    introduced — and why even that did not ultimately beat GIoU.

    Args:
        pred:   Predicted boxes (N, 4) in (x1, y1, x2, y2).
        target: Target boxes    (N, 4) in (x1, y1, x2, y2).
        img_diag_sq: Squared image diagonal for normalization.

    Returns:
        Per-box loss values, shape (N,). NOTE: unbounded above.
    """
    mpdiou = bbox_mpdiou(pred, target, img_diag_sq)
    return 1.0 - mpdiou
