# UAV-Tiny-Object-Detection
# https://github.com/Seven-zxy/UAV-Tiny-Object-Detection
#
# [DEPRECATED — NEGATIVE RESULT]
# Focaler-MPDIoU bounding-box loss.
#
# This loss was tried as a replacement for the default GIoU loss on the
# RT-DETR-L detector. It combines:
#   - MPDIoU (Minimum-Point-Distance IoU, Ma & Xu 2023): an IoU variant
#     that adds the squared corner-point distances between the predicted
#     and ground-truth boxes, normalized by the image diagonal.
#   - Focaler-IoU (Zhang & Zhang 2024): a piecewise-linear remapping of
#     the IoU value to focus the loss on a chosen difficulty interval.
#
# OUTCOME ON THIS DATASET (523 train / single class, tiny objects):
#   - Single seed:    val mAP@0.5 = 0.735  (looked promising)
#   - 3-seed average: val mAP@0.5 = 0.694  (BELOW the GIoU baseline)
#   - Variance across seeds was high; one run collapsed.
#
# Conclusion: the apparent single-run gain did not survive multi-seed
# evaluation. We reverted to the default GIoU loss. See
# results/negative_results.md for the full analysis.
#
# This file is kept for reproducibility / transparency, NOT for use.
#
# Author : Xiaoyi Zheng (Zhejiang Chinese Medical University)
# License: Apache-2.0

import torch


def bbox_mpdiou(
    pred: torch.Tensor,
    target: torch.Tensor,
    img_diag_sq: float,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute MPDIoU between predicted and target boxes.

    Boxes are in (x1, y1, x2, y2) format.

    MPDIoU = IoU - d1^2 / c^2 - d2^2 / c^2
    where d1, d2 are the top-left and bottom-right corner distances and
    c^2 is the squared image diagonal (normalization constant).

    Args:
        pred:   Predicted boxes, shape (N, 4), format (x1, y1, x2, y2).
        target: Target boxes,    shape (N, 4), format (x1, y1, x2, y2).
        img_diag_sq: Squared image diagonal (w^2 + h^2) for normalization.
        eps: Numerical stability constant.

    Returns:
        MPDIoU values, shape (N,). Range can be < 0 (unlike standard IoU).
    """
    # Intersection
    inter_x1 = torch.max(pred[:, 0], target[:, 0])
    inter_y1 = torch.max(pred[:, 1], target[:, 1])
    inter_x2 = torch.min(pred[:, 2], target[:, 2])
    inter_y2 = torch.min(pred[:, 3], target[:, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # Union
    area_pred = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    area_tgt = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = area_pred + area_tgt - inter + eps
    iou = inter / union

    # Corner-point distances (the MPDIoU contribution)
    d1_sq = (pred[:, 0] - target[:, 0]) ** 2 + (pred[:, 1] - target[:, 1]) ** 2
    d2_sq = (pred[:, 2] - target[:, 2]) ** 2 + (pred[:, 3] - target[:, 3]) ** 2

    mpdiou = iou - d1_sq / (img_diag_sq + eps) - d2_sq / (img_diag_sq + eps)
    return mpdiou


def focaler_remap(
    iou: torch.Tensor,
    d: float = 0.0,
    u: float = 0.95,
) -> torch.Tensor:
    """Apply the Focaler piecewise-linear remapping to an IoU-like value.

    focaler(iou) = clamp((iou - d) / (u - d), 0, 1)

    The interval [d, u] controls which difficulty range the loss focuses on.
    Here we used a wide interval; narrowing it did not help on tiny objects.

    Args:
        iou: IoU-like values (here, MPDIoU), shape (N,).
        d: Lower bound of the focus interval.
        u: Upper bound of the focus interval.

    Returns:
        Remapped values, shape (N,), clamped to [0, 1].
    """
    return ((iou - d) / (u - d + 1e-7)).clamp(0.0, 1.0)


def focaler_mpdiou_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    img_diag_sq: float,
    d: float = 0.0,
    u: float = 0.95,
) -> torch.Tensor:
    """Focaler-MPDIoU loss = 1 - focaler(MPDIoU).

    WARNING: deprecated. Underperformed GIoU across seeds on this dataset.

    Args:
        pred:   Predicted boxes (N, 4) in (x1, y1, x2, y2).
        target: Target boxes    (N, 4) in (x1, y1, x2, y2).
        img_diag_sq: Squared image diagonal for MPDIoU normalization.
        d, u: Focaler focus interval bounds.

    Returns:
        Per-box loss values, shape (N,).
    """
    mpdiou = bbox_mpdiou(pred, target, img_diag_sq)
    # Note: MPDIoU can be negative; Focaler clamps it into [0, 1].
    focaled = focaler_remap(mpdiou, d=d, u=u)
    return 1.0 - focaled


# ---------------------------------------------------------------------------
# Integration note (how this was wired into Ultralytics, for the record):
#
# In ultralytics/utils/loss.py, the RT-DETR bbox loss term was replaced with
# a call to focaler_mpdiou_loss(). The switch was guarded by an environment
# variable so the same loss.py could fall back to GIoU:
#
#     if os.environ.get("USE_FOCALER_MPDIOU") == "1":
#         loss_bbox = focaler_mpdiou_loss(pred_xyxy, tgt_xyxy, img_diag_sq)
#     else:
#         loss_bbox = giou_loss(pred_xyxy, tgt_xyxy)   # default
#
# After the negative multi-seed result, USE_FOCALER_MPDIOU was left unset
# (i.e., GIoU is the production default).
# ---------------------------------------------------------------------------
