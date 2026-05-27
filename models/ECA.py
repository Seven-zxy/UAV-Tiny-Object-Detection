# UAV-Tiny-Object-Detection
# https://github.com/Seven-zxy/UAV-Tiny-Object-Detection
#
# ECA (Efficient Channel Attention) module.
# Reference: Wang et al., "ECA-Net: Efficient Channel Attention for
#            Deep Convolutional Neural Networks", CVPR 2020.
#            arXiv:1910.03151
#
# Author : Xiaoyi Zheng (Zhejiang Chinese Medical University)
# License: Apache-2.0

import math
import torch
import torch.nn as nn


class ECA(nn.Module):
    """Efficient Channel Attention module.

    Performs cross-channel interaction through a 1D convolution over the
    channel dimension after global average pooling, with negligible
    parameter overhead (e.g. ~5-7 params per module on 512-2048 channels).

    Args:
        c1 (int): Number of input channels.
        k_size (int, optional): Kernel size of the 1D convolution.
            If None, it is adaptively computed from c1 following the
            formula in the original ECA-Net paper.

    Reference:
        Wang et al., CVPR 2020. arXiv:1910.03151
    """

    def __init__(self, c1: int, k_size: int = None):
        super().__init__()
        if k_size is None:
            # Adaptive kernel size determined by channel count.
            # Following ECA-Net paper Eq.(5): k = psi(C) = |log2(C)/2 + 1/2|_odd
            t = int(abs((math.log(c1, 2) + 1) / 2))
            k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size,
            padding=(k_size - 1) // 2, bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Tensor of the same shape as input, with channel-wise
            attention applied.
        """
        # x: (B, C, H, W)
        y = self.avg_pool(x)                       # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)        # (B, 1, C)
        y = self.conv(y)                           # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)      # (B, C, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
