# UAV-Tiny-Object-Detection
# https://github.com/Seven-zxy/UAV-Tiny-Object-Detection
#
# HGBlock_ECA: HGBlock (from PPHGNetV2 / Ultralytics) augmented with an
# ECA channel-attention module at its output. Used as a drop-in
# replacement for HGBlock at P3/P4/P5 stages of the HGNetv2 backbone in
# RT-DETR-L.
#
# Author : Xiaoyi Zheng (Zhejiang Chinese Medical University)
# License: Apache-2.0

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import HGBlock

from .ECA import ECA


class HGBlock_ECA(HGBlock):
    """HGBlock with an ECA channel-attention module at the output.

    A drop-in replacement for the original HGBlock. The ECA module is
    appended after the standard HGBlock forward, applying channel-wise
    attention to the output feature map before it is passed to the next
    backbone stage.

    Args:
        c1 (int): Input channels.
        cm (int): Middle channels.
        c2 (int): Output channels.
        k (int): Kernel size. Default: 3.
        n (int): Number of LightConv / Conv blocks. Default: 6.
        lightconv (bool): Whether to use LightConv. Default: False.
        shortcut (bool): Whether to use shortcut connection. Default: False.
        act (nn.Module): Activation function. Default: nn.ReLU().
    """

    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False,
        shortcut: bool = False,
        act: nn.Module = nn.ReLU(),
    ):
        super().__init__(c1, cm, c2, k, n, lightconv, shortcut, act)
        # ECA acts on the final output channel dimension (c2).
        self.eca = ECA(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: standard HGBlock output followed by ECA."""
        return self.eca(super().forward(x))
