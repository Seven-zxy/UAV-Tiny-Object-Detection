# UAV-Tiny-Object-Detection
# https://github.com/Seven-zxy/UAV-Tiny-Object-Detection
#
# Custom module registration for the Ultralytics task parser.
#
# Ultralytics resolves module names in YAML configs by looking them up in
# `ultralytics.nn.modules` and `ultralytics.nn.tasks`. To use our custom
# HGBlock_ECA (and the ECA module it depends on) inside an Ultralytics
# YAML, we expose them through those namespaces before model construction.
#
# Import and call `register_eca_modules()` BEFORE constructing any
# RTDETR / YOLO model whose YAML references HGBlock_ECA.
#
# Author : Xiaoyi Zheng (Zhejiang Chinese Medical University)
# License: Apache-2.0

import ultralytics.nn.modules as um
import ultralytics.nn.tasks as ut

from models.ECA import ECA
from models.HGBlock_ECA import HGBlock_ECA


def register_eca_modules() -> None:
    """Expose ECA and HGBlock_ECA in Ultralytics' module / task namespaces.

    Ultralytics' `parse_model` resolves module names against `globals()` in
    `ultralytics.nn.tasks`. Injecting our classes there allows YAML strings
    like `HGBlock_ECA` to resolve correctly.
    """
    if not hasattr(um, "ECA"):
        um.ECA = ECA
    if not hasattr(um, "HGBlock_ECA"):
        um.HGBlock_ECA = HGBlock_ECA

    ut.ECA = ECA
    ut.HGBlock_ECA = HGBlock_ECA