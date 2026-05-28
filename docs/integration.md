# Integrating ECA-HGNetv2 into Ultralytics

This document explains how to register the custom `ECA` and `HGBlock_ECA`
modules so that an Ultralytics-format YAML config (e.g.
`configs/rtdetr-l-eca.yaml`) can reference `HGBlock_ECA` by name.

Ultralytics resolves module names in YAML configs by looking them up in the
namespace of `ultralytics.nn.tasks` during `parse_model()`. Custom modules
must be made visible there before the model is built.

## Option A — Programmatic registration (recommended, non-invasive)

This is what `train/register_custom_modules.py` does. It injects the classes
into the relevant namespaces at runtime, without editing the installed
Ultralytics package:

```python
import ultralytics.nn.modules as um
import ultralytics.nn.tasks as ut
from models.ECA import ECA
from models.HGBlock_ECA import HGBlock_ECA

um.ECA = ECA
um.HGBlock_ECA = HGBlock_ECA
ut.ECA = ECA
ut.HGBlock_ECA = HGBlock_ECA
```

Call `register_eca_modules()` **before** constructing the model. This keeps
your Ultralytics installation clean and makes the repo self-contained.

## Option B — Editing the Ultralytics source (invasive)

If you prefer to patch the package directly:

1. Copy `models/ECA.py` and `models/HGBlock_ECA.py` into
   `ultralytics/nn/modules/`.

2. In `ultralytics/nn/modules/__init__.py`, export them:
   ```python
   from .ECA import ECA
   from .HGBlock_ECA import HGBlock_ECA
   __all__ = (..., "ECA", "HGBlock_ECA")
   ```

3. In `ultralytics/nn/tasks.py`, import the new block near the other
   block imports:
   ```python
   from ultralytics.nn.modules import HGBlock_ECA
   ```

## The Critical `parse_model` Fix

Regardless of which option you choose, there is **one essential change** in
`parse_model()` (in `ultralytics/nn/tasks.py`).

`HGBlock` is special-cased so that the `n` argument (number of internal
blocks) is consumed correctly. `HGBlock_ECA` subclasses `HGBlock`, but the
`is` check does not match subclasses. Find the line:

```python
if m is HGBlock:
    # ... handles the `n` repeat argument
```

and change it to also catch the subclass:

```python
if m in {HGBlock, HGBlock_ECA}:
    # ... now handles HGBlock_ECA correctly
```

**Why this matters:** without this fix, `parse_model` does not pass `n`
into `HGBlock_ECA` and instead instantiates `n` (=6) *sequential* copies of
the module rather than one block with `n=6` internal layers. The model will
build, but with the wrong architecture — silently. This was the single
trickiest part of the integration.

## Verifying the Integration

After registration, confirm the model builds with the expected parameter
count:

```python
from train.register_custom_modules import register_eca_modules
register_eca_modules()

from ultralytics import RTDETR
model = RTDETR("configs/rtdetr-l-eca.yaml")
print(sum(p.numel() for p in model.parameters()))
# Expect ~32.8M + 17  (the 17 extra params are the 3 ECA modules:
# k_size = 5 / 5 / 7 at P3 / P4 / P5)
```

If the parameter count is off by thousands instead of by 17, the
`parse_model` fix above is likely missing.
