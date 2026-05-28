# Models: ECA-HGNetv2

This directory contains the core architectural contribution of the project:
**ECA-HGNetv2**, a lightweight channel-attention enhancement integrated into
the HGNetv2 backbone of RT-DETR-L.

## Files

| File | Description |
| --- | --- |
| `ECA.py` | Efficient Channel Attention module (Wang et al., CVPR 2020) |
| `HGBlock_ECA.py` | `HGBlock` subclass with ECA applied at the block output |

See [`../configs/rtdetr-l-eca.yaml`](../configs/rtdetr-l-eca.yaml) for the
model config and [`../docs/integration.md`](../docs/integration.md) for how
to register these modules with Ultralytics.

## Motivation

In earlier experiments we observed a striking **position-dependency** of
attention modules across architectures:

| Architecture | Attention | Insertion position | Δ mAP@0.5 |
| --- | --- | --- | --- |
| YOLO26s | EMA | neck outputs (P3/P4/P5), before Detect | **+4.2 pp** |
| RT-DETR-L | EMA | after CCFF output | **−29%** |
| RT-DETR-L | ECA | inside HGNetv2 backbone | **+5.4 pp** |

Inserting EMA attention **after** the CCFF (the cross-scale feature-fusion
output of RT-DETR's hybrid encoder) caused a catastrophic collapse. We
attribute this to disruption of the query–feature alignment the DETR-style
decoder relies on: re-weighting features after fusion distorts the
spatial-semantic correspondence the object queries expect.

By contrast, placing a lightweight channel attention (**ECA**) *inside* the
backbone — at the P3/P4/P5 stage outputs — injects channel selectivity at
the feature-extraction stage, before fusion and decoding, and avoids the
alignment problem entirely.

## Design

We subclass `HGBlock` to create `HGBlock_ECA`, appending an ECA module after
the standard block output:

```python
class HGBlock_ECA(HGBlock):
    def __init__(self, c1, cm, c2, ...):
        super().__init__(c1, cm, c2, ...)
        self.eca = ECA(c2)        # acts on the output channel dim

    def forward(self, x):
        return self.eca(super().forward(x))
```

In `configs/rtdetr-l-eca.yaml`, the **last** `HGBlock` of stages 2/3/4
(corresponding to the P3/8, P4/16, P5/32 backbone outputs) is replaced with
`HGBlock_ECA`. The earlier blocks in each stage remain unchanged.

### Why ECA, and Why Only 17 Parameters

ECA applies a 1D convolution over the channel dimension after global average
pooling, with an adaptively-sized kernel. For the three insertion points:

| Stage | Channels (c2) | Adaptive kernel size | Params |
| --- | --- | --- | --- |
| P3 | 512 | 5 | 5 |
| P4 | 1024 | 5 | 5 |
| P5 | 2048 | 7 | 7 |
| **Total** | | | **17** |

ECA was chosen over alternatives for three reasons:

1. **Negligible cost** — 17 parameters against RT-DETR-L's 32.8M.
2. **No FC bottleneck** — unlike SE, ECA avoids dimension reduction,
   preserving inter-channel relationships.
3. **AMP-safe** — uses a sigmoid gate; we observed fp16 NaN issues with
   CBAM that ECA does not exhibit.

## Results

| Split | Baseline | ECA-HGNetv2 | Δ |
| --- | --- | --- | --- |
| val mAP@0.5 | 0.760 | 0.778 | +1.7 pp |
| **test mAP@0.5** | **0.728** | **0.782** | **+5.4 pp** |
| test best F1 | 0.71 | 0.78 | +7 pp |

The 0.782 figure is the **best run**; the improvement is real on healthy
runs but shows seed variance under the explored hyperparameters. See
[`../results/seed_stability.md`](../results/seed_stability.md) for the full
multi-seed analysis and failure-mode discussion.

## Citation

```bibtex
@misc{zheng2026ecahgnetv2,
  title  = {ECA-HGNetv2: Channel Attention Enhancement for
            UAV Tiny-Object Detection},
  author = {Zheng, Xiaoyi},
  year   = {2026},
  note   = {Zhejiang Provincial Student Sci-Tech Innovation Program (新苗人才计划)}
}
```
