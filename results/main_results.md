# Main Results

## Test-Set Performance (held-out, 76 images)

The headline numbers reported in the project README are measured on the
held-out **test set**, evaluated once after training completes.

| Model | test mAP@0.5 | test best F1 | Δ mAP@0.5 |
| --- | --- | --- | --- |
| RT-DETR-L baseline | 0.728 | 0.71 | — |
| **RT-DETR-L + ECA-HGNetv2 (ours)** | **0.782** | **0.78** | **+5.4 pp** |

A key observation: the **test-set** improvement (+5.4 pp) is *larger* than
the **validation-set** improvement (+1.7 pp, 0.761 → 0.778 against the
960/weak-aug baseline). Because the architecture choice was guided by
validation performance, a larger gain on the untouched test set is
positive evidence that the improvement reflects genuine generalization
rather than validation-set overfitting.

## The Core Finding: Attention Position Matters

The central scientific contribution of this work is not the absolute mAP
number, but the demonstration that **the same attention mechanism produces
opposite outcomes depending on where it is inserted**:

| Architecture | Attention | Insertion Position | Δ mAP@0.5 |
| --- | --- | --- | --- |
| YOLO26s | EMA | before neck FPN | **+4.2 pp** ✅ |
| RT-DETR-L | EMA | after CCFF output | **−29%** ❌ |
| RT-DETR-L | ECA | inside HGNetv2 backbone (P3/P4/P5) | **+5.4 pp** ✅ |

### Interpretation

Inserting EMA attention **after** the CCFF (the cross-scale feature-fusion
output of RT-DETR's hybrid encoder) caused a catastrophic −29% collapse.
We attribute this to disruption of the query-feature alignment that the
DETR-style decoder relies on: re-weighting features after fusion distorts
the spatial-semantic correspondence the decoder's object queries expect.

By contrast, placing a lightweight channel attention (**ECA**) *inside* the
backbone, at the P3/P4/P5 stage outputs, injects channel selectivity at
the feature-extraction stage — before fusion and decoding — and avoids the
alignment problem entirely. This is what yields the +5.4 pp test-set gain.

**Takeaway:** attention modules are **not** architecture-agnostic
plug-and-play components. Their effectiveness is strongly conditioned on
insertion position relative to the detector's structural assumptions.

## Why ECA, and Why Only 17 Parameters

ECA (Efficient Channel Attention) applies a 1D convolution over the channel
dimension after global average pooling, with an adaptively-sized kernel.
For the three insertion points:

| Stage | Channels (c2) | Adaptive kernel size | Params |
| --- | --- | --- | --- |
| P3 | 512 | 5 | 5 |
| P4 | 1024 | 5 | 5 |
| P5 | 2048 | 7 | 7 |
| **Total** | | | **17** |

The negligible parameter cost makes this improvement essentially "free" in
terms of model size, which is significant for potential edge deployment.

## Lightweight Alternative: YOLO26s + EMA

For deployment scenarios with tight compute budgets, the YOLO26s + EMA
configuration is also reported:

| Model | mAP@0.5 | Recall |
| --- | --- | --- |
| YOLO26s baseline (150 ep) | 0.581 | 0.565 |
| **YOLO26s + EMA (ours)** | **0.623** (+4.2 pp) | **0.635** (+12.5% rel.) |

This model trades roughly 6 mAP points against RT-DETR-L+ECA in exchange
for a substantially smaller and faster network.
