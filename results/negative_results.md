# Negative Results

A record of approaches that were tried and **did not** improve performance.
Documenting these is as important as the successful ones: they delimit what
does and does not transfer to small-scale tiny-object detection, and they
explain why the final design looks the way it does.

---

## 1. EMA Attention after RT-DETR CCFF — Catastrophic (−29%)

| | mAP@0.5 |
| --- | --- |
| RT-DETR-L baseline | 0.728 |
| + EMA after CCFF output | 0.516 |

The single most informative failure. The same EMA module that *helped*
YOLO26s (+4.2 pp) *destroyed* RT-DETR (−29%) when placed after the
cross-scale feature fusion. This motivated moving attention *inside* the
backbone instead (→ ECA-HGNetv2), and became the central finding of the
project (see [`main_results.md`](main_results.md)).

## 2. P2 Shallow Detection Branch — Training Collapse

| Detector | mAP@0.5 |
| --- | --- |
| YOLO26s + P2 | 0.336 |
| RT-DETR-L + P2 | ~0.0 |

Adding a high-resolution P2 detection head is a common tiny-object trick,
but on only 523 training images it severely over-parameterized the model
and collapsed training. The dataset is too small to support the extra
capacity.

## 3. Loss-Function Modifications — No Net Gain

| Method | mAP@0.5 | Note |
| --- | --- | --- |
| NWD (alpha=0.5) | 0.734 | +0.6 pp but worse at high IoU |
| NWD (alpha=0.3) | 0.708 | −2.0 pp |
| Focaler-MPDIoU (single seed) | 0.735 | looked promising... |
| Focaler-MPDIoU (3-seed mean) | 0.694 | ...but unstable across seeds |
| Pure MPDIoU | failed | did not converge well |

The Focaler-MPDIoU case is a good reminder to **always check seed
variance**: a single promising run (0.735) averaged down to 0.694 across
three seeds, below the baseline. Loss-function gains reported in the
literature did not transfer to this dataset's characteristics.

## 4. Larger Models — Over-parameterization

| Model | mAP@0.5 | Note |
| --- | --- | --- |
| RT-DETR-X | 0.563 | severely over-parameterized |
| D-FINE-L (640) | 0.588 | larger than D-FINE-M, but worse |

Scaling up the model consistently hurt on this small dataset. Capacity is
not the bottleneck; data is.

## 5. Other Attempts

| Method | Outcome |
| --- | --- |
| SET module | could not be reliably implemented (no usable reference code) |
| CBAM | NaN under fp16 AMP; abandoned in favor of ECA |

---

## Lessons That Shaped the Final Design

1. **Position over module.** Where attention goes matters more than which
   attention it is. This directly produced the ECA-in-backbone design.
2. **Small data punishes capacity.** P2 branches, RT-DETR-X, D-FINE-L all
   failed by adding parameters the 523-image set could not support. ECA's
   17-parameter footprint is a deliberate response.
3. **Always average over seeds.** Focaler-MPDIoU would have been a false
   positive on a single run.
4. **Literature tricks are dataset-conditional.** NWD and MPDIoU losses,
   well-regarded elsewhere, did not transfer here.
