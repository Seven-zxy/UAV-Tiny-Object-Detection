# Full Ablation Study

This document records the controlled experiments conducted across three
detector families. All mAP values are **validation-set mAP@0.5** unless
otherwise noted. The dataset is a UAV aerial tiny-object detection task
(523 train / 149 val / 76 test, single class, avg. object size ~10x14 px).

---

## A. YOLO26s Family (Phase 1)

Goal: establish a strong lightweight baseline and probe attention / head
modifications.

| # | Configuration | mAP@0.5 | Note |
| --- | --- | --- | --- |
| 1 | YOLO26s baseline, 100 ep | 0.535 | initial |
| 2 | YOLO26s baseline, 150 ep | **0.581** | adopted baseline |
| 3 | + strong aug + AdamW + cosine | 0.584 | ~no change |
| 4 | + two-stage warm-start | 0.610 | +2.9 pp |
| 5 | + P2 detection head (aggressive) | 0.336 | **collapse (−45%)** |
| 6 | + EMA only (lr=0.001) | failed | unstable |
| 7 | **+ EMA + tuned lr (5e-4)** | **0.623** | **best (+4.2 pp)** |

The winning YOLO26s recipe combines EMA attention before the neck with a
carefully reduced learning rate. Note that EMA *only worked* once the
learning rate was lowered to 5e-4; at the default 1e-3 it destabilized
training (row 6).

---

## B. RT-DETR-L Family (Phase 2 & 4)

Goal: push absolute accuracy with a stronger detector.

| # | Configuration | mAP@0.5 | Note |
| --- | --- | --- | --- |
| 1 | RT-DETR-L baseline (640) | 0.728 | baseline |
| 2 | + EMA after CCFF output | 0.516 | **collapse (−29%)** |
| 3 | + NWD loss (alpha=0.5) | 0.734 | marginal, hurts high-IoU |
| 4 | + NWD loss (alpha=0.3) | 0.708 | −2.0 pp |
| 5 | + P2 branch | ~0.0 | **training collapse** |
| 6 | input 960 + weak-aug | 0.761 | +3.3 pp (new baseline) |
| 7 | **+ ECA-HGNetv2 (960)** | **0.778** | **best (+1.7 pp on val)** |

Two interventions mattered most: (a) raising input resolution to 960,
which recovers spatial signal for tiny objects, and (b) the ECA-HGNetv2
backbone enhancement. Loss-function modifications (NWD) and the P2 branch
did not help on this dataset.

---

## C. D-FINE Family (Phase 3)

Goal: test whether a newer DETR-variant could surpass RT-DETR-L.

| # | Configuration | mAP@0.5 | Note |
| --- | --- | --- | --- |
| 1 | D-FINE-S (640) | 0.453 | underfit |
| 2 | D-FINE-M (640) | 0.556 | |
| 3 | D-FINE-M (800) | 0.620 | best D-FINE |
| 4 | D-FINE-L (640) | 0.588 | larger != better |

No D-FINE configuration matched RT-DETR-L. We concluded that on this small,
tiny-object dataset, RT-DETR-L's inductive biases were better suited, and
refocused effort on improving it.

---

## D. Input Resolution Sweep (RT-DETR-L)

| Input size | Augmentation | mAP@0.5 |
| --- | --- | --- |
| 640 | standard | 0.728 |
| 800 | weak | 0.169 (unstable run) |
| **960** | **weak** | **0.761** |

At 960 with weak augmentation, RT-DETR-L reached a stable 0.761 — adopted
as the Phase-4 baseline on top of which ECA-HGNetv2 was built.

---

## Summary

The progression of best-achieved validation mAP@0.5:

```
YOLO26s baseline          0.581
  + EMA (tuned lr)        0.623   (lightweight track)

RT-DETR-L baseline (640)  0.728
  + 960 / weak-aug        0.761
  + ECA-HGNetv2           0.778   (accuracy track, best)
```

See [`negative_results.md`](negative_results.md) for the failed approaches
in detail, and [`seed_stability.md`](seed_stability.md) for reproducibility.
