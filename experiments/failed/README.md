# Failed Experiments

This directory contains code for approaches that were tried and **did not
work** on this dataset. They are kept for transparency and reproducibility —
documenting what *doesn't* transfer to small-scale tiny-object detection is
as valuable as documenting what does.

For the full quantitative analysis of each failure, see
[`../../results/negative_results.md`](../../results/negative_results.md).

## Contents

| File | Approach | Outcome |
| --- | --- | --- |
| `focaler_mpdiou_loss.py` | Focaler-MPDIoU bbox loss | single seed 0.735, but **3-seed avg 0.694 < GIoU baseline** |
| `pure_mpdiou_loss.py` | Pure MPDIoU bbox loss | **unstable, 2/3 seeds collapsed** (unbounded loss) |

## Approaches Tried but Not Included as Code

These were configuration-level or architecture-level changes (no standalone
module to preserve), documented in `negative_results.md`:

- **P2 shallow detection branch** — collapsed on 523 images (over-parameterization)
- **NWD loss (alpha 0.5 / 0.3)** — marginal or negative, hurt high-IoU localization
- **RT-DETR-X / D-FINE-L** — larger models, worse results (over-parameterization)
- **CBAM attention** — NaN under fp16 AMP; abandoned in favor of ECA
- **EMA attention after RT-DETR CCFF output** — catastrophic −29% mAP collapse

## Key Lesson

The recurring theme: **on a small dataset, added capacity and aggressive
loss reshaping consistently hurt.** This directly motivated the final design
choice of ECA — a channel-attention module adding only **17 parameters** —
placed *inside the backbone* rather than after feature fusion.
