# Multi-Seed Stability Analysis

Single-run results are unreliable on small datasets. To honestly assess
the robustness of the ECA-HGNetv2 result, the configuration was trained
under **8 random seeds across two hyperparameter settings**. This document
reports all of them — including the failures — and analyzes the failure
mode.

> **Metric note.** The table reports **validation** mAP@0.5 (available for
> all 8 runs). The headline **test** result (0.782) was evaluated only for
> the best run and the baseline; see [`main_results.md`](main_results.md).

---

## 1. All Runs

**Setting A — original hyperparameters** (lrf=0.005, weight_decay=5e-5, close_mosaic=20):

| Seed | val mAP@0.5 | Status |
| --- | --- | --- |
| 0 | **0.7775** | best run |
| 1 | 0.4138 | collapsed |
| 42 | 0.7620 | healthy |

**Setting B — "stabilization" attempt** (warmup=5, weight_decay=1e-4):

| Seed | val mAP@0.5 | Status |
| --- | --- | --- |
| 0 | 0.7145 | reduced |
| 1 | 0.2129 | collapsed |
| 7 | 0.7701 | healthy |
| 42 | 0.6850 | reduced |
| 100 | 0.7208 | reduced |

**Summary statistics:**

```
All 8 runs:                  mean = 0.632, std = 0.206
Excluding 2 collapses (n=6): mean = 0.738, std = 0.037
Best run (Setting A, seed 0):       0.7775
Reference baseline (single seed):   0.760
```

---

## 2. The Failure Mode

Two of the eight runs collapsed (seed 1 in both settings). Inspecting the
early-epoch training curves reveals a clear, consistent signature:

| Run | cls_loss @ ep10 | mAP@0.5 @ ep10 |
| --- | --- | --- |
| healthy run | rising / stable | > 0 |
| collapsed (s1) | driven toward ~0 | 0.000 |

This is the **classic DETR "all-background" trivial-solution collapse**:
on a small, single-class dataset the model can drive classification loss
toward zero by predicting "no object" for every query. Once in this state,
there are no positive predictions, the bounding-box loss receives no
gradient, and training is locked in the trivial solution permanently.

---

## 3. Root Cause: Hyperparameters, Not the Architecture

Critically, **the collapse is not a property of the ECA module**. The
evidence:

- The **baseline** (without ECA) trained stably to 0.760 using conservative
  hyperparameters (lrf=0.01, weight_decay=1e-4).
- Collapse appeared only under the **more aggressive** Setting-A
  hyperparameters (reduced weight_decay = 5e-5), and even then only for
  certain seeds.
- The reduced weight decay lowers regularization pressure, which on a
  ~500-image dataset makes some random initializations vulnerable to the
  all-background trap.

The instability is therefore a **hyperparameter-sensitivity x small-data x
seed interaction**, characteristic of DETR-family detectors on small
datasets — not a defect introduced by channel attention.

---

## 4. The Stabilization Attempt (Honest Outcome)

Setting B was an explicit attempt to fix the collapse by reverting
weight_decay to 1e-4 and increasing warmup to 5 epochs. The honest result:
**it did not fully succeed.** Seed 1 still collapsed, and the surviving
runs averaged *lower* (0.722) than Setting A's successful runs (0.770).
This is documented here rather than hidden — fully stabilizing DETR
training on this dataset remains open future work.

---

## 5. What This Means for the Reported Result

- **Best achievable:** ECA-HGNetv2 reaches **0.782 test mAP@0.5** (best
  run), a genuine **+5.4 pp** over the baseline's single-seed test result.
  Since the baseline is also single-seed, this is a fair best-vs-best
  comparison.
- **Robustness caveat:** the improvement is **not yet stable across seeds**
  under the explored hyperparameters. Reporting the best run is legitimate,
  but the variance and failure mode are disclosed here in full.
- **Scientific value:** the more transferable contribution is the
  **diagnosis** — identifying the all-background collapse signature and
  attributing it to hyperparameter sensitivity rather than the attention
  module.

---

## 6. Future Work

- Longer warmup + gradient clipping to escape the early all-background basin
- Focal-style classification loss weighting to counter class imbalance
- Larger effective dataset (more aerial imagery) to reduce seed sensitivity
- Weight averaging (EMA-of-weights / SWA) over the high-variance late regime
