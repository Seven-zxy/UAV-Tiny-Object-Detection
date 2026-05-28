# Multi-Seed Stability

Single-run results can be misleading on small datasets (see the
Focaler-MPDIoU case in [`negative_results.md`](negative_results.md), where a
promising single run averaged down below baseline). To verify that the
ECA-HGNetv2 improvement is robust rather than a lucky seed, the final
configuration was re-trained under multiple random seeds.

## Confirmed Run (seed 0)

| Seed | val mAP@0.5 | val mAP@.5:.95 | Precision | Recall |
| --- | --- | --- | --- | --- |
| 0 | 0.778 | 0.382 | 0.895 | 0.747 |

## Full Multi-Seed Table

> **TODO (fill with your real numbers).** The table below lists the seeds
> that were run. Replace each row with the actual best validation mAP@0.5
> from `runs/.../seed_k/results.csv`. Do **not** report seeds you did not
> actually run.

| Seed | val mAP@0.5 | val mAP@.5:.95 |
| --- | --- | --- |
| 0 | 0.778 | 0.382 |
| 1 | _fill_ | _fill_ |
| 2 | _fill_ | _fill_ |
| 3 | _fill_ | _fill_ |
| ... | ... | ... |

**Summary statistics (to compute once filled):**

```
mean mAP@0.5 = _fill_
std  mAP@0.5 = _fill_
```

## How to Reproduce

```bash
for s in 0 1 2 3 4 5 6 7; do
    python train/train_rtdetr_eca.py \
        --model configs/rtdetr-l-eca.yaml \
        --data ./data/uav/data.yaml \
        --seed $s --name "eca_seed${s}"
done
```

Then collect the best mAP@0.5 from each run's `results.csv`:

```python
import pandas as pd, glob
for f in sorted(glob.glob("runs/rtdetr_eca/eca_seed*/results.csv")):
    df = pd.read_csv(f); df.columns = [c.strip() for c in df.columns]
    print(f, df["metrics/mAP50(B)"].max())
```
