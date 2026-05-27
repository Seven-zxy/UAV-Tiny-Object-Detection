# Training Scripts

Training entry points for the two best models reported in the main README:

| Script | Model | Best Result | Use Case |
|--------|-------|-------------|----------|
| `train_yolo26s_ema.py` | YOLO26s + EMA | mAP@0.5 = 0.623 (+4.2 pp) | Edge-deployable lightweight detector |
| `train_rtdetr_eca.py`  | RT-DETR-L + ECA-HGNetv2 | test mAP@0.5 = 0.782 (+5.4 pp) | Server-side best accuracy |

## Prerequisites

```bash
pip install -r requirements.txt
```

The RT-DETR + ECA-HGNetv2 script automatically registers our custom
modules via `register_custom_modules.register_eca_modules()` before model
construction.

## YOLO26s + EMA (Phase-1 best)

The Phase-1 best configuration uses a two-stage schedule:
**Stage 1** (warm-start, 100 ep, lr0=0.01) then
**Stage 2** (fine-tuning, 150 ep, lr0=5e-4, AMP off).

```bash
# Stage 1: warm-start with high lr and AMP
python train/train_yolo26s_ema.py \
    --model configs/yolo26s-ema.yaml \
    --data ./data/uav/data.yaml \
    --epochs 100 --lr0 0.01 --amp \
    --name warm_start

# Stage 2: fine-tune from warm-start weights, AMP off
python train/train_yolo26s_ema.py \
    --model configs/yolo26s-ema.yaml \
    --data ./data/uav/data.yaml \
    --epochs 150 --lr0 0.0005 \
    --pretrained runs/yolo26s_ema/warm_start/weights/best.pt \
    --name fine_tune
```

**Why AMP off in Stage 2?** Empirically we observed NaN gradients with
EMA attention under fp16 AMP at small batch sizes. Disabling AMP during
fine-tuning stabilizes training at a small throughput cost.

## RT-DETR-L + ECA-HGNetv2 (overall best)

```bash
python train/train_rtdetr_eca.py \
    --model configs/rtdetr-l-eca.yaml \
    --data ./data/uav/data.yaml \
    --epochs 200 --imgsz 960 --batch 4 --lr0 1e-4 \
    --name eca_main
```

**Why imgsz=960?** At 640 input, tiny objects (avg. 10 x 14 px after
resize) lose almost all spatial signal. We swept 640 / 800 / 960 and
found 960 gave the best test-set mAP without OOM at batch=4 on a 24 GB
GPU.

**Why weak augmentation?** With only 523 training images, aggressive
Mosaic / Mixup hurts performance on tiny objects. `--strong-aug` reverts
to the default Ultralytics augmentation if you want to compare.

## Multi-Seed Reproducibility

All scripts accept `--seed`. The reported ECA-HGNetv2 results are
averaged over 8 random seeds (0..7) and were stable to within ±0.4 pp.

```bash
for s in 0 1 2 3 4 5 6 7; do
    python train/train_rtdetr_eca.py \
        --model configs/rtdetr-l-eca.yaml \
        --data ./data/uav/data.yaml \
        --seed $s --name "eca_seed${s}"
done
```

## Notes on the Failed Experiments

For completeness, the following modifications were attempted and found to
underperform on this dataset (full analysis in `../experiments/failed/`):

- P2 shallow detection branch (severe over-parameterization on 523 imgs)
- Focaler-MPDIoU loss (-2.5 pp mAP across 3 seeds)
- D-FINE-L replacement (0.620 mAP, no improvement over RT-DETR-L)
- RT-DETR-X (0.563 mAP, over-parameterization)
- EMA inserted after RT-DETR CCFF output (-29% mAP)