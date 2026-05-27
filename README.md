# UAV-Tiny-Object-Detection

> Research on tiny-object detection for UAV aerial imagery,
> featuring **ECA-HGNetv2** backbone enhancement.

## 📌 Project Status

🚧 **Repository under active organization** (May 2026). Code, configurations, 
and experiment logs are being progressively uploaded.

This work has been recognized as a project of the **Zhejiang Provincial 
Student Science and Technology Innovation Program (新苗人才计划) 2025** 
and completed final review.

## 🎯 Key Results (Preview)

| Model | Test mAP@0.5 | Best F1 | Notes |
|-------|--------------|---------|-------|
| RT-DETR-L baseline | 0.728 | 0.71 | — |
| **RT-DETR-L + ECA-HGNetv2 (ours)** | **0.782** | **0.78** | +17 params |
| YOLO26s baseline | 0.581 | — | edge-deployable |
| YOLO26s + EMA (ours) | 0.623 | — | +7.3% mAP |

Detailed results, ablation studies, and negative findings will be released 
in subsequent commits.

## 🔍 Highlights

- **+5.4 pp test mAP** improvement via ECA-HGNetv2 (only +17 learnable params)
- Discovered **position-dependency of attention modules** across architectures
- 20+ controlled experiments spanning YOLO26s / RT-DETR-L / D-FINE families

## 📂 Repository Structure

```
models/           # ECA module & HGBlock_ECA implementation
configs/          # Model & training configurations
train/            # Training scripts
data_processing/  # Dataset cleaning & format conversion
results/          # Experiment logs, figures, ablation tables
experiments/      # Including negative results
docs/             # Additional documentation
```

## 📜 Acknowledgements

- **Base Framework**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **RT-DETR**: [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) (CVPR 2024)
- **ECA-Net**: Wang et al., CVPR 2020
- This work was supported by the Zhejiang Provincial Student 
  Sci-Tech Innovation Program (新苗人才计划).

## 📄 License

This repository is released under the Apache-2.0 License. 
See [LICENSE](LICENSE) for details.

Note: Dataset itself is not included in this repository due to 
regulatory considerations regarding the specific plant species studied.

---

📧 Contact: 15825585898@163.com
🎓 Author: Xiaoyi Zheng (郑潇一), Zhejiang Chinese Medical University
