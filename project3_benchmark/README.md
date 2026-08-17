# Model Benchmarking: YOLOv8n vs YOLOv8s

A controlled comparison of two YOLOv8 model sizes on the same drone detection task, evaluating the accuracy-speed-size tradeoff to inform a deployment recommendation.

## Motivation

[Project 1](../project1_detection/) used YOLOv8n (nano) as the base model. This project asks: is nano the right choice, or would a larger model justify its cost in speed and size? Rather than assume, both models were trained and evaluated identically for a data-backed answer.

## Method

- Identical dataset, train/valid/test split, and hyperparameters (50 epochs, image size 640, batch size 16) for both models
- Evaluated on the same held-out test set (127 images, never seen during training)
- Trained on Kaggle (Tesla T4 GPU)

## Results

| Metric | YOLOv8n | YOLOv8s |
|---|---|---|
| Precision | **97.7%** | 90.3% |
| Recall | 82.5% | **84.4%** |
| mAP@50 | 87.8% | **88.8%** |
| mAP@50-95 | **70.3%** | 69.7% |
| Inference speed | **3.9 ms/image** | 6.5 ms/image |
| Model size | **6.2 MB** | 22.6 MB |

### Per-class mAP@50

| Class | YOLOv8n | YOLOv8s |
|---|---|---|
| Competition | 82.6% | **88.4%** |
| Fixed-wing | **97.9%** | 94.3% |
| Multicopter | 90.5% | **93.2%** |
| VTOL | **80.1%** | 79.2% |

## Conclusion

YOLOv8s achieves marginally higher recall and mAP@50 (within 1-2%), while YOLOv8n delivers meaningfully faster inference (**1.7x faster**) and a **3.6x smaller** model footprint, with higher precision.

**Recommendation:** For real-time or resource-constrained deployment (edge devices, embedded systems, live surveillance), **YOLOv8n is the stronger choice** — the accuracy difference is marginal, but the speed and size advantage is substantial. For applications where missing a detection is more costly than inference latency (e.g., offline/forensic analysis), YOLOv8s's higher recall may be worth the tradeoff.

## Tech Stack

Python, PyTorch (Ultralytics YOLOv8), trained on Kaggle GPU notebooks

## Limitations

- Both models trained for 50 epochs; neither was exhaustively hyperparameter-tuned, so this compares "reasonable defaults" rather than each model's absolute best possible performance
- Benchmark reflects this specific dataset and task; results may not generalize to other detection problems
- Only two model sizes compared (nano, small); larger variants (medium, large, extra-large) were not evaluated due to compute/time constraints

## Setup

```bash
pip install ultralytics
```

Both models were trained using the same pipeline as [Project 1](../project1_detection/), swapping `yolov8n.pt` for `yolov8s.pt` as the base checkpoint.
