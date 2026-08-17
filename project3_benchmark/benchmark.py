"""
Model Benchmark: YOLOv8n vs YOLOv8s
Trains both model sizes on the same dataset/split and compares
precision, recall, mAP, inference speed, and model size.
"""

from ultralytics import YOLO
import os

DATA_YAML = "Anti-drone-2-1/data.yaml"
EPOCHS = 50
IMG_SIZE = 640
BATCH = 16


def train_and_evaluate(model_name, run_name):
    """Train a YOLOv8 model and evaluate it on the held-out test set."""
    model = YOLO(f"{model_name}.pt")

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        name=run_name,
    )

    metrics = model.val(data=DATA_YAML, split="test")

    weights_path = f"runs/detect/{run_name}/weights/best.pt"
    model_size_mb = os.path.getsize(weights_path) / (1024 * 1024) if os.path.exists(weights_path) else None

    return {
        "model": model_name,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
        "mAP50": metrics.box.map50,
        "mAP50_95": metrics.box.map,
        "size_mb": model_size_mb,
    }


def print_comparison(results_a, results_b):
    print(f"{'Metric':<15}{results_a['model']:<15}{results_b['model']:<15}")
    print(f"{'Precision':<15}{results_a['precision']:.3f}          {results_b['precision']:.3f}")
    print(f"{'Recall':<15}{results_a['recall']:.3f}          {results_b['recall']:.3f}")
    print(f"{'mAP50':<15}{results_a['mAP50']:.3f}          {results_b['mAP50']:.3f}")
    print(f"{'mAP50-95':<15}{results_a['mAP50_95']:.3f}          {results_b['mAP50_95']:.3f}")
    if results_a["size_mb"] and results_b["size_mb"]:
        print(f"{'Size (MB)':<15}{results_a['size_mb']:.1f}           {results_b['size_mb']:.1f}")


if __name__ == "__main__":
    nano_results = train_and_evaluate("yolov8n", "drone_detector_nano")
    small_results = train_and_evaluate("yolov8s", "drone_detector_small")

    print_comparison(nano_results, small_results)
