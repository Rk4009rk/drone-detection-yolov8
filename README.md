# Multi-Modal Drone Detection & Perception System

A computer vision pipeline for detecting, classifying, and tracking drones — built as a progressive series of components, each independently functional and measured, together forming the foundation of a larger perception system.

## Project Components

### 1. [Drone Detection & Classification](./project1_detection/)
YOLOv8-based detection and classification of drones into 4 categories (Fixed-wing, Multicopter, VTOL, Competition).
- **97.7% precision / 82.5% recall / 87.8% mAP@50** on held-out test set

### 2. [Kalman Filter Tracking](./project2_tracking/)
Custom 2D Kalman filter (built from scratch in NumPy) that smooths noisy detections and tracks the drone through missed frames.
- **69.1% reduction in trajectory jitter** on real drone footage

### 3. [Model Benchmarking](./project3_benchmark/)
Controlled comparison of YOLOv8n vs YOLOv8s for drone detection.
- **YOLOv8n: 1.7x faster inference, 3.6x smaller** with comparable accuracy (mAP@50 within 1%)

## Vision

This is Phase 1 of a longer-term multi-modal drone detection and localization system. Planned extensions include camera-based bearing estimation, acoustic/RF sensing, and sensor fusion — see individual project READMEs for component-specific roadmaps.

## Tech Stack

Python, PyTorch (via Ultralytics YOLOv8), OpenCV, NumPy, Google Colab (GPU training)

## Setup

Each project folder contains its own `requirements.txt` and notebook/scripts. See individual READMEs for details.
