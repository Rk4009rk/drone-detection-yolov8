"""
Full tracking pipeline: runs YOLOv8 detection on video, applies confidence +
temporal gating, feeds cleaned detections into the Kalman filter, and
computes the jitter-reduction metric.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from kalman_tracker import KalmanTracker2D

CONF_THRESHOLD = 0.35
MAX_JUMP_PX = 150  # max plausible pixel movement between consecutive frames


def extract_detections(model, video_path):
    """Run YOLOv8 on each frame, return list of (x, y) centers or None."""
    raw_centers = []
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        boxes = results[0].boxes

        if len(boxes) > 0:
            best = boxes[boxes.conf.argmax()]
            conf = float(best.conf[0])
            if conf >= CONF_THRESHOLD:
                x1, y1, x2, y2 = best.xyxy[0].tolist()
                raw_centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
            else:
                raw_centers.append(None)
        else:
            raw_centers.append(None)

    cap.release()
    return raw_centers


def temporal_gate(raw_centers, max_jump=MAX_JUMP_PX):
    """Reject detections that imply an implausible frame-to-frame jump."""
    cleaned = []
    last_valid = None
    for det in raw_centers:
        if det is None:
            cleaned.append(None)
            continue
        if last_valid is None:
            cleaned.append(det)
            last_valid = det
            continue
        dist = ((det[0] - last_valid[0]) ** 2 + (det[1] - last_valid[1]) ** 2) ** 0.5
        if dist > max_jump:
            cleaned.append(None)
        else:
            cleaned.append(det)
            last_valid = det
    return cleaned


def run_kalman_filter(cleaned_centers):
    first_valid = next(c for c in cleaned_centers if c is not None)
    tracker = KalmanTracker2D(*first_valid)
    filtered_path = []
    for det in cleaned_centers:
        if det is None:
            fx, fy = tracker.predict()
        else:
            tracker.predict()
            fx, fy = tracker.update(*det)
        filtered_path.append((fx, fy))
    return filtered_path


def compute_jitter(path):
    """Mean frame-to-frame jerk — lower means smoother trajectory."""
    path = np.array(path)
    velocity = np.diff(path, axis=0)
    jerk = np.diff(velocity, axis=0)
    return np.mean(np.linalg.norm(jerk, axis=1))


if __name__ == "__main__":
    model = YOLO("path/to/best.pt")
    video_path = "path/to/video.mp4"

    raw_centers = extract_detections(model, video_path)
    cleaned_centers = temporal_gate(raw_centers)
    filtered_path = run_kalman_filter(cleaned_centers)

    raw_valid = [c for c in cleaned_centers if c is not None]
    raw_jitter = compute_jitter(raw_valid)
    filtered_jitter = compute_jitter(filtered_path)

    print(f"Raw detection jitter: {raw_jitter:.3f}")
    print(f"Kalman-filtered jitter: {filtered_jitter:.3f}")
    print(f"Reduction: {(1 - filtered_jitter / raw_jitter) * 100:.1f}%")
