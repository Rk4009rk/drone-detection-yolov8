"""
Kalman Filter Tracker for Drone Detections
Tracks the (x, y) center of a detected drone across frames, smoothing noisy
detections and predicting position even when a frame has no detection.
"""

import numpy as np


class KalmanTracker2D:
    """Constant-velocity Kalman filter tracking (x, y) position of a drone."""

    def __init__(self, initial_x, initial_y):
        self.x = np.array([initial_x, initial_y, 0.0, 0.0])
        self.P = np.eye(4) * 500.0
        dt = 1.0
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        self.R = np.eye(2) * 15.0
        self.Q = np.eye(4) * 1.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0], self.x[1]

    def update(self, measured_x, measured_y):
        z = np.array([measured_x, measured_y])
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[0], self.x[1]
