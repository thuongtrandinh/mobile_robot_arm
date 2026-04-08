"""
Advanced velocity filtering with outlier detection and Kalman prediction
Merged from: yolo/algorithms/velocity_filter.py + state_machine/zed2_optimizer.py

Features:
- Exponential Moving Average (EMA) smoothing
- Velocity outlier detection (statistical z-score)
- Kalman prediction for position during occlusion/gaps
"""

import numpy as np
from typing import Optional, List, Tuple


class VelocityFilter:
    """
    Advanced velocity filter with EMA smoothing and outlier detection.

    Replaces simple EMA with:
    - Velocity history tracking
    - Outlier detection using standard deviation
    - Graceful fallback to last good value on anomaly
    """

    def __init__(self, alpha: float = 0.3, history_size: int = 5):
        """
        Args:
            alpha: EMA smoothing factor (0 < alpha <= 1)
                - Higher alpha (0.5-1.0): Faster response, more noise
                - Lower alpha (0.1-0.3): Smoother, more lag
            history_size: Number of past velocities to keep for outlier detection
        """
        self.alpha = alpha
        self.history_size = history_size
        self.velocity_history: List[np.ndarray] = []
        self.filtered_velocity = None

    def update(self, velocity: np.ndarray) -> np.ndarray:
        """
        Update filter with new velocity measurement.

        Detects outliers and applies EMA smoothing.

        Args:
            velocity: [vx, vy] or [vx, vy, vz] velocity vector

        Returns:
            filtered_velocity: smoothed velocity (same shape as input)
        """
        velocity = np.asarray(velocity, dtype=np.float32)

        # Outlier detection: compare against history
        if len(self.velocity_history) >= 2:
            history_array = np.array(self.velocity_history)
            mean_v = np.mean(history_array, axis=0)
            std_v = np.std(history_array, axis=0)

            # Z-score: how many standard deviations away from mean
            z_score = np.abs((velocity - mean_v) / (std_v + 1e-6))

            # If any component is outlier (z > 2.0), skip update and return last good value
            if np.any(z_score > 2.0):
                return self.filtered_velocity.copy() if self.filtered_velocity is not None else velocity

        # Apply EMA smoothing
        if self.filtered_velocity is None:
            self.filtered_velocity = velocity.copy()
        else:
            self.filtered_velocity = (self.alpha * velocity +
                                     (1 - self.alpha) * self.filtered_velocity)

        # Maintain history for next outlier detection
        self.velocity_history.append(velocity.copy())
        if len(self.velocity_history) > self.history_size:
            self.velocity_history.pop(0)

        return self.filtered_velocity.copy()

    def is_outlier(self, velocity: np.ndarray, threshold: float = 2.0) -> bool:
        """
        Check if velocity is statistical outlier.

        Args:
            velocity: [vx, vy] velocity to check
            threshold: Z-score threshold (default 2.0 sigma)

        Returns:
            True if outlier, False otherwise
        """
        if len(self.velocity_history) < 2:
            return False

        history_array = np.array(self.velocity_history)
        mean_v = np.mean(history_array, axis=0)
        std_v = np.std(history_array, axis=0)

        z_score = np.abs((velocity - mean_v) / (std_v + 1e-6))
        return np.any(z_score > threshold)

    def reset(self):
        """Reset filter state for new track"""
        self.filtered_velocity = None
        self.velocity_history.clear()


class KalmanPredictor:
    """
    Simple Kalman filter for tracking position and velocity during occlusion.

    State: [x, y, vx, vy]
    - (x, y): 2D position
    - (vx, vy): 2D velocity

    Used for:
    - Predicting target position when temporarily out of view
    - Smooth trajectory during detection gaps
    - Velocity estimation if depth measurement fails
    """

    def __init__(self, dt: float = 0.033):
        """
        Args:
            dt: Time step between updates (seconds). Default 0.033 ≈ 30 FPS.
        """
        self.dt = dt

        # State vector: [x, y, vx, vy]
        self.state = np.zeros(4, dtype=np.float32)

        # State covariance (uncertainty)
        self.P = np.eye(4, dtype=np.float32) * 0.1

        # Process noise covariance (how much we trust motion model)
        self.Q = np.eye(4, dtype=np.float32)
        self.Q[0:2, 0:2] *= 0.01   # Position noise (tight - we trust transitions)
        self.Q[2:4, 2:4] *= 0.1    # Velocity noise (looser - velocities vary)

        # Measurement noise covariance (how much we trust measurements)
        self.R = np.eye(2, dtype=np.float32) * 0.05
        self.initialized = False

    def predict(self, dt: Optional[float] = None) -> Tuple[float, float, float, float]:
        """
        Predict next state without measurement.

        Used when detection is missing (occlusion, tracking gap).

        Args:
            dt: Time step (uses self.dt if None)

        Returns:
            (x_pred, y_pred, vx_pred, vy_pred)
        """
        dt = dt or self.dt

        # State transition matrix: position += velocity * dt
        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # Predict state
        self.state = F @ self.state

        # Predict covariance: P = F @ P @ F^T + Q
        self.P = F @ self.P @ F.T + self.Q

        return (float(self.state[0]), float(self.state[1]),
                float(self.state[2]), float(self.state[3]))

    def update(self, x: float, y: float, vx: float, vy: float):
        """
        Update state with new position and velocity measurement.

        Args:
            x, y: Measured position
            vx, vy: Measured velocity
        """
        z = np.array([x, y], dtype=np.float32)

        # Measurement matrix: we measure position only
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float32)

        # Innovation: difference between measurement and prediction
        innovation = z - (H @ self.state)[:2]

        # Innovation covariance: S = H @ P @ H^T + R
        S = H @ self.P @ H.T + self.R

        # Kalman gain: K = P @ H^T @ S^-1
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Singular matrix, skip update
            return

        # Update state: x = x + K @ innovation
        self.state[:2] += (K @ innovation).flatten()

        # Update velocity directly (from measurement)
        self.state[2] = vx
        self.state[3] = vy

        # Update covariance: P = (I - K @ H) @ P
        self.P = (np.eye(4, dtype=np.float32) - K @ H) @ self.P

        self.initialized = True

    def get_state(self) -> Tuple[float, float, float, float]:
        """Get current state: (x, y, vx, vy)"""
        return (float(self.state[0]), float(self.state[1]),
                float(self.state[2]), float(self.state[3]))

    def reset(self):
        """Reset Kalman filter state"""
        self.state = np.zeros(4, dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 0.1
        self.initialized = False
