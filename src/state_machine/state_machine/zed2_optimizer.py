"""
ZED2 Camera Tracking Optimizer
Optimized for chest-to-feet FOV view with velocity filtering and prediction
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List
import math


@dataclass
class ZED2Config:
    """ZED2 camera configuration for tracking optimization"""
    # FOV characteristics (ZED2 standard: ~110 degrees)
    fov_horizontal: float = 110.0          # degrees
    fov_vertical: float = 70.0             # degrees
    
    # Camera positioning (chest-down view)
    camera_height_from_ground: float = 1.8  # meters, typical person height
    min_viewing_distance: float = 0.3       # meters
    max_viewing_distance: float = 5.0       # meters
    
    # For person detection optimization
    person_height_expected: float = 1.7    # meters
    person_width_expected: float = 0.5    # meters
    
    # Velocity filtering (exponential smoothing)
    velocity_alpha: float = 0.3            # Smoothing factor
    velocity_history_size: int = 5         # Keep last N measurements
    
    # Outlier detection
    velocity_std_threshold: float = 2.0    # Standard deviations for outlier
    max_acceleration: float = 2.0          # m/s^2 (human walking max)
    
    # Re-tracking with prediction
    prediction_horizon: float = 0.5        # seconds ahead


class VelocityFilter:
    """Exponential smoothing filter for velocity estimation"""
    
    def __init__(self, alpha: float = 0.3, history_size: int = 5):
        self.alpha = alpha
        self.history_size = history_size
        self.velocity_history: List[float] = []
        self.last_velocity = np.array([0.0, 0.0])
    
    def filter(self, vx: float, vy: float) -> tuple:
        """Apply exponential smoothing to velocity"""
        velocity = np.array([vx, vy])
        
        # First measurement
        if len(self.velocity_history) == 0:
            self.last_velocity = velocity
            self.velocity_history.append(velocity)
            return (vx, vy)
        
        # Exponential smoothing: v_smooth = alpha * v_measured + (1-alpha) * v_previous
        smoothed = self.alpha * velocity + (1 - self.alpha) * self.last_velocity
        self.last_velocity = smoothed
        
        # Keep history for outlier detection
        self.velocity_history.append(velocity)
        if len(self.velocity_history) > self.history_size:
            self.velocity_history.pop(0)
        
        return tuple(smoothed)
    
    def is_outlier(self, vx: float, vy: float, threshold: float = 2.0) -> bool:
        """Detect velocity outliers using standard deviation"""
        if len(self.velocity_history) < 2:
            return False
        
        history_array = np.array(self.velocity_history)
        mean_v = np.mean(history_array, axis=0)
        std_v = np.std(history_array, axis=0)
        
        z_score = np.abs((np.array([vx, vy]) - mean_v) / (std_v + 1e-6))
        return np.any(z_score > threshold)


class KalmanPredictor:
    """Simple Kalman predictor for target position during occlusion"""
    
    def __init__(self, dt: float = 0.033):
        self.dt = dt
        
        # State: [x, y, vx, vy]
        self.state = np.zeros(4)
        self.P = np.eye(4) * 0.1  # Covariance
        
        # Process model
        self.Q = np.eye(4)
        self.Q[0:2, 0:2] *= 0.01   # Position noise (low)
        self.Q[2:4, 2:4] *= 0.1    # Velocity noise (higher)
        
        # Measurement noise
        self.R = np.eye(2) * 0.05
    
    def predict(self, dt: Optional[float] = None) -> tuple:
        """Predict next position and velocity"""
        dt = dt or self.dt
        
        # F matrix: position += velocity * dt
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Predict state
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q
        
        return (self.state[0], self.state[1], self.state[2], self.state[3])
    
    def update(self, x: float, y: float, vx: float, vy: float):
        """Update with new measurement"""
        z = np.array([x, y])
        
        # Kalman gain
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        y_residual = z - (H @ self.state)[:2]
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T / (S + 1e-6)
        
        # Update state
        self.state[:2] += (K @ y_residual).flatten()
        self.state[2] = vx
        self.state[3] = vy
        
        # Update covariance
        self.P = (np.eye(4) - K @ H) @ self.P


class ZED2TrackingOptimizer:
    """
    Optimizes tracking for ZED2 camera with chest-to-feet FOV
    """
    
    def __init__(self, config: Optional[ZED2Config] = None):
        self.config = config or ZED2Config()
        self.velocity_filter = VelocityFilter(
            alpha=self.config.velocity_alpha,
            history_size=self.config.velocity_history_size
        )
        self.kalman_predictor = KalmanPredictor()
    
    def filter_detections(self, detections: List[dict]) -> List[dict]:
        """
        Filter and validate detections based on ZED2 FOV
        Args:
            detections: List of detection dicts with 'px', 'py', 'dist', 'bbox'
        Returns:
            Filtered detections
        """
        valid_detections = []
        
        for det in detections:
            # Distance check
            if not (self.config.min_viewing_distance <= det.get('dist', 0) <= self.config.max_viewing_distance):
                continue
            
            # FOV check (approximate angular field of view)
            px, py = det.get('px', 0), det.get('py', 0)
            angle_rad = math.atan2(px, py)  # Assuming Y is forward
            angle_deg = math.degrees(angle_rad)
            
            if abs(angle_deg) > self.config.fov_horizontal / 2:
                continue
            
            valid_detections.append(det)
        
        return valid_detections
    
    def smooth_velocity(self, vx: float, vy: float, 
                       check_outliers: bool = True) -> tuple:
        """
        Smooth velocity with outlier detection
        """
        if check_outliers:
            if self.velocity_filter.is_outlier(vx, vy, self.config.velocity_std_threshold):
                # Use last known good velocity
                return self.velocity_filter.last_velocity
        
        return self.velocity_filter.filter(vx, vy)
    
    def predict_position(self, px: float, py: float, vx: float, vy: float, 
                        dt: float) -> tuple:
        """
        Predict position during occlusion
        Args:
            px, py: Current position
            vx, vy: Velocity
            dt: Time to predict ahead
        Returns:
            Predicted (px_pred, py_pred)
        """
        # Update Kalman filter with current state
        self.kalman_predictor.update(px, py, vx, vy)
        
        # Predict ahead
        _, _, pred_lines = self.kalman_predictor.predict(dt)
        x_pred, y_pred, _, _ = self.kalman_predictor.predict(dt)
        
        return (x_pred, y_pred)
    
    def estimate_person_distance(self, bbox_height_pixels: int, 
                                image_height_pixels: int) -> float:
        """
        Estimate person distance using bounding box height
        Useful for validating ZED2 depth measurements
        """
        if bbox_height_pixels == 0:
            return self.config.max_viewing_distance
        
        # Intrinsic camera parameters (approximate for ZED2)
        focal_length = image_height_pixels / (2 * math.tan(math.radians(self.config.fov_vertical / 2)))
        
        # Distance = (Person height * focal length) / bbox height
        estimated_distance = (self.config.person_height_expected * focal_length) / bbox_height_pixels
        
        # Clamp to valid range
        return min(self.config.max_viewing_distance, 
                  max(self.config.min_viewing_distance, estimated_distance))
    
    def is_detection_stable(self, current_det: dict, last_det: dict, 
                           max_displacement: float = 0.5) -> bool:
        """
        Check if detection is stable (prevent tracking noise)
        """
        dx = current_det.get('px', 0) - last_det.get('px', 0)
        dy = current_det.get('py', 0) - last_det.get('py', 0)
        displacement = math.sqrt(dx*dx + dy*dy)
        
        return displacement <= max_displacement
    
    def get_optimized_parameters(self) -> dict:
        """Get recommended parameters for ROS2 launch"""
        return {
            'velocity_alpha': self.config.velocity_alpha,
            'min_viewing_distance': self.config.min_viewing_distance,
            'max_viewing_distance': self.config.max_viewing_distance,
            'gesture_detection_margin': int(math.tan(math.radians(self.config.fov_horizontal / 4)) * 100),
            'tracking_timeout': 0.3,
            'retracking_timeout': 2.5,
        }
