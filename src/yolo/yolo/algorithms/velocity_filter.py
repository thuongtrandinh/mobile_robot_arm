"""
Advanced velocity filtering with outlier detection and Kalman prediction
Merged from: yolo/algorithms/velocity_filter.py + state_machine/zed2_optimizer.py

Features:
- Exponential Moving Average (EMA) smoothing
- Velocity outlier detection (statistical z-score)
- Kalman prediction for position during occlusion/gaps
"""

import math
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

            # FIX: Outlier Rejection Trap - prevent division by zero when stationary
            # When standing still, std_v = 0.0, causing z_score to explode when motion starts
            # Use safe minimum std to allow natural acceleration from rest
            # safe_std = 0.3 m/s allows smooth transitions: z = |0.3 - 0| / 0.3 = 1.0 < 2.0 threshold
            safe_std = np.maximum(std_v, 0.3)
            
            # Z-score: how many standard deviations away from mean
            # Only reject if jump exceeds 2 sigma of safe_std (>0.6 m/s in single frame = actual noise)
            z_score = np.abs((velocity - mean_v) / safe_std)

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


class GoalPoseFilter:
    """
    Advanced Goal Pose filter for Navigation Controllers (RL-MPC Optimized).
    
    Tích hợp bộ phân tích xu hướng (Trend Analysis) để hành xử như con người:
    - Người đi ra xa: Đạp ga bám theo bình thường.
    - Người tiến lại gần xe: Chủ động rà phanh từ xa (chậm dần) để nhường đường.
    - Chạm mốc 1.5m: Khóa bánh dừng hẳn mượt mà.
    
    Logic:
    1. Track khoảng cách frame này vs frame trước (delta_dist)
    2. Nếu delta_dist < -0.01m (khoảng cách đang giảm) → nearby approaching
    3. Áp dụng damping 30% tới raw_goal để giảm tốc độ chủ động
    4. Bộ lọc EMA không đối xứng: phanh chậm 0.15, đuổi theo nhanh 0.25
    """
    
    def __init__(self, safe_distance: float = 1.5, 
                 alpha_follow: float = 0.25,   # Tăng tốc để đuổi theo (Phản hồi nhanh)
                 alpha_brake: float = 0.15,    # Giảm tốc mượt mà (Rà phanh êm ái)
                 creep_thresh: float = 0.08):
        """
        Args:
            safe_distance: Target distance to maintain (1.5m)
            alpha_follow: EMA coefficient when expanding goal (person moving away) - faster response
            alpha_brake: EMA coefficient when contracting goal (person moving close) - smoother deceleration
            creep_thresh: Threshold for anti-creep snap-to-zero (8cm)
        """
        self.safe_distance = safe_distance
        self.alpha_follow = alpha_follow
        self.alpha_brake = alpha_brake
        self.creep_thresh = creep_thresh
        
        self.last_goal_x = 0.0
        self.last_goal_y = 0.0
        self.prev_distance = None  # Lưu vết khoảng cách để phân tích xu hướng

    def process(self, ekf_x: float, ekf_y: float) -> Tuple[float, float]:
        """
        Process position and output filtered goal based on approach trend.
        
        Args:
            ekf_x: Filtered person X position (camera frame)
            ekf_y: Filtered person Y position (camera frame)
            
        Returns:
            (goal_x, goal_y): Filtered goal position for Nav2
        """
        distance_to_person = math.hypot(ekf_x, ekf_y)
        
        # Khởi tạo khoảng cách ở frame đầu tiên
        if self.prev_distance is None:
            self.prev_distance = distance_to_person
            
        # 1. PHÂN TÍCH XU HƯỚNG (Trend Analysis)
        delta_dist = distance_to_person - self.prev_distance
        self.prev_distance = distance_to_person
        
        # Nếu khoảng cách giảm > 1cm mỗi frame -> Người đang đi hướng về phía xe
        is_approaching = delta_dist < -0.01 
        
        # 2. TÍNH TOÁN RAW GOAL (Đích đến thô)
        if distance_to_person > self.safe_distance:
            ratio = (distance_to_person - self.safe_distance) / distance_to_person
            raw_goal_x = ekf_x * ratio
            raw_goal_y = ekf_y * ratio
            
            # LOGIC THÔNG MINH: Đang ở xa (VD: 2m) nhưng người tiến lại gần
            if is_approaching:
                # Xe chủ động giảm 70% lực tiến (Damping)
                # Thay vì cố lao lên, xe sẽ lơi ga và đi chậm dần lại chờ người tới
                damping = 0.3 
                raw_goal_x *= damping
                raw_goal_y *= damping
        else:
            # Chạm mốc 1.5m -> Cắt đích về 0 (Lệnh phanh)
            raw_goal_x = 0.0
            raw_goal_y = 0.0
            
        # 3. BỘ LỌC BẤT ĐỐI XỨNG (Asymmetric EMA)
        current_mag = math.hypot(self.last_goal_x, self.last_goal_y)
        raw_mag = math.hypot(raw_goal_x, raw_goal_y)
        
        # Nếu đích đang thu hẹp lại (cần phanh) -> Dùng alpha_brake để hãm từ từ êm ái
        # Nếu đích đang xa ra (cần đuổi theo) -> Dùng alpha_follow để bám sát không bị trễ
        alpha = self.alpha_brake if raw_mag < current_mag else self.alpha_follow
        
        self.last_goal_x = alpha * raw_goal_x + (1 - alpha) * self.last_goal_x
        self.last_goal_y = alpha * raw_goal_y + (1 - alpha) * self.last_goal_y
        
        # 4. CHỐNG TRƯỜN (Anti-creep Snap to Zero)
        if raw_mag == 0.0 and math.hypot(self.last_goal_x, self.last_goal_y) < self.creep_thresh:
            self.last_goal_x = 0.0
            self.last_goal_y = 0.0
            
        return float(self.last_goal_x), float(self.last_goal_y)
        
    def reset(self):
        """Khôi phục biến trạng thái khi mất dấu hoặc reset IDLE"""
        self.last_goal_x = 0.0
        self.last_goal_y = 0.0
        self.prev_distance = None
