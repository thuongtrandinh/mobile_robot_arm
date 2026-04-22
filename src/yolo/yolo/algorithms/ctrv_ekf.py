"""
CTRV EKF (Constant Turn Rate and Velocity - Extended Kalman Filter)
Predicts pedestrian trajectory with curvilinear motion model
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class CTRVState:
    """CTRV state: [x, y, v, psi, omega]
    - (x, y): position in global frame
    - v: velocity magnitude
    - psi: heading angle
    - omega: turn rate (rad/s)
    """
    x: float = 0.0
    y: float = 0.0
    v: float = 0.0  # velocity magnitude
    psi: float = 0.0  # heading angle (radians)
    omega: float = 0.0  # turn rate (rad/s)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.v, self.psi, self.omega])
    
    def from_array(self, arr: np.ndarray):
        self.x, self.y, self.v, self.psi, self.omega = arr


class CTRV_EKF:
    """Extended Kalman Filter with CTRV motion model for pedestrian tracking"""
    
    def __init__(self, dt: float = 1/30.0):
        """
        Args:
            dt: Time step (default 30 Hz = 0.033s)
        """
        self.dt = dt
        self.state = CTRVState()
        
        # Covariance matrices
        self.P = np.eye(5) * 0.1  # State covariance
        self.Q = np.eye(5)  # Process noise - adjusted for pedestrian motion
        self.Q[0:2, 0:2] *= 0.01  # Position noise (tight)
        self.Q[2, 2] = 0.1  # Velocity noise
        self.Q[3, 3] = 0.05  # Heading noise
        self.Q[4, 4] = 0.2  # Turn rate noise (loose - can vary)
        
        self.R = np.eye(2) * 0.1  # Measurement noise (position only)
        
        # Track last position for velocity estimation
        self.last_pos = np.array([0.0, 0.0])
        self.last_time = None
        self.initialized = False
    
    def _motion_model(self, state: np.ndarray, dt: float) -> np.ndarray:
        """CTRV motion model: predict next state
        
        Kinematics with constant turn rate and velocity:
        x_k+1 = x_k + (v/ω)(sin(ψ_k + ω*dt) - sin(ψ_k))
        y_k+1 = y_k + (v/ω)(-cos(ψ_k + ω*dt) + cos(ψ_k))
        ψ_k+1 = ψ_k + ω*dt
        v_k+1 = v_k (constant velocity)
        ω_k+1 = ω_k (constant turn rate)
        
        For near-zero turn rate, use Taylor approximation to avoid division by zero
        """
        x, y, v, psi, omega = state
        
        # Near-zero turn rate → linear motion
        if np.abs(omega) < 1e-6:
            # Linear approximation: x += v*cos(psi)*dt
            x_next = x + v * np.cos(psi) * dt
            y_next = y + v * np.sin(psi) * dt
        else:
            # CTRV model with turn rate
            v_over_w = v / omega
            sin_psi = np.sin(psi)
            sin_psi_new = np.sin(psi + omega * dt)
            cos_psi = np.cos(psi)
            cos_psi_new = np.cos(psi + omega * dt)
            
            x_next = x + v_over_w * (sin_psi_new - sin_psi)
            y_next = y + v_over_w * (-cos_psi_new + cos_psi)
        
        psi_next = psi + omega * dt
        v_next = v
        omega_next = omega
        
        return np.array([x_next, y_next, v_next, psi_next, omega_next])
    
    def _jacobian_F(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Jacobian of motion model for EKF"""
        x, y, v, psi, omega = state
        
        F = np.eye(5)
        
        if np.abs(omega) < 1e-6:
            # Linear motion Jacobian
            cos_psi = np.cos(psi)
            sin_psi = np.sin(psi)
            F[0, 2] = cos_psi * dt  # ∂x/∂v
            F[0, 3] = -v * sin_psi * dt  # ∂x/∂psi
            F[1, 2] = sin_psi * dt  # ∂y/∂v
            F[1, 3] = v * cos_psi * dt  # ∂y/∂psi
        else:
            # CTRV Jacobian
            v_over_w = v / omega
            sin_psi = np.sin(psi)
            sin_psi_new = np.sin(psi + omega * dt)
            cos_psi = np.cos(psi)
            cos_psi_new = np.cos(psi + omega * dt)
            
            # ∂x/∂v
            F[0, 2] = (sin_psi_new - sin_psi) / omega
            # ∂x/∂psi
            F[0, 3] = v_over_w * (cos_psi_new - cos_psi)
            # ∂x/∂ω
            F[0, 4] = (v / (omega * omega)) * (-sin_psi_new + sin_psi) + \
                      (v_over_w) * dt * cos_psi_new
            
            # ∂y/∂v
            F[1, 2] = (-cos_psi_new + cos_psi) / omega
            # ∂y/∂psi
            F[1, 3] = v_over_w * (-sin_psi_new + sin_psi)
            # ∂y/∂ω
            F[1, 4] = (v / (omega * omega)) * (cos_psi_new - cos_psi) + \
                      (v_over_w) * dt * sin_psi_new
            
            F[3, 4] = dt  # ∂ψ/∂ω
        
        return F
    
    def predict(self):
        """EKF predict step"""
        state_array = self.state.to_array()
        
        # Predict state
        state_array = self._motion_model(state_array, self.dt)
        self.state.from_array(state_array)
        
        # Predict covariance
        F = self._jacobian_F(state_array, self.dt)
        self.P = F @ self.P @ F.T + self.Q
    
    def update(self, position: np.ndarray, velocity: np.ndarray):
        """EKF update step with position measurement
        
        Args:
            position: [x, y] measured position
            velocity: [vx, vy] linear velocity
        """
        # Measurement function: we measure x, y only
        # Predict measurement
        z = np.array([self.state.x, self.state.y])
        
        # Measurement Jacobian
        H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0]
        ])
        
        # Innovation
        y = position - z
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        state_array = self.state.to_array()
        state_array = state_array + K @ y
        self.state.from_array(state_array)
        
        # Update covariance
        self.P = (np.eye(5) - K @ H) @ self.P
        
        # FIX: Remove velocity deadzone - allow EKF to update freely for all movements
        # Previously had if vel_mag < 0.015 check that froze velocity, preventing acceleration detection
        vel_mag = np.linalg.norm(velocity)
        
        # Always blend velocity - no deadzone to prevent momentum accumulation
        alpha = 0.3
        self.state.v = (1 - alpha) * self.state.v + alpha * vel_mag
        
        # Update heading safely
        measured_psi = np.arctan2(velocity[1], velocity[0])
        diff = measured_psi - self.state.psi
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        
        self.state.psi += alpha * diff
    
    def predict_trajectory(self, horizon: float = 5.0) -> np.ndarray:
        """Predict future positions over horizon seconds
        
        Args:
            horizon: Prediction time horizon in seconds
            
        Returns:
            Array of predicted positions shape (N, 2) where N = horizon/dt
        """
        N = int(horizon / self.dt)
        trajectory = np.zeros((N, 2))
        
        # Use current state for prediction
        state = self.state.to_array().copy()
        
        for i in range(N):
            trajectory[i] = state[0:2]
            state = self._motion_model(state, self.dt)
        
        return trajectory
    
    def get_predicted_position(self, time_ahead: float) -> np.ndarray:
        """Get single predicted position at time_ahead seconds
        
        Args:
            time_ahead: Seconds into future to predict
            
        Returns:
            [x, y] predicted position
        """
        state = self.state.to_array().copy()
        steps = int(time_ahead / self.dt)
        
        for _ in range(steps):
            state = self._motion_model(state, self.dt)
        
        return state[0:2]
    
    def reset(self):
        """Reset filter state"""
        self.state = CTRVState()
        self.P = np.eye(5) * 0.1
        self.initialized = False


class MultiObjectCTRVTracker:
    """Tracker for multiple pedestrians using CTRV EKF"""
    
    def __init__(self, dt: float = 1/30.0):
        self.dt = dt
        self.filters = {}  # {track_id: CTRV_EKF}
    
    def update(self, track_id: int, position: np.ndarray, velocity: np.ndarray):
        """Update or create EKF for a track
        
        Args:
            track_id: Unique track ID
            position: [x, y] position
            velocity: [vx, vy] velocity
        """
        if track_id not in self.filters:
            self.filters[track_id] = CTRV_EKF(dt=self.dt)
        
        ekf = self.filters[track_id]
        
        if not ekf.initialized:
            # Initialize filter with first measurement
            ekf.state.x = position[0]
            ekf.state.y = position[1]
            ekf.state.v = np.linalg.norm(velocity)
            ekf.state.psi = np.arctan2(velocity[1], velocity[0]) if np.linalg.norm(velocity) > 0.01 else 0
            ekf.state.omega = 0.0
            ekf.initialized = True
        else:
            ekf.predict()
            ekf.update(position, velocity)
    
    def predict(self, track_id: int, time_ahead: float) -> np.ndarray:
        """Predict position for a track
        
        Args:
            track_id: Track ID
            time_ahead: Seconds ahead to predict
            
        Returns:
            [x, y] predicted position, or None if track not found
        """
        if track_id not in self.filters:
            return None
        
        return self.filters[track_id].get_predicted_position(time_ahead)
    
    def get_state(self, track_id: int) -> CTRVState:
        """Get current CTRV state for a track"""
        if track_id not in self.filters:
            return None
        return self.filters[track_id].state
    
    def remove_track(self, track_id: int):
        """Remove lost track"""
        if track_id in self.filters:
            del self.filters[track_id]
    
    def clean_old_tracks(self, active_track_ids: set):
        """Remove tracks not in active set"""
        to_remove = set(self.filters.keys()) - active_track_ids
        for track_id in to_remove:
            del self.filters[track_id]
