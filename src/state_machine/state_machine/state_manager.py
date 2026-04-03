"""
Optimized State Machine Manager for Mobile Robot Arm
Handles state transitions: IDLE -> START -> TRACKING -> RE-TRACKING -> STOP -> IDLE
Optimized for ZED2 camera with limited FOV (chest down to feet)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable
import time


class RobotState(Enum):
    """Robot operational states"""
    IDLE = "IDLE"                      # Waiting for start gesture
    START = "START"                    # Start gesture detected, preparing to track
    TRACKING = "TRACKING"              # Active target tracking
    RE_TRACKING = "RE_TRACKING"        # Target lost, predicting position
    STOP = "STOP"                      # Stop gesture detected, transitioning to IDLE


@dataclass
class StateConfig:
    """Configuration for state machine behavior"""
    # Timing thresholds (seconds)
    gesture_hold_time: float = 1.5         # Hold gesture to confirm (reduced from 2.0)
    tracking_timeout: float = 0.3          # Time to switch to RE_TRACKING (reduced from 0.2 for ZED2)
    retracking_timeout: float = 3.0        # Time to give up and go IDLE
    
    # Gestures
    start_gesture_class: int = 0           # Hand class for START
    stop_gesture_class: int = 1            # Hand class for STOP
    gesture_detection_margin: int = 150    # Pixel margin for gesture-person overlap (increased for FOV)
    
    # ZED2 optimization
    min_detection_confidence: float = 0.5  # Min confidence for person detection
    velocity_filter_alpha: float = 0.3     # Exponential smoothing for velocity
    use_velocity_prediction: bool = True   # Enable velocity-based prediction
    
    # Tracking robustness
    max_tracked_persons: int = 5           # Limit FOV analysis
    distance_threshold: float = 5.0        # Max tracking distance (meters)
    velocity_threshold: float = 3.0        # Max realistic velocity (m/s)


@dataclass
class TrackingTarget:
    """Target person being tracked"""
    id: int
    px: float                          # Position X (m)
    py: float                          # Position Y (m)
    vx: float                          # Velocity X (m/s)
    vy: float                          # Velocity Y (m/s)
    last_detection_time: float         # Timestamp of last detection
    last_position_update_time: float   # Last position update time
    
    def is_valid_velocity(self, threshold: float = 3.0) -> bool:
        """Check if velocity is physically realistic"""
        import math
        speed = math.sqrt(self.vx**2 + self.vy**2)
        return speed <= threshold


class RobotStateMachine:
    """
    Optimized state machine for robot tracking with ZED2 camera
    Manages transitions between operational states with proper timing and conditions
    """
    
    def __init__(self, config: Optional[StateConfig] = None):
        self.config = config or StateConfig()
        self.current_state = RobotState.IDLE
        self.previous_state = RobotState.IDLE
        
        # Target tracking
        self.tracked_target: Optional[TrackingTarget] = None
        self.target_id: Optional[int] = None
        
        # Gesture timing
        self._gesture_start_time: Optional[float] = None
        self._last_gesture_type: Optional[int] = None
        
        # State transition callbacks
        self._state_callbacks: dict = {}
        
        # Statistics
        self.state_change_count = 0
        self.total_tracking_duration = 0.0
        self._tracking_start_time: Optional[float] = None
        
    def register_state_callback(self, state: RobotState, callback: Callable):
        """Register callback when entering a state"""
        if state not in self._state_callbacks:
            self._state_callbacks[state] = []
        self._state_callbacks[state].append(callback)
    
    def _trigger_state_callbacks(self, state: RobotState):
        """Trigger all callbacks for state entry"""
        if state in self._state_callbacks:
            for callback in self._state_callbacks[state]:
                try:
                    callback()
                except Exception as e:
                    print(f"Error in state callback: {e}")
    
    def update_tracked_target(self, target: TrackingTarget, current_time: float):
        """Update target position and handle state transitions"""
        current_time_ns = time.time()
        
        # Validate target
        if not target.is_valid_velocity(self.config.velocity_threshold):
            return
        
        # Update target
        old_target = self.tracked_target
        self.tracked_target = target
        self.target_id = target.id
        
        # Update state based on current state
        if self.current_state == RobotState.START:
            self._transition_to(RobotState.TRACKING, current_time_ns)
        elif self.current_state == RobotState.TRACKING:
            # Validate target still exists
            if target.id == self.target_id:
                pass  # Stay in TRACKING
        elif self.current_state == RobotState.RE_TRACKING:
            # Re-acquired target
            if target.id == self.target_id:
                self._transition_to(RobotState.TRACKING, current_time_ns)
                self.tracked_target.last_detection_time = current_time_ns
    
    def handle_gesture(self, gesture_class: int, detected_position: tuple, 
                      person_bbox: tuple, current_time: float):
        """
        Handle hand gesture detection
        Args:
            gesture_class: 0=START, 1=STOP
            detected_position: (x, y) pixel center of gesture
            person_bbox: (x1, y1, x2, y2) from handsign detector
            current_time: Timestamp in seconds
        """
        current_time_ns = time.time()

        if detected_position is None or person_bbox is None:
            return
        
        hx, hy = detected_position
        px1, py1, px2, py2 = person_bbox
        
        # Check if gesture overlaps with person (with margin for ZED2 FOV)
        gesture_in_person = (
            px1 - self.config.gesture_detection_margin < hx < px2 + self.config.gesture_detection_margin
            and py1 - self.config.gesture_detection_margin < hy < py2 + self.config.gesture_detection_margin
        )
        
        if not gesture_in_person:
            return
        
        # Process gesture
        if gesture_class == self.config.start_gesture_class:  # START
            self._handle_start_gesture(current_time_ns)
        elif gesture_class == self.config.stop_gesture_class:  # STOP
            self._handle_stop_gesture(current_time_ns)
    
    def _handle_start_gesture(self, current_time: float):
        """Handle START gesture detection"""
        if self._last_gesture_type != 0:  # New gesture type
            self._gesture_start_time = current_time
            self._last_gesture_type = 0
        elif self._gesture_start_time is not None:
            elapsed = current_time - self._gesture_start_time
            
            if elapsed >= self.config.gesture_hold_time:
                if self.current_state == RobotState.IDLE:
                    self._transition_to(RobotState.START, current_time)
                self._gesture_start_time = None
                self._last_gesture_type = None
    
    def _handle_stop_gesture(self, current_time: float):
        """Handle STOP gesture detection"""
        if self._last_gesture_type != 1:  # New gesture type
            self._gesture_start_time = current_time
            self._last_gesture_type = 1
        elif self._gesture_start_time is not None:
            elapsed = current_time - self._gesture_start_time
            
            if elapsed >= self.config.gesture_hold_time:
                if self.current_state in [RobotState.TRACKING, RobotState.RE_TRACKING]:
                    self._transition_to(RobotState.STOP, current_time)
                self._gesture_start_time = None
                self._last_gesture_type = None
    
    def update_no_detection(self, current_time: float):
        """Handle when no detection is received"""
        current_time_ns = time.time()
        
        if self.current_state == RobotState.TRACKING and self.tracked_target is not None:
            elapsed = current_time_ns - self.tracked_target.last_detection_time
            
            if elapsed >= self.config.tracking_timeout:
                self._transition_to(RobotState.RE_TRACKING, current_time_ns)
        
        elif self.current_state == RobotState.RE_TRACKING:
            elapsed = current_time_ns - self._tracking_start_time if self._tracking_start_time else 0
            
            if elapsed >= self.config.retracking_timeout:
                self._transition_to(RobotState.IDLE, current_time_ns)
    
    def force_idle(self):
        """Force state machine to IDLE (emergency stop)"""
        self._transition_to(RobotState.IDLE, time.time())
    
    def _transition_to(self, new_state: RobotState, current_time: float):
        """Perform state transition with proper cleanup and callbacks"""
        if new_state == self.current_state:
            return
        
        old_state = self.current_state
        self.current_state = new_state
        self.state_change_count += 1
        
        # Update tracking duration
        if old_state == RobotState.TRACKING and self._tracking_start_time:
            self.total_tracking_duration += current_time - self._tracking_start_time
        
        # State-specific initialization
        if new_state == RobotState.TRACKING:
            self._tracking_start_time = current_time
        elif new_state == RobotState.RE_TRACKING:
            self._tracking_start_time = current_time
        elif new_state == RobotState.IDLE:
            self.tracked_target = None
            self.target_id = None
            self._tracking_start_time = None
        
        # Reset gesture state on IDLE
        if new_state == RobotState.IDLE:
            self._gesture_start_time = None
            self._last_gesture_type = None
        
        # Trigger callbacks
        self._trigger_state_callbacks(new_state)
    
    def get_state_string(self) -> str:
        """Get human-readable state string"""
        return self.current_state.value
    
    def get_stats(self) -> dict:
        """Get state machine statistics"""
        return {
            'current_state': self.current_state.value,
            'state_changes': self.state_change_count,
            'total_tracking_time': self.total_tracking_duration,
            'has_target': self.target_id is not None,
            'target_id': self.target_id,
        }
