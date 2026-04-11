"""
Optimized State Machine Manager for Mobile Robot Arm
Handles state transitions: IDLE -> START -> TRACKING -> RE-TRACKING -> STOP -> IDLE
TRUSTS PERCEPTION LAYER: Transitions instantly based on YOLO node commands.
No duplicate timing - tracking_node.py already handles all gesture filtering and timing.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable
import time


class RobotState(Enum):
    IDLE = "IDLE"                      
    START = "START"                    
    TRACKING = "TRACKING"              
    RE_TRACKING = "RE_TRACKING"  # [ĐÃ SỬA] Khớp với interface yêu cầu
    STOP = "STOP"                      


@dataclass
class StateConfig:
    # Timing (Gesture hold removed because tracking_node.py already handles it)
    tracking_timeout: float = 0.3          
    retracking_timeout: float = 10.0       # [SỬA] Tăng từ 3.0 → 10.0 để đồng bộ với tracking_node.py
    start_gesture_class: int = 0           
    stop_gesture_class: int = 1            
    velocity_filter_alpha: float = 0.3     
    velocity_threshold: float = 3.0


@dataclass
class TrackingTarget:
    id: int
    px: float                          
    py: float                          
    vx: float                          
    vy: float                          
    last_detection_time: float         
    last_position_update_time: float   
    
    def is_valid_velocity(self, threshold: float = 3.0) -> bool:
        import math
        speed = math.sqrt(self.vx**2 + self.vy**2)
        return speed <= threshold


class RobotStateMachine:
    def __init__(self, config: Optional[StateConfig] = None):
        self.config = config or StateConfig()
        self.current_state = RobotState.IDLE
        self.previous_state = RobotState.IDLE
        
        self.tracked_target: Optional[TrackingTarget] = None
        self.target_id: Optional[int] = None
        self._state_callbacks: dict = {}
        
        self.state_change_count = 0
        self.total_tracking_duration = 0.0
        self._tracking_start_time: Optional[float] = None
        
    def register_state_callback(self, state: RobotState, callback: Callable):
        if state not in self._state_callbacks:
            self._state_callbacks[state] = []
        self._state_callbacks[state].append(callback)
    
    def _trigger_state_callbacks(self, state: RobotState):
        if state in self._state_callbacks:
            for callback in self._state_callbacks[state]:
                try:
                    callback()
                except Exception as e:
                    print(f"Error in state callback: {e}")
    
    def update_tracked_target(self, target: TrackingTarget, current_time: float):
        current_time_ns = time.time()
        
        if not target.is_valid_velocity(self.config.velocity_threshold):
            return
        
        self.tracked_target = target
        self.target_id = target.id
        
        # Automatically transition to TRACKING when target appears
        if self.current_state == RobotState.START:
            self._transition_to(RobotState.TRACKING, current_time_ns)
            
        elif self.current_state == RobotState.RE_TRACKING:
            self._transition_to(RobotState.TRACKING, current_time_ns)
            self.tracked_target.last_detection_time = current_time_ns
    
    def handle_gesture(self, gesture_class: int, current_time: float):
        """
        INSTANT TRANSITION: Skip timing, trust YOLO node filtering.
        YOLO tracking_node.py already verified gesture for 1 second before sending.
        """
        current_time_ns = time.time()

        if gesture_class == self.config.start_gesture_class:  # START
            if self.current_state == RobotState.IDLE:
                self._transition_to(RobotState.START, current_time_ns)
                
        elif gesture_class == self.config.stop_gesture_class:  # STOP
            if self.current_state in [RobotState.TRACKING, RobotState.RE_TRACKING]:
                self._transition_to(RobotState.STOP, current_time_ns)
                self._transition_to(RobotState.IDLE, current_time_ns)
    
    def update_no_detection(self, current_time: float):
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
        self._transition_to(RobotState.IDLE, time.time())
    
    def _transition_to(self, new_state: RobotState, current_time: float):
        if new_state == self.current_state:
            return
        
        old_state = self.current_state
        self.current_state = new_state
        self.state_change_count += 1
        
        if old_state == RobotState.TRACKING and self._tracking_start_time:
            self.total_tracking_duration += current_time - self._tracking_start_time
        
        if new_state == RobotState.TRACKING or new_state == RobotState.RE_TRACKING:
            self._tracking_start_time = current_time
        elif new_state == RobotState.IDLE:
            self.tracked_target = None
            self.target_id = None
            self._tracking_start_time = None
        
        self._trigger_state_callbacks(new_state)
