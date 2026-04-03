#!/usr/bin/env python3
"""
State Machine Node - Manages robot states based on hand gestures
States: IDLE -> START -> TRACKING -> RE_TRACKING -> STOP -> IDLE

Optimized for ZED2 camera (FOV: chest to feet), no body skeleton tracking needed
Integrated obstacle avoidance with dynamic obstacles detection
"""

import rclpy
import time
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from interfaces.msg import HumanState, Obstacles, Obstacles

from .state_manager import RobotStateMachine, StateConfig, TrackingTarget, RobotState


class RobotStateMachineNode(Node):
    """
    Simplified state machine node for hand gesture-based robot control
    """
    
    def __init__(self):
        super().__init__('robot_state_machine')
        
        # ===== 1. LOAD PARAMETERS =====
        # Timing requirements from specification
        self.declare_parameter('gesture_hold_time', 0.8)      # 1.0s START, 2.0s STOP (using 0.8s base)
        self.declare_parameter('tracking_timeout', 1.0)       # Lost target → Re-TRACKING timeout
        self.declare_parameter('retracking_timeout', 5.0)     # Re-TRACKING → IDLE timeout
        self.declare_parameter('velocity_filter_alpha', 0.3)  # EMA smoothing
        
        gesture_hold = self.get_parameter('gesture_hold_time').value
        tracking_timeout = self.get_parameter('tracking_timeout').value
        retrack_timeout = self.get_parameter('retracking_timeout').value
        vel_alpha = self.get_parameter('velocity_filter_alpha').value
        
        # ===== 2. INITIALIZE STATE MACHINE =====
        state_config = StateConfig(
            gesture_hold_time=gesture_hold,
            tracking_timeout=tracking_timeout,
            retracking_timeout=retrack_timeout,
            velocity_filter_alpha=vel_alpha,
        )
        self.state_machine = RobotStateMachine(state_config)
        
        # ===== 3. VELOCITY FILTERING (Simple exponential smoothing) =====
        self.vel_alpha = vel_alpha
        self.last_vx = 0.0
        self.last_vy = 0.0
        
        # ===== 4. STATE CALLBACKS =====
        self.state_machine.register_state_callback(RobotState.START, self._on_start)
        self.state_machine.register_state_callback(RobotState.TRACKING, self._on_tracking)
        self.state_machine.register_state_callback(RobotState.RE_TRACKING, self._on_retracking)
        self.state_machine.register_state_callback(RobotState.STOP, self._on_stop)
        self.state_machine.register_state_callback(RobotState.IDLE, self._on_idle)
        
        # ===== 5. PUBLISHERS =====
        qos_best = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.state_pub = self.create_publisher(String, '/system/robot_status', 10)
        self.system_state_pub = self.create_publisher(String, '/system_state', 10)  # Decision layer output
        self.predicted_target_pub = self.create_publisher(HumanState, '/tracking/predicted_target', qos_best)
        
        # ===== 6. SUBSCRIBERS =====
        # Subscribe to selected person from person_selector_node
        self.create_subscription(HumanState, '/tracking/main_person', self._person_callback, qos_best)
        # Subscribe to dynamic obstacles (other persons) for avoidance
        self.create_subscription(Obstacles, '/tracking/dynamic_obstacles', self._obstacles_callback, qos_best)
        # Subscribe to gesture commands from handsign_detector_node
        self.create_subscription(String, '/yolo/gesture_command', self._gesture_callback, 10)
        
        # ===== 7. CONTROL LOOP TIMER (20 Hz) =====
        self.create_timer(0.05, self._control_loop)
        
        # ===== 8. INTERNAL STATE =====
        self.last_detection_time = time.time()
        self.last_person_state = None
        self.dynamic_obstacles = None  # Current obstacles for avoidance
        
        self.get_logger().info(
            f'✅ State Machine Node initialized (Decision Layer)\n'
            f'   States: IDLE → START(1.0s) → TRACKING → RE_TRACKING(5.0s) → IDLE\n'
            f'   Timing:\n'
            f'      START gesture: ≤1.0s to lock target\n'
            f'      STOP gesture: ≤2.0s to unlock target\n'
            f'      Detection timeout: {tracking_timeout}s (TRACKING → RE_TRACKING)\n'
            f'      Re-tracking timeout: {retrack_timeout}s (RE_TRACKING → IDLE)\n'
            f'   Velocity smoothing: α={vel_alpha}\n'
            f'   Output: /system_state (IDLE|TRACKING|RE_TRACKING)'
        )
    
    def _smooth_velocity(self, vx_new, vy_new):
        """Apply exponential smoothing to velocity"""
        self.last_vx = self.vel_alpha * vx_new + (1 - self.vel_alpha) * self.last_vx
        self.last_vy = self.vel_alpha * vy_new + (1 - self.vel_alpha) * self.last_vy
        return self.last_vx, self.last_vy
    
    def _person_callback(self, msg: HumanState):
        """Handle person updates from person_selector_node (using correct field names)"""
        current_time = time.time()
        self.last_detection_time = current_time
        self.last_person_state = msg
        
        # Apply velocity smoothing  (px, py, vx, vy fields)
        vx_smooth, vy_smooth = self._smooth_velocity(msg.vx, msg.vy)
        
        # Update state machine with tracking target
        target = TrackingTarget(
            id=0,  # Single person tracking
            px=msg.px,      # Position X (normalized -1 to 1)
            py=msg.py,      # Position Y (normalized -1 to 1)
            vx=vx_smooth,   # Velocity X
            vy=vy_smooth,   # Velocity Y
            last_detection_time=current_time,
            last_position_update_time=current_time,
        )
        
        self.state_machine.update_tracked_target(target, current_time)
    
    def _obstacles_callback(self, msg: Obstacles):
        """Handle dynamic obstacles (other persons nearby for avoidance)"""
        self.dynamic_obstacles = msg
        if msg.circles:
            self.get_logger().debug(f'⚠️ Dynamic obstacles detected: {len(msg.circles)} persons')
    
    def _gesture_callback(self, msg: String):
        """
        Handle gesture detection from YOLO
        Format: "START(850ms)<p=3>" or "STOP(1200ms)<p=3>"
        Simplified - only parse gesture type (START/STOP)
        """
        try:
            # Extract gesture type (before parenthesis)
            gesture_str = msg.data.split('(')[0].strip()
            gesture_class = 0 if gesture_str == "START" else 1
            
            # Dummy position (will be updated from HumanState callback)
            # state_manager only needs gesture_class + person_bbox overlap check
            hand_pos = (0.0, 0.0)  # Placeholder
            person_bbox = (0.0, 0.0, 100.0, 100.0)  # Placeholder
            
            self.state_machine.handle_gesture(
                gesture_class, 
                hand_pos,
                person_bbox,
                time.time()
            )
            
            self.get_logger().debug(f'🎯 Gesture: {gesture_str}')
        except Exception as e:
            self.get_logger().warning(f'Gesture parse error: {e}', throttle_duration_sec=2.0)
    
    def _control_loop(self):
        """Main control loop (20Hz) - process timeouts and state transitions"""
        current_time = time.time()
        
        # Check for detection timeout (0.5s tracking_timeout, 5.0s re-tracking_timeout)
        self.state_machine.update_no_detection(current_time)
        
        # Publish decision layer output
        state_msg = String()
        state_msg.data = self.state_machine.current_state.value
        self.state_pub.publish(state_msg)
        
        # Also publish to /system_state for robot controller
        self.system_state_pub.publish(state_msg)
        
        # Predict position during RE_TRACKING (velocity-based extrapolation)
        if self.state_machine.current_state == RobotState.RE_TRACKING and self.last_person_state:
            self._publish_prediction()
    
    def _publish_prediction(self):
        """Publish simple velocity-based prediction during RE_TRACKING"""
        if self.last_person_state is None:
            return
            
        # Simple 0.5s ahead prediction using constant velocity model
        dt_pred = 0.5
        msg = HumanState()
        msg.px = self.last_person_state.px + self.last_vx * dt_pred
        msg.py = self.last_person_state.py + self.last_vy * dt_pred
        msg.vx = self.last_vx
        msg.vy = self.last_vy
        msg.radius = self.last_person_state.radius
        msg.trajectory = []
        
        self.predicted_target_pub.publish(msg)
    
    # ===== STATE CALLBACKS =====
    def _on_start(self):
        """Called when entering START state"""
        self.get_logger().info('🎯 START: Gesture detected, preparing to track')
    
    def _on_tracking(self):
        """Called when entering TRACKING state"""
        self.get_logger().info('🚀 TRACKING: Following target')
    
    def _on_retracking(self):
        """Called when entering RE_TRACKING state"""
        self.get_logger().warn('⚠️ RE_TRACKING: Predicting target (lost for <3s)')
    
    def _on_stop(self):
        """Called when entering STOP state"""
        self.get_logger().info('⏹️ STOP: Gesture detected, stopping tracking')
    
    def _on_idle(self):
        """Called when entering IDLE state"""
        self.get_logger().info('⏸️ IDLE: Waiting for START gesture')


def main(args=None):
    rclpy.init(args=args)
    node = RobotStateMachineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
