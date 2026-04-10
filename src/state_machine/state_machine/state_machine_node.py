#!/usr/bin/env python3
"""
Optimized State Machine Node - INSTANT RESPONSE MODE
Receives gesture commands directly from tracking_node.py
trusts that tracking_node has already:
  - Verified gesture for 1 second (buffer voting)
  - Associated hand with correct person
  - Validated depth consistency
State machine simply executes commands immediately without redundant timing.
"""

import rclpy
import time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from interfaces.msg import HumanState, SystemState

from .state_manager import RobotStateMachine, StateConfig, TrackingTarget, RobotState


class RobotStateMachineNode(Node):
    
    def __init__(self):
        super().__init__('robot_state_machine')
        
        self.declare_parameter('tracking_timeout', 0.5)       
        self.declare_parameter('retracking_timeout', 5.0)     
        self.declare_parameter('velocity_filter_alpha', 0.3)  
        
        tracking_timeout = self.get_parameter('tracking_timeout').value
        retrack_timeout = self.get_parameter('retracking_timeout').value
        vel_alpha = self.get_parameter('velocity_filter_alpha').value
        
        state_config = StateConfig(
            tracking_timeout=tracking_timeout,
            retracking_timeout=retrack_timeout,
            velocity_filter_alpha=vel_alpha,
        )
        self.state_machine = RobotStateMachine(state_config)
        
        self.vel_alpha = vel_alpha
        self.last_vx = 0.0
        self.last_vy = 0.0
        
        self.state_machine.register_state_callback(RobotState.START, self._on_start)
        self.state_machine.register_state_callback(RobotState.TRACKING, self._on_tracking)
        self.state_machine.register_state_callback(RobotState.RE_TRACKING, self._on_retracking)
        self.state_machine.register_state_callback(RobotState.STOP, self._on_stop)
        self.state_machine.register_state_callback(RobotState.IDLE, self._on_idle)
        
        qos_best = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.state_pub = self.create_publisher(String, '/system/robot_status', 10)
        self.system_state_pub = self.create_publisher(SystemState, '/system_state', 10) 
        self.predicted_target_pub = self.create_publisher(HumanState, '/tracking/predicted_target', qos_best)
        
        self.create_subscription(HumanState, '/tracking/main_person', self._person_callback, qos_best)
        self.create_subscription(String, '/yolo/gesture_command', self._gesture_callback, 10)
        
        self.create_timer(0.05, self._control_loop)
        
        self.last_detection_time = time.time()
        self.last_person_state = None
        
        self.get_logger().info('✅ State Machine Node initialized (INSTANT RESPONSE MODE)')
    
    def _smooth_velocity(self, vx_new, vy_new):
        self.last_vx = self.vel_alpha * vx_new + (1 - self.vel_alpha) * self.last_vx
        self.last_vy = self.vel_alpha * vy_new + (1 - self.vel_alpha) * self.last_vy
        return self.last_vx, self.last_vy
    
    def _person_callback(self, msg: HumanState):
        current_time = time.time()
        self.last_detection_time = current_time
        self.last_person_state = msg
        
        vx_smooth, vy_smooth = self._smooth_velocity(msg.vx, msg.vy)
        
        target = TrackingTarget(
            id=0,  
            px=msg.px,      
            py=msg.py,      
            vx=vx_smooth,   
            vy=vy_smooth,   
            last_detection_time=current_time,
            last_position_update_time=current_time,
        )
        
        self.state_machine.update_tracked_target(target, current_time)
    
    def _gesture_callback(self, msg: String):
        """
        Nhận lệnh từ YOLO. Chỉ cần YOLO gửi, State Machine sẽ thực thi ngay lập tức.
        """
        try:
            gesture_str = msg.data.split('(')[0].strip()
            gesture_class = 0 if gesture_str == "START" else 1
            
            # Gửi thẳng vào state_machine mà không cần dummy bbox
            self.state_machine.handle_gesture(gesture_class, time.time())
            
            self.get_logger().info(f'⚡ EXECUTE GESTURE COMMAND: {gesture_str}')
        except Exception as e:
            self.get_logger().warning(f'Gesture parse error: {e}', throttle_duration_sec=2.0)
    
    def _control_loop(self):
        current_time = time.time()
        self.state_machine.update_no_detection(current_time)
        
        state_msg = String()
        state_msg.data = self.state_machine.current_state.value
        self.state_pub.publish(state_msg)
        
        # Publish system state using SystemState interface
        system_state_msg = SystemState()
        system_state_msg.state = self.state_machine.current_state.value
        self.system_state_pub.publish(system_state_msg)
        
        if self.state_machine.current_state == RobotState.RE_TRACKING and self.last_person_state:
            self._publish_prediction()
    
    def _publish_prediction(self):
        if self.last_person_state is None:
            return
            
        dt_pred = 0.5
        msg = HumanState()
        msg.px = self.last_person_state.px + self.last_vx * dt_pred
        msg.py = self.last_person_state.py + self.last_vy * dt_pred
        msg.vx = self.last_vx
        msg.vy = self.last_vy
        msg.radius = self.last_person_state.radius
        msg.trajectory = []
        
        self.predicted_target_pub.publish(msg)
    
    def _on_start(self):
        self.get_logger().info('🎯 START: Gesture Confirmed, Lock Target!')
    
    def _on_tracking(self):
        self.get_logger().info('🚀 TRACKING: Following Target')
    
    def _on_retracking(self):
        self.get_logger().warn('⚠️ RE_TRACKING: Predicting target (lost for <5s)')
    
    def _on_stop(self):
        self.get_logger().info('⏹️ STOP: Gesture Confirmed, Tracking Stopped!')
    
    def _on_idle(self):
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
