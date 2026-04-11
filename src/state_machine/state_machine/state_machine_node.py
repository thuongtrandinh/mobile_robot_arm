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
import math
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from interfaces.msg import HumanState, SystemState

from .state_manager import RobotStateMachine, StateConfig, TrackingTarget, RobotState

# Import goal filter for trend-based control
try:
    from yolo.algorithms.velocity_filter import GoalPoseFilter
except ImportError:
    # Fallback: define simple GoalPoseFilter locally if yolo package not available
    class GoalPoseFilter:
        def __init__(self, safe_distance=1.5, alpha_follow=0.25, alpha_brake=0.15, creep_thresh=0.08):
            self.safe_distance = safe_distance
            self.alpha_follow = alpha_follow
            self.alpha_brake = alpha_brake
            self.creep_thresh = creep_thresh
            self.last_goal_x = 0.0
            self.last_goal_y = 0.0
            self.prev_distance = None
            
        def process(self, ekf_x, ekf_y):
            distance = math.hypot(ekf_x, ekf_y)
            if self.prev_distance is None:
                self.prev_distance = distance
            delta_dist = distance - self.prev_distance
            self.prev_distance = distance
            is_approaching = delta_dist < -0.01
            
            if distance > self.safe_distance:
                ratio = (distance - self.safe_distance) / distance
                raw_goal_x = ekf_x * ratio
                raw_goal_y = ekf_y * ratio
                if is_approaching:
                    damping = 0.3
                    raw_goal_x *= damping
                    raw_goal_y *= damping
            else:
                raw_goal_x = 0.0
                raw_goal_y = 0.0
            
            current_mag = math.hypot(self.last_goal_x, self.last_goal_y)
            raw_mag = math.hypot(raw_goal_x, raw_goal_y)
            alpha = self.alpha_brake if raw_mag < current_mag else self.alpha_follow
            
            self.last_goal_x = alpha * raw_goal_x + (1 - alpha) * self.last_goal_x
            self.last_goal_y = alpha * raw_goal_y + (1 - alpha) * self.last_goal_y
            
            if raw_mag == 0.0 and math.hypot(self.last_goal_x, self.last_goal_y) < self.creep_thresh:
                self.last_goal_x = 0.0
                self.last_goal_y = 0.0
                
            return float(self.last_goal_x), float(self.last_goal_y)
        
        def reset(self):
            self.last_goal_x = 0.0
            self.last_goal_y = 0.0
            self.prev_distance = None


class RobotStateMachineNode(Node):
    
    def __init__(self):
        super().__init__('robot_state_machine')
        
        self.declare_parameter('tracking_timeout', 0.5)       
        self.declare_parameter('retracking_timeout', 10.0)    # [SỬA] Tăng từ 5.0 → 10.0 để đồng bộ với tracking_node.py
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
        
        # ===== GOAL POSE FILTER WITH TREND ANALYSIS =====
        # Automatically detects when person approaching and applies damping
        self.goal_filter = GoalPoseFilter(
            safe_distance=1.5,
            alpha_follow=0.25,   # Faster response when person moving away
            alpha_brake=0.15,    # Slower smooth braking when person approaching
            creep_thresh=0.08    # Anti-creep threshold
        )
        
        self.state_machine.register_state_callback(RobotState.START, self._on_start)
        self.state_machine.register_state_callback(RobotState.TRACKING, self._on_tracking)
        self.state_machine.register_state_callback(RobotState.RE_TRACKING, self._on_retracking)
        self.state_machine.register_state_callback(RobotState.STOP, self._on_stop)
        self.state_machine.register_state_callback(RobotState.IDLE, self._on_idle)
        
        # =====================================================================
        # [ĐÃ ĐỒNG BỘ] Sử dụng QoS 10 (Reliable) cho TẤT CẢ các topic để khớp 
        # 100% với tracking_node.py (tránh rớt gói tin do lệch QoS)
        # =====================================================================
        self.state_pub = self.create_publisher(String, '/system/robot_status', 10)
        self.system_state_pub = self.create_publisher(SystemState, '/system_state', 10) 
        self.predicted_target_pub = self.create_publisher(HumanState, '/tracking/predicted_target', 10)
        
        # [THÊM MỚI] Gánh việc xuất goal_pose thay cho Tracking Node khi mất dấu
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
        self.create_subscription(HumanState, '/tracking/main_person', self._person_callback, 10)
        self.create_subscription(String, '/yolo/gesture_command', self._gesture_callback, 10)
        
        self.create_timer(0.05, self._control_loop)
        
        self.last_detection_time = time.time()
        self.last_person_state = None
        
        self.get_logger().info('✅ State Machine Node initialized (SYNCED & INSTANT MODE)')
    
    def _smooth_velocity(self, vx_new, vy_new):
        self.last_vx = self.vel_alpha * vx_new + (1 - self.vel_alpha) * self.last_vx
        self.last_vy = self.vel_alpha * vy_new + (1 - self.vel_alpha) * self.last_vy
        return self.last_vx, self.last_vy
    
    def _person_callback(self, msg: HumanState):
        current_time = time.time()
        self.last_detection_time = current_time
        
        # Lưu lại trạng thái thật cuối cùng để làm gốc dự đoán
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
            
        dt_pred = 0.05  # Tiến tới 0.05s mỗi chu kỳ (20Hz) để xe chạy mượt như thật
        
        # [ĐÃ TỐI ƯU] Trượt tọa độ lên từ từ thay vì cộng dồn 1 cục
        pred_px = self.last_person_state.px + self.last_vx * dt_pred
        pred_py = self.last_person_state.py + self.last_vy * dt_pred
        
        # Cập nhật lại gốc dự đoán (Ghost Target di chuyển liên tục)
        self.last_person_state.px = pred_px
        self.last_person_state.py = pred_py
        
        # 1. Publish HumanState ảo (Cho Rviz / Debug)
        msg = HumanState()
        msg.px = pred_px
        msg.py = pred_py
        msg.vx = self.last_vx
        msg.vy = self.last_vy
        msg.radius = self.last_person_state.radius
        msg.trajectory = []
        self.predicted_target_pub.publish(msg)
        
        # 2. [THÊM MỚI] Publish Goal Pose để giữ xe lướt đi theo vận tốc cuối cùng
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = "camera_link"
        
        # [THÊM MỚI] Use GoalPoseFilter with trend analysis
        # Automatically detects if person approaching and applies damping
        goal_x, goal_y = self.goal_filter.process(pred_px, pred_py)

        goal_msg.pose.position.x = float(goal_x)
        goal_msg.pose.position.y = float(goal_y)
        goal_msg.pose.position.z = 0.0
        
        yaw = math.atan2(pred_py, pred_px)
        goal_msg.pose.orientation.x = 0.0
        goal_msg.pose.orientation.y = 0.0
        goal_msg.pose.orientation.z = float(math.sin(yaw / 2.0))
        goal_msg.pose.orientation.w = float(math.cos(yaw / 2.0))

        self.goal_pub.publish(goal_msg)
    
    def _on_start(self):
        self.get_logger().info('🎯 START: Gesture Confirmed, Lock Target!')
    
    def _on_tracking(self):
        self.get_logger().info('🚀 TRACKING: Following Target')
    
    def _on_retracking(self):
        self.get_logger().warn('⚠️ Re-TRACKING: Target lost! Predicting and moving robot for 5s...')
    
    def _on_stop(self):
        self.get_logger().info('⏹️ STOP: Tracking Stopped!')
    
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
