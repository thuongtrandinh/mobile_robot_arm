#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
import math

class OdomCalculator(Node):
    def __init__(self):
        super().__init__('odom_calculator')
        
        self.declare_parameter('wheel_radius', 0.05)
        self.declare_parameter('wheel_separation', 0.46) 
        
        self.radius = self.get_parameter('wheel_radius').value
        self.separation = self.get_parameter('wheel_separation').value

        self.odom_pub = self.create_publisher(Odometry, '/diff_cont/odom', 10)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)

        self.x = self.y = self.th = 0.0
        self.last_left_pos = self.last_right_pos = None
        self.last_time = self.get_clock().now()

    def joint_callback(self, msg):
        try:
            # Resolve wheel indices by joint name to avoid order-dependent odometry errors.
            l_idx = msg.name.index('left_wheel_joint')
            r_idx = msg.name.index('right_wheel_joint')
            l_pos = msg.position[l_idx]
            r_pos = msg.position[r_idx]
        except (ValueError, IndexError):
            return

        # --- ĐỒNG BỘ THỜI GIAN ---
        now = self.get_clock().now()
        
        if self.last_left_pos is None:
            self.last_left_pos, self.last_right_pos, self.last_time = l_pos, r_pos, now
            return

        dt = (now - self.last_time).nanoseconds / 1e9
        if dt <= 0: return

        d_l = (l_pos - self.last_left_pos) * self.radius
        d_r = (r_pos - self.last_right_pos) * self.radius
        
        d_c = (d_r + d_l) / 2.0
        # Đảm bảo hướng quay khớp với thực tế robot quay trái là dương
        d_th = (d_l - d_r) / self.separation

        self.x += d_c * math.cos(self.th + d_th/2.0)
        self.y += d_c * math.sin(self.th + d_th/2.0)
        self.th += d_th

        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_footprint'
        
        # Pose
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = math.sin(self.th / 2.0)
        odom.pose.pose.orientation.w = math.cos(self.th / 2.0)
        
        # --- THÊM HIỆP PHƯƠNG SAI CHO POSE ---
        # Ưu tiên tin tưởng vị trí (0.01) và hướng (0.01)
        odom.pose.covariance = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.01, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 1e6, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 1e6, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 1e6, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.01]
        
        # Twist
        odom.twist.twist.linear.x = d_c / dt
        odom.twist.twist.angular.z = d_th / dt

        # --- THÊM HIỆP PHƯƠNG SAI CHO TWIST ---
        # Cực kỳ quan trọng để EKF dung hợp vận tốc xoay với IMU
        odom.twist.covariance = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.01, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 1e6, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 1e6, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 1e6, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.01]

        self.odom_pub.publish(odom)
        self.last_left_pos, self.last_right_pos, self.last_time = l_pos, r_pos, now

def main():
    rclpy.init()
    rclpy.spin(OdomCalculator())
    rclpy.shutdown()

if __name__ == '__main__':
    main()