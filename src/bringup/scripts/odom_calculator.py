#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
import math

class OdomCalculator(Node):
    def __init__(self):
        # Bạn có thể đặt tên node là 'diff_cont' trong launch file để nó tự đọc thông số từ ros2_control.yaml
        super().__init__('odom_calculator')
        
        # Thông số mặc định (sẽ bị ghi đè nếu bạn truyền file yaml vào)
        self.declare_parameter('wheel_radius', 0.046)
        self.declare_parameter('wheel_separation', 0.5) 
        
        self.radius = self.get_parameter('wheel_radius').value
        self.separation = self.get_parameter('wheel_separation').value

        # TRỌNG TÂM Ở ĐÂY: Phát đúng topic mà EKF trong package localization đang lắng nghe
        self.odom_pub = self.create_publisher(Odometry, '/diff_cont/odom', 10)
        
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)

        # Trạng thái ban đầu
        self.x = self.y = self.th = 0.0
        self.last_left_pos = self.last_right_pos = None
        self.last_time = self.get_clock().now()

    def joint_callback(self, msg):
        try:
            l_idx = msg.name.index('left_wheel_joint')
            r_idx = msg.name.index('right_wheel_joint')
            l_pos, r_pos = msg.position[l_idx], msg.position[r_idx]
        except (ValueError, IndexError): 
            return

        now = self.get_clock().now()
        if self.last_left_pos is None:
            self.last_left_pos, self.last_right_pos, self.last_time = l_pos, r_pos, now
            return

        dt = (now - self.last_time).nanoseconds / 1e9
        if dt <= 0:
            return

        # Tính toán delta
        d_l = (l_pos - self.last_left_pos) * self.radius
        d_r = (r_pos - self.last_right_pos) * self.radius
        
        d_c = (d_r + d_l) / 2.0
        d_th = (d_r - d_l) / self.separation

        # Cập nhật vị trí X, Y, Yaw
        self.x += d_c * math.cos(self.th + d_th/2.0)
        self.y += d_c * math.sin(self.th + d_th/2.0)
        self.th += d_th

        # Đóng gói Message Odometry
        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_footprint'
        
        # Gán Pose (Vị trí và Hướng)
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = math.sin(self.th / 2.0)
        odom.pose.pose.orientation.w = math.cos(self.th / 2.0)
        
        # Gán Twist (Vận tốc) - EKF rất cần dữ liệu này để kết hợp với IMU
        odom.twist.twist.linear.x = d_c / dt
        odom.twist.twist.angular.z = d_th / dt

        # Publish ra ngoài
        self.odom_pub.publish(odom)
        
        # Lưu lại trạng thái cho vòng lặp sau
        self.last_left_pos, self.last_right_pos, self.last_time = l_pos, r_pos, now

def main():
    rclpy.init()
    rclpy.spin(OdomCalculator())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
