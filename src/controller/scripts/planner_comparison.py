#!/usr/bin/python3
"""
Local Planner Comparison Framework
Compares performance of DWA, TEB, and custom AMPCC algorithms
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64, String
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener
import csv
import os
from datetime import datetime
import math


class PlannerComparison(Node):
    """
    Compares multiple local planners and logs their performance metrics
    """

    def __init__(self):
        super().__init__("planner_comparison")

        # Parameters
        self.declare_parameter("test_duration", 60.0)  # seconds
        self.declare_parameter("active_planner", "rl_mppi")  # dwa, teb, rl_mppi
        self.declare_parameter("data_dir", "./planner_comparison_data")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("velocity_odom_topic", "/odometry/filtered")

        self.test_duration = self.get_parameter("test_duration").value
        self.active_planner = self.get_parameter("active_planner").value
        self.data_dir = self.get_parameter("data_dir").value
        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.velocity_odom_topic = self.get_parameter("velocity_odom_topic").value

        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)

        # State tracking
        self.current_pose = None
        self.current_twist = None
        self.global_path = None
        self.start_time = None
        self.start_pose = None
        self.goal_pose = None
        self.goal_reached = False
        self.recording_started = False  # Bắt đầu ghi khi nhận goal_pose
        self.goal_received_time = None  # Thời điểm nhận goal
        
        # Metrics
        self.metrics = {
            'total_distance': 0.0,
            'path_length': 0.0,
            'collision_count': 0,
            'computation_times': [],
            'costs': [],
            'velocities': [],
            'angular_velocities': [],
            'trajectory_points': [],
            'time_to_goal': None,
            'efficiency': 0.0,
            'success': False,
            'num_oscillations': 0,
            'avg_computation_time': 0.0,
        }

        # Timeseries data collection
        self.timeseries_data = []
        self.timeseries_start_time = None
        self.current_clearance = 10.0
        self.current_cmd_v = 0.0
        self.current_cmd_w = 0.0
        self.current_v_linear = 0.0
        self.current_v_angular = 0.0
        self.clearance_history = []  # Lưu lịch sử khoảng cách an toàn
        self.slam_jump_count = 0

        # TF pose tracking: position metrics use map -> base_link, not odom.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers
        self.odom_sub = None
        if self.velocity_odom_topic:
            self.odom_sub = self.create_subscription(
                Odometry,
                self.velocity_odom_topic,
                self.odom_velocity_callback,
                10,
            )
        self.path_sub = self.create_subscription(Path, "global_path", self.path_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, "scan", self.scan_callback, 10)
        self.dwa_cost_sub = self.create_subscription(Float64, "dwa_cost", self.dwa_cost_callback, 10)
        self.teb_cost_sub = self.create_subscription(Float64, "teb_cost", self.teb_cost_callback, 10)
        self.cmd_vel_sub = self.create_subscription(Twist, "cmd_vel", self.cmd_vel_callback, 10)
        self.cmd_vel_unstamped_sub = self.create_subscription(Twist, "/diff_cont/cmd_vel_unstamped", self.cmd_callback, 10)

        # Publishers
        self.stats_pub = self.create_publisher(String, "planner_stats", 10)
        self.comparison_pub = self.create_publisher(MarkerArray, "planner_comparison", 10)

        # Timers
        self.metrics_timer = self.create_timer(0.1, self.update_metrics)
        self.pose_timer = self.create_timer(0.05, self.pose_update_callback)

        self.get_logger().info(f"Planner Comparison Node initialized for {self.active_planner}")

    def odom_velocity_callback(self, msg: Odometry):
        """Extract raw encoder/odometry velocity only; pose comes from TF."""
        self.current_v_linear = msg.twist.twist.linear.x
        self.current_v_angular = msg.twist.twist.angular.z
        self.current_twist = msg.twist.twist

    def pose_update_callback(self):
        """Sample one synchronized row: TF pose + latest cmd + latest encoder + clearance."""
        if not self.recording_started:
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(),
            )
        except Exception:
            return

        current_x = transform.transform.translation.x
        current_y = transform.transform.translation.y

        if self.current_pose is None:
            self.current_pose = Pose()
        self.current_pose.position.x = current_x
        self.current_pose.position.y = current_y
        self.current_pose.orientation = transform.transform.rotation

        if self.metrics['trajectory_points']:
            prev_x, prev_y = self.metrics['trajectory_points'][-1]
            step_distance = math.hypot(current_x - prev_x, current_y - prev_y)

            # Keep distance robust to TF noise and SLAM loop-closure jumps.
            if 0.01 < step_distance <= 0.15:
                self.metrics['total_distance'] += step_distance
                self.metrics['trajectory_points'].append((current_x, current_y))
            elif step_distance > 0.15:
                self.slam_jump_count += 1
                self.get_logger().warn(
                    f"SLAM jump #{self.slam_jump_count}: {step_distance:.3f}m; distance not accumulated."
                )
                self.metrics['trajectory_points'].append((current_x, current_y))
        else:
            self.metrics['trajectory_points'].append((current_x, current_y))

        current_time = self.get_clock().now().nanoseconds / 1e9
        if self.timeseries_start_time is None:
            self.timeseries_start_time = current_time

        t = current_time - self.timeseries_start_time
        self.timeseries_data.append({
            'time': round(t, 3),
            'x': round(current_x, 3),
            'y': round(current_y, 3),
            'v_linear_cmd': round(self.current_cmd_v, 3),
            'v_linear_real': round(self.current_v_linear, 3),
            'v_angular_cmd': round(self.current_cmd_w, 3),
            'v_angular_real': round(self.current_v_angular, 3),
            'clearance': round(self.current_clearance, 3),
            'event': ''
        })

    def path_callback(self, msg: Path):
        self.global_path = msg.poses
        if msg.poses:
            self.goal_pose = msg.poses[-1].pose
            
            # Bắt đầu ghi dữ liệu từ khi nhận goal
            if not self.recording_started:
                self.recording_started = True
                self.goal_received_time = self.get_clock().now()
                self.start_time = self.goal_received_time
                self.start_pose = self.current_pose if self.current_pose else None
                self.timeseries_start_time = self.get_clock().now().nanoseconds / 1e9
                
                # Reset metrics
                self.metrics['total_distance'] = 0.0
                self.metrics['trajectory_points'] = []
                self.metrics['collision_count'] = 0
                self.metrics['velocities'] = []
                self.metrics['angular_velocities'] = []
                self.metrics['costs'] = []
                self.metrics['time_to_goal'] = None
                self.metrics['success'] = False
                self.metrics['efficiency'] = 0.0
                self.timeseries_data = []
                self.clearance_history = []
                self.goal_reached = False
                self.slam_jump_count = 0
                
                self.get_logger().info(
                    f"Goal received - Recording {self.active_planner} with TF pose "
                    f"({self.map_frame}->{self.base_frame}) and encoder velocity ({self.velocity_odom_topic})"
                )

    def scan_callback(self, msg: LaserScan):
        # Lấy ra các tia laser hợp lệ (không bị nhiễu, không phải inf/nan)
        valid_ranges = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
        
        # Nếu có tia hợp lệ thì tìm khoảng cách nhỏ nhất
        if valid_ranges:
            min_range = min(valid_ranges)
            # Loại bỏ các giá trị nhiễu vô cực
            if min_range < 10.0:
                self.current_clearance = min_range
                self.clearance_history.append(min_range)  # Lưu vào lịch sử
            
            # Nếu có vật cản cách robot dưới 0.3m -> Tính là 1 lần va chạm (hoặc rủi ro cao)
            if min_range < 0.3:
                self.metrics['collision_count'] += 1

    def dwa_cost_callback(self, msg: Float64):
        if self.active_planner == "dwa":
            self.metrics['costs'].append(msg.data)

    def teb_cost_callback(self, msg: Float64):
        if self.active_planner == "teb":
            self.metrics['costs'].append(msg.data)

    def cmd_vel_callback(self, msg: Twist):
        self.metrics['velocities'].append(msg.linear.x)
        self.metrics['angular_velocities'].append(msg.angular.z)

    def cmd_callback(self, msg: Twist):
        """Cập nhật liên tục lệnh vận tốc gửi xuống động cơ"""
        self.current_cmd_v = msg.linear.x
        self.current_cmd_w = msg.angular.z

    def update_metrics(self):
        """Update performance metrics"""
        if not self.recording_started or not self.current_pose or not self.goal_pose:
            return

        # Calculate distance to goal
        dx = self.goal_pose.position.x - self.current_pose.position.x
        dy = self.goal_pose.position.y - self.current_pose.position.y
        distance_to_goal = math.sqrt(dx**2 + dy**2)

        # Check if goal is reached
        if distance_to_goal < 0.3 and not self.goal_reached:
            self.goal_reached = True
            elapsed_time = (self.get_clock().now() - self.goal_received_time).nanoseconds / 1e9
            self.metrics['time_to_goal'] = elapsed_time
            self.metrics['success'] = True
            self.get_logger().info(f"Goal reached in {elapsed_time:.2f} seconds")
            self.save_metrics()  # Tự động lưu khi đạt goal

        # Calculate efficiency
        if self.global_path:
            global_path_length = 0.0
            for i in range(len(self.global_path) - 1):
                p1 = self.global_path[i].pose.position
                p2 = self.global_path[i + 1].pose.position
                global_path_length += math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

            self.metrics['path_length'] = global_path_length
            if global_path_length > 0:
                self.metrics['efficiency'] = global_path_length / max(self.metrics['total_distance'], 0.001)

        # Check elapsed time (từ khi nhận goal)
        elapsed_time = (self.get_clock().now() - self.goal_received_time).nanoseconds / 1e9
        if self.recording_started and elapsed_time > self.test_duration and not self.goal_reached:
            self.get_logger().info("Test duration exceeded")
            self.save_metrics()
            self.metrics_timer.cancel()

    def calculate_oscillations(self):
        """Calculate number of direction changes (oscillations)"""
        if len(self.metrics['velocities']) < 3:
            return 0

        oscillations = 0
        for i in range(1, len(self.metrics['velocities']) - 1):
            # Check if velocity direction changed
            if (self.metrics['velocities'][i-1] > 0 and self.metrics['velocities'][i] < 0) or \
               (self.metrics['velocities'][i-1] < 0 and self.metrics['velocities'][i] > 0):
                oscillations += 1

        self.metrics['num_oscillations'] = oscillations
        return oscillations

    def save_metrics(self):
        """Save metrics to CSV file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.data_dir, f"{self.active_planner}_metrics_{timestamp}.csv")

        # Calculate final metrics
        if self.metrics['costs']:
            avg_cost = np.mean(self.metrics['costs'])
        else:
            avg_cost = 0.0

        if self.metrics['computation_times']:
            self.metrics['avg_computation_time'] = np.mean(self.metrics['computation_times'])

        self.calculate_oscillations()

        # Calculate average clearance
        avg_clearance = np.mean(self.clearance_history) if self.clearance_history else 0.0

        # Save summary
        summary = {
            'planner': self.active_planner,
            'test_duration': self.test_duration,
            'total_distance': self.metrics['total_distance'],
            'path_length': self.metrics['path_length'],
            'efficiency': self.metrics['efficiency'],
            'time_to_goal': self.metrics['time_to_goal'] if self.metrics['time_to_goal'] else self.test_duration,
            'success': self.metrics['success'],
            'avg_clearance': avg_clearance,
            'collision_count': self.metrics['collision_count'],
            'oscillations': self.metrics['num_oscillations'],
            'slam_jumps': self.slam_jump_count,
            'avg_computation_time': self.metrics['avg_computation_time'],
            'timestamp': timestamp,
        }

        # Write to CSV
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary.keys())
            writer.writeheader()
            writer.writerow(summary)

        self.get_logger().info(f"Metrics saved to {filename}")

        # Save timeseries data
        ts_filename = os.path.join(self.data_dir, f"{self.active_planner}_timeseries_{timestamp}.csv")
        if self.timeseries_data:
            with open(ts_filename, 'w', newline='') as f:
                fieldnames = ['time', 'x', 'y', 'v_linear_cmd', 'v_linear_real',
                              'v_angular_cmd', 'v_angular_real', 'clearance', 'event']
                ts_writer = csv.DictWriter(f, fieldnames=fieldnames)
                ts_writer.writeheader()
                ts_writer.writerows(self.timeseries_data)
            self.get_logger().info(f"Timeseries data saved to {ts_filename} with {len(self.timeseries_data)} rows")

        # Publish summary (ĐÃ SỬA: An toàn khi tắt bằng Ctrl+C)
        stats_msg = String()
        stats_msg.data = str(summary)
        try:
            if rclpy.ok():  # Chỉ publish nếu ROS 2 context còn sống
                self.stats_pub.publish(stats_msg)
        except Exception:
            pass

    def get_summary_string(self) -> str:
        """Get a formatted summary of metrics"""
        summary = f"""
        ===== {self.active_planner.upper()} PLANNER METRICS =====
        Total Distance: {self.metrics['total_distance']:.2f} m
        Path Length: {self.metrics['path_length']:.2f} m
        Efficiency: {self.metrics['efficiency']:.2f}
        Time to Goal: {self.metrics['time_to_goal'] if self.metrics['time_to_goal'] else 'N/A'}
        Success: {self.metrics['success']}
        Collisions: {self.metrics['collision_count']}
        Oscillations: {self.metrics['num_oscillations']}
        """
        return summary


def main(args=None):
    rclpy.init(args=args)
    node = PlannerComparison()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_metrics()
        print(node.get_summary_string())
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
