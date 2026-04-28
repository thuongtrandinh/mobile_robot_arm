#!/usr/bin/python3
"""
Local Planner Comparison Framework
Compares performance of DWA, TEB, and custom AMPCC algorithms
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64, String
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
import time
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

        self.test_duration = self.get_parameter("test_duration").value
        self.active_planner = self.get_parameter("active_planner").value
        self.data_dir = self.get_parameter("data_dir").value

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

        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, "odom", self.odom_callback, 10)
        self.path_sub = self.create_subscription(Path, "global_path", self.path_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, "scan", self.scan_callback, 10)
        self.dwa_cost_sub = self.create_subscription(Float64, "dwa_cost", self.dwa_cost_callback, 10)
        self.teb_cost_sub = self.create_subscription(Float64, "teb_cost", self.teb_cost_callback, 10)
        self.cmd_vel_sub = self.create_subscription(Twist, "cmd_vel", self.cmd_vel_callback, 10)

        # Publishers
        self.stats_pub = self.create_publisher(String, "planner_stats", 10)
        self.comparison_pub = self.create_publisher(MarkerArray, "planner_comparison", 10)

        # Timer
        self.timer = self.create_timer(0.1, self.update_metrics)

        self.get_logger().info(f"Planner Comparison Node initialized for {self.active_planner}")

    def odom_callback(self, msg: Odometry):
        if not self.start_time:
            self.start_time = self.get_clock().now()
            self.start_pose = msg.pose.pose

        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

        # Update metrics
        if len(self.metrics['trajectory_points']) > 0:
            prev_x, prev_y = self.metrics['trajectory_points'][-1]
            dx = msg.pose.pose.position.x - prev_x
            dy = msg.pose.pose.position.y - prev_y
            self.metrics['total_distance'] += math.sqrt(dx**2 + dy**2)

        self.metrics['trajectory_points'].append((msg.pose.pose.position.x, msg.pose.pose.position.y))

    def path_callback(self, msg: Path):
        self.global_path = msg.poses
        if msg.poses:
            self.goal_pose = msg.poses[-1].pose

    def scan_callback(self, msg: LaserScan):
        # Lấy ra các tia laser hợp lệ (không bị nhiễu, không phải inf/nan)
        valid_ranges = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
        
        # Nếu có tia hợp lệ thì tìm khoảng cách nhỏ nhất
        if valid_ranges:
            min_range = min(valid_ranges)
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

    def update_metrics(self):
        """Update performance metrics"""
        if not self.current_pose or not self.goal_pose:
            return

        # Calculate distance to goal
        dx = self.goal_pose.position.x - self.current_pose.position.x
        dy = self.goal_pose.position.y - self.current_pose.position.y
        distance_to_goal = math.sqrt(dx**2 + dy**2)

        # Check if goal is reached
        if distance_to_goal < 0.3 and not self.goal_reached:
            self.goal_reached = True
            elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            self.metrics['time_to_goal'] = elapsed_time
            self.metrics['success'] = True
            self.get_logger().info(f"Goal reached in {elapsed_time:.2f} seconds")

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

        # Check elapsed time
        elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        if elapsed_time > self.test_duration and not self.goal_reached:
            self.get_logger().info("Test duration exceeded")
            self.save_metrics()
            self.timer.cancel()

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

        # Save summary
        summary = {
            'planner': self.active_planner,
            'test_duration': self.test_duration,
            'total_distance': self.metrics['total_distance'],
            'path_length': self.metrics['path_length'],
            'efficiency': self.metrics['efficiency'],
            'time_to_goal': self.metrics['time_to_goal'] if self.metrics['time_to_goal'] else self.test_duration,
            'success': self.metrics['success'],
            'collision_count': self.metrics['collision_count'],
            'num_oscillations': self.metrics['num_oscillations'],
            'avg_cost': avg_cost,
            'avg_computation_time': self.metrics['avg_computation_time'],
            'timestamp': timestamp,
        }

        # Write to CSV
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary.keys())
            writer.writeheader()
            writer.writerow(summary)

        self.get_logger().info(f"Metrics saved to {filename}")

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
