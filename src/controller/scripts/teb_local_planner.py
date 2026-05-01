#!/usr/bin/python3
"""
Timed Elastic Band (TEB) Local Planner Wrapper
Simplified trajectory optimization approach
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float64, ColorRGBA
import math
from collections import deque
from scipy.optimize import minimize
from tf_transformations import euler_from_quaternion, quaternion_from_euler


class TEBLocalPlanner(Node):
    """
    Timed Elastic Band Local Planner
    Generates optimal trajectories considering kinematic constraints
    """

    def __init__(self):
        super().__init__("teb_local_planner")

        # Parameters
        self.declare_parameter("max_linear_vel", 0.8)
        self.declare_parameter("max_angular_vel", 1.5)
        self.declare_parameter("max_accel_lin", 1.0)
        self.declare_parameter("max_accel_ang", 2.0)
        self.declare_parameter("trajectory_dt", 0.1)
        self.declare_parameter("num_trajectory_points", 20)
        self.declare_parameter("obstacle_range", 3.0)
        self.declare_parameter("inflation_radius", 0.25)
        self.declare_parameter("goal_tolerance", 0.3)
        self.declare_parameter("weight_obstacle", 10.0)
        self.declare_parameter("weight_trajectory", 1.0)
        self.declare_parameter("weight_time", 0.1)
        
        self.max_linear_vel = self.get_parameter("max_linear_vel").value
        self.max_angular_vel = self.get_parameter("max_angular_vel").value
        self.max_accel_lin = self.get_parameter("max_accel_lin").value
        self.max_accel_ang = self.get_parameter("max_accel_ang").value
        self.trajectory_dt = self.get_parameter("trajectory_dt").value
        self.num_trajectory_points = self.get_parameter("num_trajectory_points").value
        self.obstacle_range = self.get_parameter("obstacle_range").value
        self.inflation_radius = self.get_parameter("inflation_radius").value
        self.goal_tolerance = self.get_parameter("goal_tolerance").value
        self.weight_obstacle = self.get_parameter("weight_obstacle").value
        self.weight_trajectory = self.get_parameter("weight_trajectory").value
        self.weight_time = self.get_parameter("weight_time").value

        # State
        self.current_pose = None
        self.current_twist = None
        self.global_path = None
        self.scan = None

        # Subscribers
        self.pose_sub = self.create_subscription(Odometry, "odom", self.odom_callback, 10)
        self.path_sub = self.create_subscription(Path, "global_path", self.path_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, "scan", self.scan_callback, 10)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.trajectory_pub = self.create_publisher(MarkerArray, "teb_trajectory", 10)
        self.cost_pub = self.create_publisher(Float64, "teb_cost", 10)

        self.last_command = Twist()
        self.get_logger().info("TEB Local Planner initialized")

    def odom_callback(self, msg: Odometry):
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

    def path_callback(self, msg: Path):
        self.global_path = msg.poses

    def scan_callback(self, msg: LaserScan):
        self.scan = msg

    def get_local_goal(self):
        """Get the next goal point from the global path"""
        if not self.global_path or not self.current_pose:
            return None

        # Find the nearest point ahead on the path
        min_dist = float('inf')
        local_goal = None

        for pose in self.global_path:
            dx = pose.pose.position.x - self.current_pose.position.x
            dy = pose.pose.position.y - self.current_pose.position.y
            dist = math.sqrt(dx**2 + dy**2)

            if 0.1 < dist < 2.0 and dist < min_dist:
                min_dist = dist
                local_goal = pose.pose

        return local_goal or (self.global_path[-1].pose if self.global_path else None)

    def simulate_trajectory(self, linear_vel: float, angular_vel: float, duration: float = None):
        """
        Simulate a trajectory given velocity commands
        Returns list of (x, y, theta) points
        """
        if duration is None:
            duration = self.trajectory_dt * self.num_trajectory_points

        trajectory = []
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        _, _, theta = euler_from_quaternion([
            self.current_pose.orientation.x,
            self.current_pose.orientation.y,
            self.current_pose.orientation.z,
            self.current_pose.orientation.w
        ])

        # Simulate trajectory
        for _ in range(self.num_trajectory_points):
            # Kinematic model
            x += linear_vel * math.cos(theta) * self.trajectory_dt
            y += linear_vel * math.sin(theta) * self.trajectory_dt
            theta += angular_vel * self.trajectory_dt

            trajectory.append((x, y, theta))

        return trajectory

    def collision_cost(self, trajectory: list) -> float:
        """Calculate collision cost for a trajectory"""
        if not self.scan:
            return 0.0

        collision_cost = 0.0

        # Check each trajectory point
        for x, y, _ in trajectory:
            min_range = float('inf')

            # Find closest obstacle point to this trajectory point
            for i, range_val in enumerate(self.scan.ranges):
                if self.scan.range_min < range_val < self.scan.range_max:
                    angle = self.scan.angle_min + i * self.scan.angle_increment
                    obs_x = x + range_val * math.cos(angle)
                    obs_y = y + range_val * math.sin(angle)

                    dist_to_obs = math.sqrt((x - obs_x)**2 + (y - obs_y)**2)
                    min_range = min(min_range, dist_to_obs)

            # Add cost based on proximity to obstacles
            if min_range < self.inflation_radius:
                collision_cost += 100.0  # High penalty
            elif min_range < self.obstacle_range:
                collision_cost += (self.obstacle_range - min_range) / (self.obstacle_range - self.inflation_radius)

        return collision_cost / len(trajectory)

    def trajectory_cost(self, trajectory: list) -> float:
        """Calculate cost for deviation from global path"""
        if not self.global_path:
            return 0.0

        path_cost = 0.0

        for x, y, _ in trajectory:
            # Find closest point on global path
            min_dist = float('inf')

            for pose in self.global_path:
                dx = pose.pose.position.x - x
                dy = pose.pose.position.y - y
                dist = math.sqrt(dx**2 + dy**2)
                min_dist = min(min_dist, dist)

            path_cost += min_dist

        return path_cost / len(trajectory)

    def time_cost(self, linear_vel: float) -> float:
        """Prefer faster trajectories (but penalize high velocities)"""
        return -linear_vel / (self.max_linear_vel + 0.01)

    def evaluate_trajectory(self, linear_vel: float, angular_vel: float) -> tuple:
        """Evaluate a velocity command and return (cost, trajectory)"""
        trajectory = self.simulate_trajectory(linear_vel, angular_vel)

        col_cost = self.collision_cost(trajectory)
        traj_cost = self.trajectory_cost(trajectory)
        time_cost = self.time_cost(linear_vel)

        total_cost = (
            self.weight_obstacle * col_cost +
            self.weight_trajectory * traj_cost +
            self.weight_time * time_cost
        )

        return total_cost, trajectory

    def optimize_trajectory(self):
        """Optimize velocity commands using trajectory optimization"""
        if not self.current_pose or not self.current_twist or not self.global_path:
            return

        best_cost = float('inf')
        best_cmd = Twist()
        best_trajectory = []

        # Sample velocity space more intelligently
        v_samples = np.linspace(0, self.max_linear_vel, 8)
        w_samples = np.linspace(-self.max_angular_vel, self.max_angular_vel, 8)

        for linear_vel in v_samples:
            for angular_vel in w_samples:
                cost, trajectory = self.evaluate_trajectory(linear_vel, angular_vel)

                if cost < best_cost:
                    best_cost = cost
                    best_cmd.linear.x = linear_vel
                    best_cmd.angular.z = angular_vel
                    best_trajectory = trajectory

        # Apply acceleration limits
        max_accel_v = min(
            abs(best_cmd.linear.x - self.last_command.linear.x) / 0.1,
            self.max_accel_lin
        )
        max_accel_w = min(
            abs(best_cmd.angular.z - self.last_command.angular.z) / 0.1,
            self.max_accel_ang
        )

        self.last_command = best_cmd

        # Publish command
        self.cmd_vel_pub.publish(best_cmd)

        # Publish cost
        cost_msg = Float64()
        cost_msg.data = best_cost
        self.cost_pub.publish(cost_msg)

        # Publish trajectory visualization
        self.publish_trajectory_markers(best_trajectory)

        self.get_logger().debug(f"TEB: v={best_cmd.linear.x:.2f}, w={best_cmd.angular.z:.2f}, cost={best_cost:.3f}")

    def publish_trajectory_markers(self, trajectory: list):
        """Publish trajectory as markers for visualization"""
        markers = MarkerArray()

        for i, (x, y, theta) in enumerate(trajectory):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.7)

            markers.markers.append(marker)

        self.trajectory_pub.publish(markers)

    def run(self):
        """Main loop"""
        timer = self.create_timer(0.1, self.optimize_trajectory)


def main(args=None):
    rclpy.init(args=args)
    node = TEBLocalPlanner()
    node.run()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
