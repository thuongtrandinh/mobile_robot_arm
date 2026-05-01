#!/usr/bin/python3
"""
Dynamic Window Approach (DWA) Local Planner Wrapper for Nav2
This planner uses the DWB (Dynamic Window Behavior) controller from Nav2
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float64
from tf2_ros import Buffer, TransformListener, TransformException
import math
from collections import deque


class DWALocalPlanner(Node):
    """
    Dynamic Window Approach Local Planner
    Generates velocity commands based on local costmap and laser scans
    """

    def __init__(self):
        super().__init__("dwa_local_planner")

        # Parameters
        self.declare_parameter("max_linear_vel", 0.8)
        self.declare_parameter("max_angular_vel", 1.5)
        self.declare_parameter("min_linear_vel", 0.1)
        self.declare_parameter("prediction_time", 2.0)
        self.declare_parameter("linear_sim_granularity", 0.05)
        self.declare_parameter("angular_sim_granularity", 0.05)
        self.declare_parameter("obstacle_range", 3.0)
        self.declare_parameter("inflation_radius", 0.25)
        self.declare_parameter("goal_tolerance", 0.3)
        
        self.max_linear_vel = self.get_parameter("max_linear_vel").value
        self.max_angular_vel = self.get_parameter("max_angular_vel").value
        self.min_linear_vel = self.get_parameter("min_linear_vel").value
        self.prediction_time = self.get_parameter("prediction_time").value
        self.linear_sim_granularity = self.get_parameter("linear_sim_granularity").value
        self.angular_sim_granularity = self.get_parameter("angular_sim_granularity").value
        self.obstacle_range = self.get_parameter("obstacle_range").value
        self.inflation_radius = self.get_parameter("inflation_radius").value
        self.goal_tolerance = self.get_parameter("goal_tolerance").value

        # TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # State
        self.current_pose = None
        self.current_twist = None
        self.global_path = None
        self.costmap = None
        self.scan = None

        # Subscribers
        self.pose_sub = self.create_subscription(Odometry, "odom", self.odom_callback, 10)
        self.path_sub = self.create_subscription(Path, "global_path", self.path_callback, 10)
        self.costmap_sub = self.create_subscription(OccupancyGrid, "local_costmap/costmap", self.costmap_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, "scan", self.scan_callback, 10)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.cost_pub = self.create_publisher(Float64, "dwa_cost", 10)

        self.get_logger().info("DWA Local Planner initialized")

    def odom_callback(self, msg: Odometry):
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

    def path_callback(self, msg: Path):
        self.global_path = msg.poses

    def costmap_callback(self, msg: OccupancyGrid):
        self.costmap = msg

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

            if 0.1 < dist < 1.0 and dist < min_dist:
                min_dist = dist
                local_goal = pose.pose

        return local_goal or (self.global_path[-1].pose if self.global_path else None)

    def check_collision(self, linear_vel: float, angular_vel: float) -> float:
        """
        Check if a velocity command would result in collision
        Returns cost (0 = no collision, 1 = collision)
        """
        if not self.scan:
            return 0.0

        # Get closest obstacle in front
        min_range = float('inf')
        front_angle_range = 45  # degrees

        for i, range_val in enumerate(self.scan.ranges):
            if self.scan.range_min < range_val < self.scan.range_max:
                angle = self.scan.angle_min + i * self.scan.angle_increment
                # Convert to degrees and check if in front
                angle_deg = math.degrees(angle)
                if abs(angle_deg) < front_angle_range:
                    min_range = min(min_range, range_val)

        if min_range < self.inflation_radius:
            return 1.0
        elif min_range < self.obstacle_range:
            return (self.obstacle_range - min_range) / (self.obstacle_range - self.inflation_radius)

        return 0.0

    def calculate_heading_cost(self, linear_vel: float, angular_vel: float) -> float:
        """Calculate cost based on heading difference to goal"""
        local_goal = self.get_local_goal()
        if not local_goal or not self.current_pose:
            return 0.0

        dx = local_goal.position.x - self.current_pose.position.x
        dy = local_goal.position.y - self.current_pose.position.y
        goal_angle = math.atan2(dy, dx)

        # Get current heading
        from tf_transformations import euler_from_quaternion
        _, _, current_angle = euler_from_quaternion([
            self.current_pose.orientation.x,
            self.current_pose.orientation.y,
            self.current_pose.orientation.z,
            self.current_pose.orientation.w
        ])

        angle_diff = abs(goal_angle - current_angle)
        # Normalize to [-pi, pi]
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

        return abs(angle_diff) / math.pi

    def calculate_distance_cost(self, linear_vel: float, angular_vel: float) -> float:
        """Calculate cost based on predicted distance to goal"""
        local_goal = self.get_local_goal()
        if not local_goal or not self.current_pose:
            return 0.0

        dx = local_goal.position.x - self.current_pose.position.x
        dy = local_goal.position.y - self.current_pose.position.y
        distance = math.sqrt(dx**2 + dy**2)

        if distance < self.goal_tolerance:
            return 0.0

        return min(1.0, distance / 2.0)

    def plan(self):
        """Main planning loop"""
        if not self.current_pose or not self.current_twist:
            return

        best_cost = float('inf')
        best_cmd = Twist()

        # Sample velocity space
        for linear_vel in np.arange(self.min_linear_vel, self.max_linear_vel + 0.01, self.linear_sim_granularity):
            for angular_vel in np.arange(-self.max_angular_vel, self.max_angular_vel + 0.01, self.angular_sim_granularity):
                # Calculate costs
                collision_cost = self.check_collision(linear_vel, angular_vel)
                heading_cost = self.calculate_heading_cost(linear_vel, angular_vel)
                distance_cost = self.calculate_distance_cost(linear_vel, angular_vel)
                
                # Weighted sum
                total_cost = (
                    5.0 * collision_cost +      # High penalty for collision
                    1.0 * heading_cost +        # Prefer aligned heading
                    0.5 * distance_cost         # Prefer moving closer to goal
                )

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_cmd.linear.x = linear_vel
                    best_cmd.angular.z = angular_vel

        # Publish command and cost
        self.cmd_vel_pub.publish(best_cmd)
        cost_msg = Float64()
        cost_msg.data = best_cost
        self.cost_pub.publish(cost_msg)

        self.get_logger().debug(f"DWA: v={best_cmd.linear.x:.2f}, w={best_cmd.angular.z:.2f}, cost={best_cost:.3f}")


def main(args=None):
    rclpy.init(args=args)
    node = DWALocalPlanner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
