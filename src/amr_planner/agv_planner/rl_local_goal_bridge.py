#!/usr/bin/env python3
import importlib.util
import math
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import rclpy
import torch as th
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan

# AMR custom messages for ZED2 integration
from amr_interfaces.msg import HumanStateArray, ObstacleStateArray


class RLLocalGoalBridge(Node):
    def __init__(self):
        super().__init__("rl_local_goal_bridge")

        self.declare_parameter("halo_drl_dir", "/home/thuong/LVTN/amr_ws/HALO/drl_moudle")
        self.declare_parameter("config", "configs/mpc_rl.py")
        self.declare_parameter("model_dir", "/home/thuong/LVTN/amr_ws/HALO/drl_moudle/train_data/run_04")
        self.declare_parameter("action_dim", 9)
        self.declare_parameter("action_range", 2.25)
        self.declare_parameter("goal_tolerance", 0.25)
        self.declare_parameter("max_linear_speed", 0.6)
        self.declare_parameter("max_angular_speed", 1.2)
        self.declare_parameter("k_linear", 0.8)
        self.declare_parameter("k_angular", 1.2)
        self.declare_parameter("robot_radius", 0.25)
        self.declare_parameter("wheel_base", 0.46)
        self.declare_parameter("obstacle_radius", 0.2)
        self.declare_parameter("obstacle_sample_step", 16)
        self.declare_parameter("timer_period", 0.1)
        self.declare_parameter("odom_topic", "/odometry/filtered")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("cmd_topic", "/diff_cont/cmd_vel")
        self.declare_parameter("use_zed2_data", True)  # Enable ZED2 integration

        self.halo_drl_dir = self.get_parameter("halo_drl_dir").get_parameter_value().string_value
        self.config_path = self.get_parameter("config").get_parameter_value().string_value
        self.model_dir = self.get_parameter("model_dir").get_parameter_value().string_value

        self.action_dim = self.get_parameter("action_dim").get_parameter_value().integer_value
        self.action_range = self.get_parameter("action_range").get_parameter_value().double_value
        self.goal_tolerance = self.get_parameter("goal_tolerance").get_parameter_value().double_value
        self.max_linear_speed = self.get_parameter("max_linear_speed").get_parameter_value().double_value
        self.max_angular_speed = self.get_parameter("max_angular_speed").get_parameter_value().double_value
        self.k_linear = self.get_parameter("k_linear").get_parameter_value().double_value
        self.k_angular = self.get_parameter("k_angular").get_parameter_value().double_value
        self.robot_radius = self.get_parameter("robot_radius").get_parameter_value().double_value
        self.wheel_base = self.get_parameter("wheel_base").get_parameter_value().double_value
        self.obstacle_radius = self.get_parameter("obstacle_radius").get_parameter_value().double_value
        self.obstacle_sample_step = self.get_parameter("obstacle_sample_step").get_parameter_value().integer_value
        self.use_zed2_data = self.get_parameter("use_zed2_data").get_parameter_value().bool_value

        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.goal_topic = self.get_parameter("goal_topic").get_parameter_value().string_value
        self.cmd_topic = self.get_parameter("cmd_topic").get_parameter_value().string_value

        self.current_pose: Optional[Tuple[float, float, float]] = None
        self.current_twist: Optional[Tuple[float, float]] = None
        self.current_goal: Optional[Tuple[float, float]] = None
        self.latest_scan: Optional[LaserScan] = None
        self.map_bounds: Optional[Tuple[float, float, float, float]] = None

        # ZED2 data storage
        self.zed2_humans: List = []
        self.zed2_obstacles: List = []

        self.model = None
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

        self.FullState = None
        self.ObservableState = None
        self.ObstacleState = None
        self.WallState = None
        self.JointState = None
        self.joint_state_as_graph = None

        self._init_halo_model()

        self.create_subscription(Odometry, self.odom_topic, self._odom_callback, 10)
        self.create_subscription(LaserScan, self.scan_topic, self._scan_callback, 10)
        self.create_subscription(PoseStamped, self.goal_topic, self._goal_callback, 10)
        self.create_subscription(OccupancyGrid, "/map", self._map_callback, 10)

        # ZED2 subscriptions with BEST_EFFORT QoS to match publisher
        if self.use_zed2_data:
            zed_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
            self.create_subscription(HumanStateArray, "/zed2/humans", self._humans_callback, zed_qos)
            self.create_subscription(ObstacleStateArray, "/zed2/obstacles", self._obstacles_callback, zed_qos)
            self.get_logger().info("ZED2 integration enabled")

        self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_topic, 10)

        timer_period = self.get_parameter("timer_period").get_parameter_value().double_value
        self.create_timer(timer_period, self._control_loop)

        self.get_logger().info("RL Local Goal Bridge started.")
        self.get_logger().info(f"Model dir: {self.model_dir}")
        self.get_logger().info(f"Device: {self.device}")

    def _init_halo_model(self) -> None:
        if not os.path.isdir(self.halo_drl_dir):
            raise RuntimeError(f"halo_drl_dir does not exist: {self.halo_drl_dir}")

        if self.halo_drl_dir not in sys.path:
            sys.path.insert(0, self.halo_drl_dir)

        config_path = self.config_path
        if not os.path.isabs(config_path):
            config_path = os.path.join(self.halo_drl_dir, config_path)

        if not os.path.isfile(config_path):
            raise RuntimeError(f"Config file not found: {config_path}")

        import gym
        import crowd_sim.envs
        from algorithms.mpc_ppo import MpcPPO
        from modules.policies import ExternalPolicy
        from crowd_sim.envs.utils.robot import Robot
        from crowd_sim.envs.utils.state import FullState, ObservableState, ObstacleState, WallState, JointState
        from algorithms.graph_ppo import joint_state_as_graph

        spec = importlib.util.spec_from_file_location("config", config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        env_config = config.EnvConfig(False)

        env = gym.make("CrowdSim-v0", disable_env_checker=True)
        env.configure(env_config)
        env.set_phase(10)

        robot = Robot(env_config, "robot")
        robot.time_step = env.time_step
        robot.set_policy(ExternalPolicy())
        env.set_robot(robot)

        goal_range = (-self.action_range, self.action_range)
        env.num_actions_per_dim = self.action_dim
        env.goal_coord_range = goal_range
        env.action_space = gym.spaces.Discrete(self.action_dim * self.action_dim)
        env.use_AM = True
        env.use_action_mask = True
        env.use_PL = True

        self.model = MpcPPO(
            "GraphPolicy",
            env,
            device=self.device,
            action_dim=self.action_dim,
            goal_coord_range=goal_range,
            use_ros=False,
        )

        model_path = os.path.join(self.model_dir, "best_model")
        self.model.set_parameters(model_path, device=self.device)

        self.FullState = FullState
        self.ObservableState = ObservableState
        self.ObstacleState = ObstacleState
        self.WallState = WallState
        self.JointState = JointState
        self.joint_state_as_graph = joint_state_as_graph

    @staticmethod
    def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _odom_callback(self, msg: Odometry) -> None:
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = self._yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self.current_pose = (px, py, yaw)

        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        self.current_twist = (v, w)

    def _scan_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    def _goal_callback(self, msg: PoseStamped) -> None:
        self.current_goal = (msg.pose.position.x, msg.pose.position.y)

    def _map_callback(self, msg: OccupancyGrid) -> None:
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        width_m = msg.info.width * msg.info.resolution
        height_m = msg.info.height * msg.info.resolution
        self.map_bounds = (origin_x, origin_x + width_m, origin_y, origin_y + height_m)

    def _humans_callback(self, msg: HumanStateArray) -> None:
        """Callback for ZED2 human detections"""
        if self.ObservableState is None:
            return
        self.zed2_humans = []
        if self.current_pose is None:
            return
        px, py, yaw = self.current_pose
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)

        for human in msg.humans:
            # Transform from robot frame to world frame
            wx = px + human.px * cos_yaw - human.py * sin_yaw
            wy = py + human.px * sin_yaw + human.py * cos_yaw
            wvx = human.vx * cos_yaw - human.vy * sin_yaw
            wvy = human.vx * sin_yaw + human.vy * cos_yaw
            self.zed2_humans.append(
                self.ObservableState(wx, wy, wvx, wvy, human.radius)
            )

    def _obstacles_callback(self, msg: ObstacleStateArray) -> None:
        """Callback for ZED2 obstacle detections"""
        if self.ObstacleState is None:
            return
        self.zed2_obstacles = []
        if self.current_pose is None:
            return
        px, py, yaw = self.current_pose
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)

        for obs in msg.obstacles:
            # Transform from robot frame to world frame
            wx = px + obs.px * cos_yaw - obs.py * sin_yaw
            wy = py + obs.px * sin_yaw + obs.py * cos_yaw
            self.zed2_obstacles.append(
                self.ObstacleState(wx, wy, obs.radius)
            )

    def _build_walls(self) -> List:
        if self.map_bounds is None:
            if self.current_pose is None:
                xmin, xmax, ymin, ymax = -10.0, 10.0, -10.0, 10.0
            else:
                px, py, _ = self.current_pose
                xmin, xmax, ymin, ymax = px - 10.0, px + 10.0, py - 10.0, py + 10.0
        else:
            xmin, xmax, ymin, ymax = self.map_bounds

        left_wall = self.WallState(xmin, ymin, xmin, ymax)
        right_wall = self.WallState(xmax, ymin, xmax, ymax)
        return [left_wall, right_wall]

    def _scan_to_obstacles(self) -> List:
        obstacles = []
        if self.latest_scan is None or self.current_pose is None:
            return obstacles

        px, py, yaw = self.current_pose
        ranges = self.latest_scan.ranges
        angle_min = self.latest_scan.angle_min
        angle_inc = self.latest_scan.angle_increment
        r_min = self.latest_scan.range_min
        r_max = self.latest_scan.range_max

        for i in range(0, len(ranges), max(1, self.obstacle_sample_step)):
            r = ranges[i]
            if not math.isfinite(r):
                continue
            if r < r_min or r > r_max:
                continue

            ang = yaw + angle_min + i * angle_inc
            ox = px + r * math.cos(ang)
            oy = py + r * math.sin(ang)
            obstacles.append(self.ObstacleState(ox, oy, self.obstacle_radius))

        return obstacles

    def _publish_stop(self) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = 0.0
        msg.twist.angular.z = 0.0
        self.cmd_pub.publish(msg)

    def _control_loop(self) -> None:
        if self.model is None:
            return
        if self.current_pose is None or self.current_twist is None or self.current_goal is None:
            return

        px, py, yaw = self.current_pose
        gx, gy = self.current_goal

        goal_dist = math.hypot(gx - px, gy - py)
        if goal_dist < self.goal_tolerance:
            self._publish_stop()
            return

        v, w = self.current_twist
        v_left = v - 0.5 * self.wheel_base * w
        v_right = v + 0.5 * self.wheel_base * w

        full_state = self.FullState(
            px,
            py,
            v_left,
            v_right,
            self.robot_radius,
            gx,
            gy,
            0.5,
            yaw,
        )

        obstacles = self._scan_to_obstacles()
        # Add ZED2 obstacles (people detected as obstacles)
        if self.use_zed2_data:
            obstacles.extend(self.zed2_obstacles)

        walls = self._build_walls()

        # Use ZED2 humans or empty list
        humans = self.zed2_humans if self.use_zed2_data else []

        observed_state = (humans, obstacles, walls, [])

        try:
            joint_state = self.JointState(full_state, observed_state)
            ob_graph = self.joint_state_as_graph(joint_state, device=self.device)

            with th.no_grad():
                action = self.model.policy.predict(ob_graph, action_mask=None, deterministic=True).squeeze()
                action_idx = int(action.item() if hasattr(action, "item") else action)

            local_goal_x, local_goal_y = self.model.map_action_to_goal(action_idx)

            heading_error = math.atan2(local_goal_y, max(local_goal_x, 1e-6))
            cmd_linear = self._clamp(self.k_linear * local_goal_x, 0.0, self.max_linear_speed)
            cmd_angular = self._clamp(self.k_angular * heading_error, -self.max_angular_speed, self.max_angular_speed)

            if goal_dist < 1.0:
                cmd_linear *= goal_dist

            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.twist.linear.x = float(cmd_linear)
            msg.twist.angular.z = float(cmd_angular)
            self.cmd_pub.publish(msg)
        except Exception as exc:
            self.get_logger().error(f"Inference step failed: {exc}")
            self._publish_stop()


def main(args=None):
    rclpy.init(args=args)
    node = RLLocalGoalBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_stop()
        node.destroy_node()
        rclpy.shutdown()
