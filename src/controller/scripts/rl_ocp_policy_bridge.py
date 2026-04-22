#!/usr/bin/python3
import json
import math
import os
import shlex
import subprocess
import sys
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from interfaces.msg import ObstacleState, WallState, PolyState, HumanState
from interfaces.msg import JointState as InterfaceJointState
from interfaces.srv import OcpLocalPlann
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import FollowPath
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from nav2_msgs.action import FollowPath
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rcl_interfaces.msg import SetParametersResult
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener


class RlOcpPolicyBridge(Node):
    _GOAL_SOURCE_RL = "rl_local_goal"
    _GOAL_SOURCE_RVIZ = "rviz_global_goal"

    def __init__(self):
        super().__init__("rl_ocp_policy_bridge")

        self.declare_parameter("policy_env_name", "mpc_rl")
        self.declare_parameter("halo_drl_dir", "HALO_1/drl_moudle")
        self.declare_parameter("config", "configs/mpc_rl.py")
        self.declare_parameter("model_path", "src/controller/policy/best_model.zip")
        self.declare_parameter("action_dim", 9)
        self.declare_parameter("action_range", 2.25)

        self.declare_parameter("goal_tolerance", 0.25)
        self.declare_parameter("robot_radius", 0.25)
        self.declare_parameter("axle_half_width", 0.23)
        self.declare_parameter("v_pref", 0.8)
        self.declare_parameter("obstacle_radius", 0.2)
        self.declare_parameter("obstacle_sample_step", 16)
        self.declare_parameter("use_map_static_obstacles", True)
        self.declare_parameter("map_static_obstacle_radius", 0.2)
        self.declare_parameter("map_static_sample_step_m", 0.3)
        self.declare_parameter("map_static_obstacle_limit", 40)
        self.declare_parameter("enforce_scene_entities", True)
        self.declare_parameter("human_fallback_enabled", True)
        self.declare_parameter("human_fallback_limit", 20)
        self.declare_parameter("human_fallback_radius", 0.25)
        self.declare_parameter("auto_relax_constraints", False)
        self.declare_parameter("timer_period", 0.1)
        self.declare_parameter("policy_hz", 10.0)
        self.declare_parameter("service_hz", 10.0)
        self.declare_parameter("goal_source_mode", self._GOAL_SOURCE_RL)
        self.declare_parameter("policy_period", 0.2)
        self.declare_parameter("mpc_period", 0.05)

        self.declare_parameter("planner_half_width", 5.8)
        self.declare_parameter("planner_half_height", 9.8)
        self.declare_parameter("planner_hard_half_width", 5.99)
        self.declare_parameter("planner_hard_half_height", 9.99)
        self.declare_parameter("map_geometry_obstacle_threshold", 90)
        self.declare_parameter("map_geometry_poly_epsilon_ratio", 0.02)
        self.declare_parameter("map_geometry_wall_linearity_ratio", 20.0)
        self.declare_parameter("map_geometry_wall_linearity_span", 0.50)
        self.declare_parameter("map_geometry_boundary_span_threshold", 2.5)
        self.declare_parameter("map_geometry_wall_min_segment_length", 0.35)
        self.declare_parameter("map_geometry_wall_merge_angle_deg", 12.0)
        self.declare_parameter("map_geometry_boundary_edge_margin_px", 8)
        self.declare_parameter("map_geometry_boundary_area_ratio", 0.25)
        self.declare_parameter("map_geometry_debug_log", True)
        self.declare_parameter("map_geometry_debug_period_sec", 2.0)
        self.declare_parameter("visualize_actions", True)
        self.declare_parameter("action_marker_topic", "/policy/action_markers")
        self.declare_parameter("action_marker_frame", "odom")
        self.declare_parameter("action_mask_clearance", 0.05)
        self.declare_parameter("publish_debug_joint_state", True)
        self.declare_parameter("debug_joint_state_topic", "/debug/joint_state_req")
        self.declare_parameter("publish_policy_debug_status", True)
        self.declare_parameter("policy_debug_status_topic", "/debug/policy_status")
        self.declare_parameter("visualize_planner_scene", True)
        self.declare_parameter("planner_scene_marker_topic", "/planner/debug_markers")
        self.declare_parameter("planner_scene_frame", "map")
        self.declare_parameter("action_debug_topic", "/debug/policy_actions_scene")
        self.declare_parameter("planner_scene_debug_topic", "/debug/planner_scene")

        self.declare_parameter("odom_topic", "/odometry/filtered")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("human_topic", "/tracking/humans")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("planner_service", "/ocp_plann")
        self.declare_parameter("follow_path_action", "/follow_path")
        self.declare_parameter("controller_id", "FollowPath")
        self.declare_parameter("goal_checker_id", "general_goal_checker")
        self.declare_parameter("local_goal_topic", "/rl/local_goal")
        self.declare_parameter("local_path_topic", "/rl/local_path")
        self.declare_parameter("follow_path_replan_min_interval_sec", 0.35)
        self.declare_parameter("follow_path_replan_path_delta", 0.20)

        self.policy_env_name = self.get_parameter("policy_env_name").get_parameter_value().string_value
        self.halo_drl_dir = self.get_parameter("halo_drl_dir").get_parameter_value().string_value
        self.config_path = self.get_parameter("config").get_parameter_value().string_value
        self.model_path = self.get_parameter("model_path").get_parameter_value().string_value

        self.action_dim = self.get_parameter("action_dim").get_parameter_value().integer_value
        self.action_range = self.get_parameter("action_range").get_parameter_value().double_value
        self.goal_tolerance = self.get_parameter("goal_tolerance").get_parameter_value().double_value
        self.robot_radius = self.get_parameter("robot_radius").get_parameter_value().double_value
        self.axle_half_width = self.get_parameter("axle_half_width").get_parameter_value().double_value
        self.v_pref = self.get_parameter("v_pref").get_parameter_value().double_value
        self.obstacle_radius = self.get_parameter("obstacle_radius").get_parameter_value().double_value
        self.obstacle_sample_step = self.get_parameter("obstacle_sample_step").get_parameter_value().integer_value
        self.use_map_static_obstacles = self.get_parameter("use_map_static_obstacles").get_parameter_value().bool_value
        self.map_static_obstacle_radius = self.get_parameter("map_static_obstacle_radius").get_parameter_value().double_value
        self.map_static_sample_step_m = self.get_parameter("map_static_sample_step_m").get_parameter_value().double_value
        self.map_static_obstacle_limit = self.get_parameter("map_static_obstacle_limit").get_parameter_value().integer_value
        self.enforce_scene_entities = self.get_parameter("enforce_scene_entities").get_parameter_value().bool_value
        self.human_fallback_enabled = self.get_parameter("human_fallback_enabled").get_parameter_value().bool_value
        self.human_fallback_limit = self.get_parameter("human_fallback_limit").get_parameter_value().integer_value
        self.human_fallback_radius = self.get_parameter("human_fallback_radius").get_parameter_value().double_value
        self.auto_relax_constraints = self.get_parameter("auto_relax_constraints").get_parameter_value().bool_value
        self.policy_hz = max(0.001, self.get_parameter("policy_hz").get_parameter_value().double_value)
        self.service_hz = max(0.001, self.get_parameter("service_hz").get_parameter_value().double_value)
        self.goal_source_mode = self._normalize_goal_source_mode(
            self.get_parameter("goal_source_mode").get_parameter_value().string_value
        )
        self.policy_period = 1.0 / self.policy_hz
        self.mpc_period = 1.0 / self.service_hz
        self.planner_half_width = self.get_parameter("planner_half_width").get_parameter_value().double_value
        self.planner_half_height = self.get_parameter("planner_half_height").get_parameter_value().double_value
        self.planner_hard_half_width = self.get_parameter("planner_hard_half_width").get_parameter_value().double_value
        self.planner_hard_half_height = self.get_parameter("planner_hard_half_height").get_parameter_value().double_value
        self.effective_half_width = min(self.planner_half_width, self.planner_hard_half_width)
        self.effective_half_height = min(self.planner_half_height, self.planner_hard_half_height)
        self.map_geometry_obstacle_threshold = int(
            self.get_parameter("map_geometry_obstacle_threshold").get_parameter_value().integer_value
        )
        self.map_geometry_poly_epsilon_ratio = float(
            self.get_parameter("map_geometry_poly_epsilon_ratio").get_parameter_value().double_value
        )
        self.map_geometry_wall_linearity_ratio = float(
            self.get_parameter("map_geometry_wall_linearity_ratio").get_parameter_value().double_value
        )
        self.map_geometry_wall_linearity_span = float(
            self.get_parameter("map_geometry_wall_linearity_span").get_parameter_value().double_value
        )
        self.map_geometry_boundary_span_threshold = float(
            self.get_parameter("map_geometry_boundary_span_threshold").get_parameter_value().double_value
        )
        self.map_geometry_wall_min_segment_length = float(
            self.get_parameter("map_geometry_wall_min_segment_length").get_parameter_value().double_value
        )
        self.map_geometry_wall_merge_angle_deg = float(
            self.get_parameter("map_geometry_wall_merge_angle_deg").get_parameter_value().double_value
        )
        self.map_geometry_boundary_edge_margin_px = int(
            self.get_parameter("map_geometry_boundary_edge_margin_px").get_parameter_value().integer_value
        )
        self.map_geometry_boundary_area_ratio = float(
            self.get_parameter("map_geometry_boundary_area_ratio").get_parameter_value().double_value
        )
        self.map_geometry_debug_log = self.get_parameter("map_geometry_debug_log").get_parameter_value().bool_value
        self.map_geometry_debug_period_sec = max(
            0.1,
            self.get_parameter("map_geometry_debug_period_sec").get_parameter_value().double_value,
        )
        self.visualize_actions = self.get_parameter("visualize_actions").get_parameter_value().bool_value
        self.action_marker_topic = self.get_parameter("action_marker_topic").get_parameter_value().string_value
        self.action_marker_frame = self.get_parameter("action_marker_frame").get_parameter_value().string_value
        self.action_mask_clearance = self.get_parameter("action_mask_clearance").get_parameter_value().double_value
        self.publish_debug_joint_state = self.get_parameter("publish_debug_joint_state").get_parameter_value().bool_value
        self.debug_joint_state_topic = self.get_parameter("debug_joint_state_topic").get_parameter_value().string_value
        self.publish_policy_debug_status = self.get_parameter("publish_policy_debug_status").get_parameter_value().bool_value
        self.policy_debug_status_topic = self.get_parameter("policy_debug_status_topic").get_parameter_value().string_value
        self.visualize_planner_scene = self.get_parameter("visualize_planner_scene").get_parameter_value().bool_value
        self.planner_scene_marker_topic = self.get_parameter("planner_scene_marker_topic").get_parameter_value().string_value
        self.planner_scene_frame = self.get_parameter("planner_scene_frame").get_parameter_value().string_value
        self.action_debug_topic = self.get_parameter("action_debug_topic").get_parameter_value().string_value
        self.planner_scene_debug_topic = self.get_parameter("planner_scene_debug_topic").get_parameter_value().string_value

        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.goal_topic = self.get_parameter("goal_topic").get_parameter_value().string_value
        self.map_topic = self.get_parameter("map_topic").get_parameter_value().string_value
        self.human_topic = self.get_parameter("human_topic").get_parameter_value().string_value
        self.map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        self.planner_service = self.get_parameter("planner_service").get_parameter_value().string_value
        self.follow_path_action = self.get_parameter("follow_path_action").get_parameter_value().string_value
        self.controller_id = self.get_parameter("controller_id").get_parameter_value().string_value
        self.goal_checker_id = self.get_parameter("goal_checker_id").get_parameter_value().string_value
        self.local_goal_topic = self.get_parameter("local_goal_topic").get_parameter_value().string_value
        self.local_path_topic = self.get_parameter("local_path_topic").get_parameter_value().string_value
        self.follow_path_replan_min_interval_sec = max(
            0.0,
            self.get_parameter("follow_path_replan_min_interval_sec").get_parameter_value().double_value,
        )
        self.follow_path_replan_path_delta = max(
            0.01,
            self.get_parameter("follow_path_replan_path_delta").get_parameter_value().double_value,
        )

        self.current_pose: Optional[Tuple[float, float, float]] = None
        self.current_twist: Optional[Tuple[float, float]] = None
        self.current_goal: Optional[Tuple[float, float]] = None
        self.current_goal_frame: str = ""
        self.latest_scan: Optional[LaserScan] = None
        self.latest_humans: Optional[HumanArray] = None
        self.map_bounds: Optional[Tuple[float, float, float, float]] = None
        self.map_resolution: Optional[float] = None
        self.map_width: int = 0
        self.map_height: int = 0
        self.map_origin_x: float = 0.0
        self.map_origin_y: float = 0.0
        self.map_data: Optional[List[int]] = None
        self.map_occupancy_threshold: int = 50
        self.pose_frame: str = ""
        self._has_received_map = False
        self.map_geometry_polygons: List[List[Tuple[float, float]]] = []
        self.map_geometry_walls: List[Tuple[float, float, float, float]] = []
        self.map_static_circles: List[Tuple[float, float, float]] = []
        self.last_sent_map_polygons_count = 0
        self.last_sent_map_walls_count = 0
        self._last_map_geometry_log_ns = 0
        self._last_sent_geometry_log_ns = 0

        # Keep Python-side request geometry aligned with hard-coded C++ planner map limits.
        self.planner_x_limit = 5.5
        self.planner_y_limit = 9.8

        self.pending_future = None
        self.follow_path_goal_future = None
        self.follow_path_result_future = None
        self.follow_path_cancel_future = None
        self.active_follow_path_goal_handle = None
        self.pending_follow_path_msg: Optional[Path] = None
        self.last_sent_follow_path: Optional[Path] = None
        self.last_follow_path_send_ns: int = 0
        self.last_wheel_speed: Optional[Tuple[float, float]] = None
        self.latest_sub_goal: Optional[Tuple[float, float]] = None
        self.latest_sub_goal_seq = 0
        self.last_requested_sub_goal_seq = -1
        self.worker: Optional[subprocess.Popen] = None
        self.worker_stop_event = threading.Event()
        self.worker_state_lock = threading.Lock()
        self.worker_request_cond = threading.Condition(self.worker_state_lock)
        self.policy_worker_thread: Optional[threading.Thread] = None
        self.policy_worker_stderr_thread: Optional[threading.Thread] = None
        self.pending_policy_request: Optional[dict] = None
        self.pending_policy_request_seq = 0
        self.latest_policy_result: Optional[Tuple[float, float]] = None
        self.latest_policy_result_seq = -1
        self.last_applied_policy_result_seq = -1
        self.policy_worker_busy = False
        self.policy_worker_ready = False
        self.policy_worker_last_error = ""
        self.last_request_for_viz: Optional[OcpLocalPlann.Request] = None
        self.consecutive_planner_failures = 0
        self.success_streak = 0
        self.goal_reached = False
        self.last_astar_path_for_viz: List[Tuple[float, float]] = []
        self.last_mask_applied = False
        self.last_mask_snap_distance = 0.0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._start_policy_worker()

        self.create_subscription(Odometry, self.odom_topic, self._odom_callback, 10)
        self.create_subscription(LaserScan, self.scan_topic, self._scan_callback, 10)
        self.create_subscription(PoseStamped, self.goal_topic, self._goal_callback, 10)
        self.create_subscription(HumanArray, self.human_topic, self._human_callback, 10)
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(OccupancyGrid, self.map_topic, self._map_callback, map_qos)

        self.debug_joint_state_pub = self.create_publisher(InterfaceJointState, self.debug_joint_state_topic, 10)
        self.policy_debug_status_pub = self.create_publisher(String, self.policy_debug_status_topic, 10)
        self.action_debug_pub = self.create_publisher(String, self.action_debug_topic, 10)
        self.planner_scene_debug_pub = self.create_publisher(String, self.planner_scene_debug_topic, 10)
        self.local_goal_pub = self.create_publisher(PoseStamped, self.local_goal_topic, 10)
        self.local_path_pub = self.create_publisher(Path, self.local_path_topic, 10)
        self.ocp_client = self.create_client(OcpLocalPlann, self.planner_service)
        self.follow_path_client = ActionClient(self, FollowPath, self.follow_path_action)
        self.add_on_set_parameters_callback(self._on_parameters_changed)

        # Keep legacy timer_period declared for backward compatibility,
        # but run policy and planner/service loops on dedicated timers.
        self.create_timer(self.policy_period, self._policy_loop)
        self.create_timer(self.mpc_period, self._planner_loop)

        self.get_logger().info("RL OCP policy bridge started")
        self.get_logger().info(f"Model: {self.model_path}")
        self.get_logger().info(f"Service: {self.planner_service}")
        self.get_logger().info(f"FollowPath action: {self.follow_path_action}")
        self.get_logger().info(
            f"Loop rates: RL={1.0 / self.policy_period:.2f} Hz, service={1.0 / self.mpc_period:.2f} Hz"
        )
        self.get_logger().info(f"Goal source mode: {self.goal_source_mode}")
        self.get_logger().info("Pipeline mode: RL/RViz local goal -> A* service -> MPPI FollowPath")
        self.get_logger().info(f"Auto relax constraints: {self.auto_relax_constraints}")
        self.get_logger().info(f"Action visualization: {self.visualize_actions} ({self.action_marker_topic})")
        self.get_logger().info(
            f"JointState debug topic: {self.publish_debug_joint_state} ({self.debug_joint_state_topic})"
        )
        self.get_logger().info(
            f"Policy status topic: {self.publish_policy_debug_status} ({self.policy_debug_status_topic})"
        )

    def _on_parameters_changed(self, params):
        update_loop_hz = False
        update_effective_bounds = False
        for p in params:
            if p.name == "policy_hz":
                self.policy_hz = max(0.001, float(p.value))
                update_loop_hz = True
            elif p.name == "service_hz":
                self.service_hz = max(0.001, float(p.value))
                update_loop_hz = True
            elif p.name == "goal_source_mode":
                self.goal_source_mode = self._normalize_goal_source_mode(str(p.value))
                self.get_logger().info(f"Updated goal_source_mode: {self.goal_source_mode}")
            if p.name == "visualize_actions":
                self.visualize_actions = bool(p.value)
            elif p.name == "publish_debug_joint_state":
                self.publish_debug_joint_state = bool(p.value)
            elif p.name == "publish_policy_debug_status":
                self.publish_policy_debug_status = bool(p.value)
            elif p.name == "visualize_planner_scene":
                self.visualize_planner_scene = bool(p.value)
            elif p.name == "use_map_static_obstacles":
                self.use_map_static_obstacles = bool(p.value)
            elif p.name == "map_static_obstacle_radius":
                self.map_static_obstacle_radius = float(p.value)
            elif p.name == "map_static_sample_step_m":
                self.map_static_sample_step_m = float(p.value)
            elif p.name == "map_static_obstacle_limit":
                self.map_static_obstacle_limit = int(p.value)
            elif p.name == "enforce_scene_entities":
                self.enforce_scene_entities = bool(p.value)
            elif p.name == "human_fallback_enabled":
                self.human_fallback_enabled = bool(p.value)
            elif p.name == "human_fallback_limit":
                self.human_fallback_limit = int(p.value)
            elif p.name == "human_fallback_radius":
                self.human_fallback_radius = float(p.value)
            elif p.name == "auto_relax_constraints":
                self.auto_relax_constraints = bool(p.value)
            elif p.name == "action_dim":
                self.action_dim = max(2, int(p.value))
            elif p.name == "action_range":
                self.action_range = float(p.value)
            elif p.name == "planner_half_width":
                self.planner_half_width = float(p.value)
                update_effective_bounds = True
            elif p.name == "planner_half_height":
                self.planner_half_height = float(p.value)
                update_effective_bounds = True
            elif p.name == "planner_hard_half_width":
                self.planner_hard_half_width = float(p.value)
                update_effective_bounds = True
            elif p.name == "planner_hard_half_height":
                self.planner_hard_half_height = float(p.value)
                update_effective_bounds = True
            elif p.name == "map_geometry_obstacle_threshold":
                self.map_geometry_obstacle_threshold = int(p.value)
            elif p.name == "map_geometry_poly_epsilon_ratio":
                self.map_geometry_poly_epsilon_ratio = float(p.value)
            elif p.name == "map_geometry_wall_linearity_ratio":
                self.map_geometry_wall_linearity_ratio = float(p.value)
            elif p.name == "map_geometry_wall_linearity_span":
                self.map_geometry_wall_linearity_span = float(p.value)
            elif p.name == "map_geometry_boundary_span_threshold":
                self.map_geometry_boundary_span_threshold = max(0.1, float(p.value))
            elif p.name == "map_geometry_wall_min_segment_length":
                self.map_geometry_wall_min_segment_length = max(0.01, float(p.value))
            elif p.name == "map_geometry_wall_merge_angle_deg":
                self.map_geometry_wall_merge_angle_deg = max(0.1, float(p.value))
            elif p.name == "map_geometry_boundary_edge_margin_px":
                self.map_geometry_boundary_edge_margin_px = max(0, int(p.value))
            elif p.name == "map_geometry_boundary_area_ratio":
                self.map_geometry_boundary_area_ratio = max(0.01, min(1.0, float(p.value)))
            elif p.name == "follow_path_replan_min_interval_sec":
                self.follow_path_replan_min_interval_sec = max(0.0, float(p.value))
            elif p.name == "follow_path_replan_path_delta":
                self.follow_path_replan_path_delta = max(0.01, float(p.value))

        if update_effective_bounds:
            self.effective_half_width = min(self.planner_half_width, self.planner_hard_half_width)
            self.effective_half_height = min(self.planner_half_height, self.planner_hard_half_height)

        if update_loop_hz:
            self.policy_period = 1.0 / self.policy_hz
            self.mpc_period = 1.0 / self.service_hz

        return SetParametersResult(successful=True)

    def _normalize_goal_source_mode(self, mode: str) -> str:
        mode_norm = str(mode).strip().lower()
        if mode_norm in (self._GOAL_SOURCE_RL, self._GOAL_SOURCE_RVIZ):
            return mode_norm
        self.get_logger().warn(
            "Unsupported goal_source_mode '%s', fallback to '%s'"
            % (str(mode), self._GOAL_SOURCE_RL)
        )
        return self._GOAL_SOURCE_RL

    def _start_policy_worker(self) -> None:
        worker_script = os.path.join(os.path.dirname(__file__), "rl_policy_worker.py")
        if not os.path.isfile(worker_script):
            raise RuntimeError(f"Policy worker script not found: {worker_script}")

        cmd = [
            "conda", "run", "--no-capture-output", "-n", self.policy_env_name,
            "python", worker_script,
            "--halo_drl_dir", self.halo_drl_dir,
            "--config", self.config_path,
            "--model_path", self.model_path,
            "--action_dim", str(self.action_dim),
            "--action_range", str(self.action_range),
            "--v_pref", str(self.v_pref),
            "--robot_radius", str(self.robot_radius),
        ]

        self.get_logger().info(f"Starting policy worker: {' '.join(shlex.quote(x) for x in cmd)}")
        self.worker = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.policy_worker_thread = threading.Thread(
            target=self._policy_worker_loop,
            name="rl_policy_worker_loop",
            daemon=True,
        )
        self.policy_worker_thread.start()
        self.policy_worker_stderr_thread = threading.Thread(
            target=self._policy_worker_stderr_loop,
            name="rl_policy_worker_stderr",
            daemon=True,
        )
        self.policy_worker_stderr_thread.start()

    def _policy_worker_stderr_loop(self) -> None:
        if self.worker is None or self.worker.stderr is None:
            return
        for line in self.worker.stderr:
            if self.worker_stop_event.is_set():
                break
            msg = line.strip()
            if not msg:
                continue
            with self.worker_state_lock:
                self.policy_worker_last_error = msg

    def _submit_policy_request(self, subgoal_req: dict) -> None:
        with self.worker_request_cond:
            self.pending_policy_request_seq += 1
            self.pending_policy_request = {
                "seq": int(self.pending_policy_request_seq),
                "payload": subgoal_req,
            }
            self.worker_request_cond.notify()

    def _take_latest_policy_result(self) -> Optional[Tuple[Tuple[float, float], int]]:
        with self.worker_state_lock:
            if self.latest_policy_result is None:
                return None
            if self.latest_policy_result_seq <= self.last_applied_policy_result_seq:
                return None
            self.last_applied_policy_result_seq = int(self.latest_policy_result_seq)
            return self.latest_policy_result, int(self.latest_policy_result_seq)

    def _policy_worker_loop(self) -> None:
        while not self.worker_stop_event.is_set():
            request = None
            with self.worker_request_cond:
                while self.pending_policy_request is None and not self.worker_stop_event.is_set():
                    self.worker_request_cond.wait(timeout=0.1)
                if self.worker_stop_event.is_set():
                    return
                request = self.pending_policy_request
                self.pending_policy_request = None
                self.policy_worker_busy = True

            if request is None:
                continue

            sub_goal = self._query_policy_worker(request["payload"])

            with self.worker_state_lock:
                self.policy_worker_busy = False
                if sub_goal is None:
                    continue
                self.policy_worker_ready = True
                self.latest_policy_result = (float(sub_goal[0]), float(sub_goal[1]))
                self.latest_policy_result_seq = int(request["seq"])

    def _query_policy_worker(self, subgoal_req: dict) -> Optional[Tuple[float, float]]:
        if self.worker is None:
            return None
        if self.worker.poll() is not None:
            err = self.policy_worker_last_error
            self.get_logger().error(f"Policy worker exited unexpectedly: {err}")
            return None

        if self.worker.stdin is None or self.worker.stdout is None:
            return None

        self.worker.stdin.write(json.dumps(subgoal_req) + "\n")
        self.worker.stdin.flush()

        line = self.worker.stdout.readline()
        if not line:
            self.get_logger().error("Policy worker returned empty response")
            return None

        payload = json.loads(line)
        if "error" in payload:
            self.get_logger().error(f"Policy worker error: {payload['error']}")
            return None

        sub_goal = payload.get("sub_goal", None)
        if sub_goal is None or len(sub_goal) != 2:
            self.get_logger().error("Policy worker response missing sub_goal")
            return None

        return float(sub_goal[0]), float(sub_goal[1])

    def _publish_local_goal(self, sub_goal: Tuple[float, float]) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.pose.position.x = float(sub_goal[0])
        msg.pose.position.y = float(sub_goal[1])
        msg.pose.orientation.w = 1.0
        self.local_goal_pub.publish(msg)

    def _build_path_msg(self, path_points, frame_id: str) -> Path:
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = frame_id if frame_id else self.map_frame
        for pt in path_points:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(pt.x)
            pose.pose.position.y = float(pt.y)
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        return path_msg

    def _cancel_follow_path_goal(self) -> None:
        if self.active_follow_path_goal_handle is None:
            return
        if self.follow_path_cancel_future is not None and not self.follow_path_cancel_future.done():
            return
        self.follow_path_cancel_future = self.active_follow_path_goal_handle.cancel_goal_async()
        self.follow_path_cancel_future.add_done_callback(self._on_follow_path_cancel_done)

    def _path_distance(self, a: Path, b: Path) -> float:
        if len(a.poses) == 0 or len(b.poses) == 0:
            return float("inf")
        count = min(len(a.poses), len(b.poses))
        if count <= 0:
            return float("inf")
        stride = max(1, count // 20)
        max_dist = 0.0
        for i in range(0, count, stride):
            pa = a.poses[i].pose.position
            pb = b.poses[i].pose.position
            dist = math.hypot(float(pa.x) - float(pb.x), float(pa.y) - float(pb.y))
            max_dist = max(max_dist, dist)

        a_end = a.poses[-1].pose.position
        b_end = b.poses[-1].pose.position
        end_dist = math.hypot(float(a_end.x) - float(b_end.x), float(a_end.y) - float(b_end.y))
        return max(max_dist, end_dist)

    def _should_replan_follow_path(self, path_msg: Path) -> bool:
        if self.last_sent_follow_path is None:
            return True
        path_delta = self._path_distance(path_msg, self.last_sent_follow_path)
        if path_delta < self.follow_path_replan_path_delta:
            return False

        now_ns = int(self.get_clock().now().nanoseconds)
        elapsed = (now_ns - int(self.last_follow_path_send_ns)) / 1e9
        return elapsed >= self.follow_path_replan_min_interval_sec

    def _send_new_follow_path_goal(self, goal_msg: FollowPath.Goal) -> None:
        sent_path = goal_msg.path
        self.follow_path_goal_future = self.follow_path_client.send_goal_async(goal_msg)
        self.follow_path_goal_future.add_done_callback(
            lambda future, path=sent_path: self._on_follow_path_goal_response(future, path)
        )

    def _on_follow_path_cancel_done(self, future) -> None:
        try:
            _ = future.result()
        except Exception as exc:
            self.get_logger().warn(f"FollowPath cancel failed: {exc}")
        finally:
            self.follow_path_cancel_future = None
            self.active_follow_path_goal_handle = None

        if self.pending_follow_path_msg is None:
            return
        goal_msg = FollowPath.Goal()
        goal_msg.path = self.pending_follow_path_msg
        goal_msg.controller_id = self.controller_id
        goal_msg.goal_checker_id = self.goal_checker_id
        self.pending_follow_path_msg = None
        self._send_new_follow_path_goal(goal_msg)

    def _send_follow_path_goal(self, path_msg: Path) -> None:
        if len(path_msg.poses) < 2:
            self.get_logger().warn("Skip FollowPath goal because A* path has fewer than 2 poses")
            return
        if not self.follow_path_client.wait_for_server(timeout_sec=0.05):
            self.get_logger().warn(f"FollowPath action {self.follow_path_action} not available")
            return

        goal_msg = FollowPath.Goal()
        goal_msg.path = path_msg
        goal_msg.controller_id = self.controller_id
        goal_msg.goal_checker_id = self.goal_checker_id

        if self.active_follow_path_goal_handle is not None:
            if not self._should_replan_follow_path(path_msg):
                return
            self.pending_follow_path_msg = path_msg
            self._cancel_follow_path_goal()
            return

        self._send_new_follow_path_goal(goal_msg)

    def _on_follow_path_goal_response(self, future, sent_path: Path) -> None:
        self.follow_path_goal_future = None
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"FollowPath goal send failed: {exc}")
            return

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn("FollowPath goal was rejected by controller_server")
            return

        self.active_follow_path_goal_handle = goal_handle
        self.last_sent_follow_path = sent_path
        self.last_follow_path_send_ns = int(self.get_clock().now().nanoseconds)
        self.follow_path_result_future = goal_handle.get_result_async()
        self.follow_path_result_future.add_done_callback(self._on_follow_path_result)

    def _on_follow_path_result(self, future) -> None:
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error(f"FollowPath result failed: {exc}")
            return

        self.active_follow_path_goal_handle = None
        if result is None:
            return

    @staticmethod
    def _path_points_from_response(res) -> List[Tuple[float, float]]:
        return [(float(pt.x), float(pt.y)) for pt in res.astar_path]

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
        self.pose_frame = msg.header.frame_id if msg.header.frame_id else self.pose_frame

        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        self.current_twist = (v, w)

    def _scan_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    def _goal_callback(self, msg: PoseStamped) -> None:
        self.current_goal = (msg.pose.position.x, msg.pose.position.y)
        self.current_goal_frame = msg.header.frame_id if msg.header.frame_id else self.current_goal_frame

    def _human_callback(self, msg: HumanArray) -> None:
        self.latest_humans = msg

    def _map_callback(self, msg: OccupancyGrid) -> None:
        if msg.header.frame_id:
            self.map_frame = msg.header.frame_id
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        width_m = msg.info.width * msg.info.resolution
        height_m = msg.info.height * msg.info.resolution
        self.map_bounds = (origin_x, origin_x + width_m, origin_y, origin_y + height_m)
        self.map_resolution = float(msg.info.resolution)
        self.map_width = int(msg.info.width)
        self.map_height = int(msg.info.height)
        self.map_origin_x = float(origin_x)
        self.map_origin_y = float(origin_y)
        self.map_data = list(msg.data)

        if not self._has_received_map:
            self._has_received_map = True
            self.get_logger().info(
                "Received map: frame=%s size=%dx%d res=%.3f origin=(%.2f, %.2f)"
                % (
                    str(self.map_frame),
                    int(self.map_width),
                    int(self.map_height),
                    float(self.map_resolution),
                    float(self.map_origin_x),
                    float(self.map_origin_y),
                )
            )

        if msg.info.width == 0 or msg.info.height == 0:
            self.map_geometry_polygons = []
            self.map_geometry_walls = []
            self.map_static_circles = []
            return

        # Grid-based circle sampling mode: disable contour/polygon/wall extraction entirely.
        self.map_geometry_polygons = []
        self.map_geometry_walls = []
        self.map_static_circles = []

        if self.map_geometry_debug_log:
            self._throttled_log(
                "_last_map_geometry_log_ns",
                self.map_geometry_debug_period_sec,
                "[map_geometry/grid] map updated; walls=0 polygons=0 (using OccupancyGrid sampling)",
            )

    def _pixel_to_meter(self, u: int, v: int) -> Tuple[float, float]:
        row_occ = self.map_height - 1 - v
        x_world = self.map_origin_x + (float(u) + 0.5) * self.map_resolution
        y_world = self.map_origin_y + (float(row_occ) + 0.5) * self.map_resolution
        return x_world, y_world

    def _is_wall_like_geometry(self, points: List[Tuple[float, float]]) -> bool:
        if len(points) < 2:
            return False

        arr = np.array(points, dtype=np.float64)
        diffs = arr[:, None, :] - arr[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        span = float(np.max(dists)) if dists.size > 0 else 0.0
        if span < self.map_geometry_wall_linearity_span:
            return False

        if arr.shape[0] < 3:
            return True

        centered = arr - np.mean(arr, axis=0)
        cov = np.cov(centered, rowvar=False)
        if cov.shape != (2, 2):
            return False

        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(np.abs(eigvals))
        small = float(eigvals[0])
        large = float(eigvals[1])
        linearity = large / (small + 1e-9)
        return linearity >= self.map_geometry_wall_linearity_ratio

    @staticmethod
    def _wall_segment_from_points(
        points: List[Tuple[float, float]],
    ) -> Optional[Tuple[float, float, float, float]]:
        if len(points) < 2:
            return None
        arr = np.array(points, dtype=np.float64)
        diffs = arr[:, None, :] - arr[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        max_idx = np.unravel_index(np.argmax(dists), dists.shape)
        p1 = arr[max_idx[0]]
        p2 = arr[max_idx[1]]
        return (float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]))

    @staticmethod
    def _segment_length(sx: float, sy: float, ex: float, ey: float) -> float:
        return math.hypot(float(ex) - float(sx), float(ey) - float(sy))

    def _is_map_boundary_contour(self, contour, area_ratio: float) -> bool:
        if contour is None or len(contour) == 0:
            return False

        margin = max(0, int(self.map_geometry_boundary_edge_margin_px))
        x, y, w, h = self.cv2.boundingRect(contour)
        near_left = x <= margin
        near_top = y <= margin
        near_right = (x + w) >= (self.map_width - 1 - margin)
        near_bottom = (y + h) >= (self.map_height - 1 - margin)
        touch_count = int(near_left) + int(near_top) + int(near_right) + int(near_bottom)

        if touch_count >= 2:
            return True
        if area_ratio >= float(self.map_geometry_boundary_area_ratio):
            return True
        return False

    def _rectangle_walls_from_contour(self, contour) -> List[Tuple[float, float, float, float]]:
        if contour is None or len(contour) < 2:
            return []

        rect = self.cv2.minAreaRect(contour)
        box = self.cv2.boxPoints(rect)
        if box is None or len(box) != 4:
            return []

        pts: List[Tuple[float, float]] = []
        for p in box:
            u = int(round(float(p[0])))
            v = int(round(float(p[1])))
            x_m, y_m = self._pixel_to_meter(u, v)
            pts.append((float(x_m), float(y_m)))

        walls: List[Tuple[float, float, float, float]] = []
        min_len = max(0.01, float(self.map_geometry_wall_min_segment_length))
        for i in range(4):
            sx, sy = pts[i]
            ex, ey = pts[(i + 1) % 4]
            if self._segment_length(sx, sy, ex, ey) < min_len:
                continue
            walls.append((sx, sy, ex, ey))
        return walls

    def _simplify_wall_loop(
        self,
        points: List[Tuple[float, float]],
    ) -> List[Tuple[float, float, float, float]]:
        if len(points) < 2:
            return []

        min_len = max(0.01, float(self.map_geometry_wall_min_segment_length))
        merge_angle_deg = max(0.1, float(self.map_geometry_wall_merge_angle_deg))
        angle_thr = math.radians(merge_angle_deg)

        loop: List[Tuple[float, float]] = []
        for x, y in points:
            if not loop:
                loop.append((float(x), float(y)))
                continue
            px, py = loop[-1]
            if self._segment_length(px, py, x, y) < 1e-3:
                continue
            loop.append((float(x), float(y)))

        if len(loop) < 2:
            return []

        changed = True
        max_iter = max(8, len(loop) * 2)
        iter_count = 0
        while changed and len(loop) > 3 and iter_count < max_iter:
            changed = False
            iter_count += 1
            n = len(loop)
            next_loop: List[Tuple[float, float]] = []

            for i in range(n):
                prev_pt = loop[(i - 1) % n]
                cur_pt = loop[i]
                next_pt = loop[(i + 1) % n]

                v1x = cur_pt[0] - prev_pt[0]
                v1y = cur_pt[1] - prev_pt[1]
                v2x = next_pt[0] - cur_pt[0]
                v2y = next_pt[1] - cur_pt[1]
                l1 = math.hypot(v1x, v1y)
                l2 = math.hypot(v2x, v2y)

                if l1 < 1e-6 or l2 < 1e-6:
                    changed = True
                    continue

                dot = (v1x * v2x + v1y * v2y) / (l1 * l2)
                dot = max(-1.0, min(1.0, dot))
                turn = math.acos(dot)

                # Remove middle points on nearly straight runs to prevent over-segmentation.
                if turn <= angle_thr and dot > 0.0:
                    changed = True
                    continue

                next_loop.append(cur_pt)

            if len(next_loop) >= 3:
                loop = next_loop
            else:
                break

        walls: List[Tuple[float, float, float, float]] = []
        n = len(loop)
        if n < 2:
            return walls

        for i in range(n):
            sx, sy = loop[i]
            ex, ey = loop[(i + 1) % n]
            if self._segment_length(sx, sy, ex, ey) < min_len:
                continue
            walls.append((float(sx), float(sy), float(ex), float(ey)))

        return walls

    def _get_map_geometry_msgs(self) -> Tuple[List[PolyState], List[WallState]]:
        self.last_sent_map_polygons_count = 0
        self.last_sent_map_walls_count = 0
        return [], []

    def _throttled_log(self, stamp_attr: str, period_sec: float, message: str) -> None:
        now_ns = int(self.get_clock().now().nanoseconds)
        last_ns = int(getattr(self, stamp_attr, 0))
        if now_ns - last_ns < int(max(0.1, period_sec) * 1e9):
            return
        setattr(self, stamp_attr, now_ns)
        self.get_logger().info(message)

    def _transform_point_2d(
        self,
        x: float,
        y: float,
        source_frame: str,
        target_frame: str,
    ) -> Optional[Tuple[float, float]]:
        if not source_frame or not target_frame:
            return None
        if source_frame == target_frame:
            return float(x), float(y)

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=0.05),
            )
        except TransformException:
            return None

        t = tf_msg.transform.translation
        r = tf_msg.transform.rotation
        yaw = self._yaw_from_quaternion(r.x, r.y, r.z, r.w)

        tx = float(t.x)
        ty = float(t.y)
        cx = math.cos(yaw)
        sx = math.sin(yaw)
        x_out = tx + (cx * float(x)) - (sx * float(y))
        y_out = ty + (sx * float(x)) + (cx * float(y))
        return x_out, y_out

    def _point_to_map_frame(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        source_frame = self.pose_frame if self.pose_frame else self.map_frame
        return self._transform_point_2d(x, y, source_frame, self.map_frame)

    def _process_human_msgs(self) -> List[HumanState]:
        if self.latest_humans is None or not self.latest_humans.humans:
            return []

        source_frame = self.latest_humans.header.frame_id if self.latest_humans.header.frame_id else self.map_frame
        humans_map: List[HumanState] = []

        tf_yaw = 0.0
        if source_frame != self.map_frame:
            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=0.05),
                )
            except TransformException as exc:
                self._throttled_log(
                    "_last_human_tf_warn_ns",
                    2.0,
                    "[human_tracking] waiting TF %s -> %s: %s"
                    % (str(source_frame), str(self.map_frame), str(exc)),
                )
                return []

            r = tf_msg.transform.rotation
            tf_yaw = self._yaw_from_quaternion(r.x, r.y, r.z, r.w)

        for src_human in self.latest_humans.humans:
            map_xy = self._transform_point_2d(src_human.px, src_human.py, source_frame, self.map_frame)
            if map_xy is None:
                continue

            cos_yaw = math.cos(tf_yaw)
            sin_yaw = math.sin(tf_yaw)
            vx_map = float(src_human.vx) * cos_yaw - float(src_human.vy) * sin_yaw
            vy_map = float(src_human.vx) * sin_yaw + float(src_human.vy) * cos_yaw

            human_state = HumanState()
            human_state.id = int(src_human.id)
            human_state.px = float(map_xy[0])
            human_state.py = float(map_xy[1])
            human_state.vx = float(vx_map)
            human_state.vy = float(vy_map)
            human_state.radius = float(src_human.radius)
            if hasattr(src_human, "trajectory"):
                human_state.trajectory = src_human.trajectory
            humans_map.append(human_state)

        return humans_map

    def _current_pose_in_map_frame(self) -> Optional[Tuple[float, float]]:
        if self.current_pose is None:
            return None
        px, py, _ = self.current_pose
        return self._point_to_map_frame(px, py)

    def _current_pose_with_yaw_in_map_frame(self) -> Optional[Tuple[float, float, float]]:
        if self.current_pose is None:
            return None

        px, py, yaw = self.current_pose
        source_frame = self.pose_frame if self.pose_frame else self.map_frame
        map_xy = self._transform_point_2d(px, py, source_frame, self.map_frame)
        if map_xy is None:
            return None

        if source_frame == self.map_frame:
            return float(map_xy[0]), float(map_xy[1]), float(yaw)

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.map_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=0.05),
            )
        except TransformException:
            return None

        r = tf_msg.transform.rotation
        tf_yaw = self._yaw_from_quaternion(r.x, r.y, r.z, r.w)
        yaw_map = float(yaw + tf_yaw)
        while yaw_map > math.pi:
            yaw_map -= 2.0 * math.pi
        while yaw_map < -math.pi:
            yaw_map += 2.0 * math.pi

        return float(map_xy[0]), float(map_xy[1]), yaw_map

    def _current_goal_in_map_frame(self) -> Optional[Tuple[float, float]]:
        if self.current_goal is None:
            return None
        gx, gy = self.current_goal
        source_frame = self.current_goal_frame if self.current_goal_frame else self.map_frame
        return self._transform_point_2d(gx, gy, source_frame, self.map_frame)

    def _is_occupied_from_map(self, world_x: float, world_y: float, clearance: float) -> bool:
        if (
            self.map_data is None
            or self.map_resolution is None
            or self.map_width <= 0
            or self.map_height <= 0
        ):
            return False

        mx = int((world_x - self.map_origin_x) / self.map_resolution)
        my = int((world_y - self.map_origin_y) / self.map_resolution)
        if mx < 0 or mx >= self.map_width or my < 0 or my >= self.map_height:
            return True

        inflate_cells = int(math.ceil(max(0.0, clearance) / self.map_resolution))
        for dy in range(-inflate_cells, inflate_cells + 1):
            for dx in range(-inflate_cells, inflate_cells + 1):
                if dx * dx + dy * dy > inflate_cells * inflate_cells:
                    continue
                cx = mx + dx
                cy = my + dy
                if cx < 0 or cx >= self.map_width or cy < 0 or cy >= self.map_height:
                    return True
                idx = cy * self.map_width + cx
                occ = self.map_data[idx]
                if occ >= self.map_occupancy_threshold:
                    return True
        return False

    def _scan_only_obstacle_tuples(self) -> List[Tuple[float, float, float]]:
        obstacles = []
        if self.latest_scan is None or self.current_pose is None:
            return obstacles

        px, py, yaw = self.current_pose
        pose_map = self._current_pose_in_map_frame()
        if pose_map is None:
            return obstacles
        px_map, py_map = pose_map
        x_lim = min(self.effective_half_width, self.planner_x_limit)
        y_lim = min(self.effective_half_height, self.planner_y_limit)
        min_safe_range = self.robot_radius + 0.12
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
            if r < min_safe_range:
                continue

            ang = yaw + angle_min + i * angle_inc
            ox = px + r * math.cos(ang)
            oy = py + r * math.sin(ang)

            map_xy = self._point_to_map_frame(ox, oy)
            if map_xy is None:
                continue
            ox, oy = map_xy

            if abs(ox - px_map) >= x_lim or abs(oy - py_map) >= y_lim:
                continue
            obstacles.append((ox, oy, self.obstacle_radius))

        return obstacles

    def _scan_to_obstacle_tuples(self) -> List[Tuple[float, float, float]]:
        obstacles = self._scan_only_obstacle_tuples()
        obstacles.extend(self._map_to_obstacle_tuples())

        return obstacles

    def _map_to_obstacle_tuples(self) -> List[Tuple[float, float, float]]:
        if not self.use_map_static_obstacles:
            self.map_static_circles = []
            return []
        if self.current_pose is None:
            self.map_static_circles = []
            return []
        if (
            self.map_data is None
            or self.map_resolution is None
            or self.map_width <= 0
            or self.map_height <= 0
        ):
            self.map_static_circles = []
            return []

        pose_map = self._current_pose_in_map_frame()
        if pose_map is None:
            self.map_static_circles = []
            return []
        px, py = pose_map
        x_lim = min(self.effective_half_width, self.planner_x_limit)
        y_lim = min(self.effective_half_height, self.planner_y_limit)
        cell_step = max(1, int(round(max(0.05, self.map_static_sample_step_m) / self.map_resolution)))
        rx = int((px - self.map_origin_x) / self.map_resolution)
        ry = int((py - self.map_origin_y) / self.map_resolution)
        wx = int(x_lim / self.map_resolution)
        wy = int(y_lim / self.map_resolution)

        x0 = max(0, rx - wx)
        x1 = min(self.map_width - 1, rx + wx)
        y0 = max(0, ry - wy)
        y1 = min(self.map_height - 1, ry + wy)

        static_obs: List[Tuple[float, float, float]] = []
        for my in range(y0, y1 + 1, cell_step):
            for mx in range(x0, x1 + 1, cell_step):
                idx = my * self.map_width + mx
                occ = self.map_data[idx]
                if occ < self.map_occupancy_threshold:
                    continue
                ox = self.map_origin_x + (mx + 0.5) * self.map_resolution
                oy = self.map_origin_y + (my + 0.5) * self.map_resolution
                if abs(ox - px) >= x_lim or abs(oy - py) >= y_lim:
                    continue
                # avoid flooding with cells under robot footprint
                if math.hypot(ox - px, oy - py) < (self.robot_radius + 0.05):
                    continue
                static_obs.append((ox, oy, max(0.05, self.map_static_obstacle_radius)))

        if len(static_obs) > max(1, self.map_static_obstacle_limit):
            static_obs.sort(key=lambda p: math.hypot(p[0] - px, p[1] - py))
            static_obs = static_obs[: max(1, self.map_static_obstacle_limit)]

        self.map_static_circles = list(static_obs)

        return static_obs

    @staticmethod
    def _rotate_to_world(px: float, py: float, yaw: float, lx: float, ly: float) -> Tuple[float, float]:
        wx = px + lx * math.cos(yaw) - ly * math.sin(yaw)
        wy = py + lx * math.sin(yaw) + ly * math.cos(yaw)
        return wx, wy

    def _generate_action_candidates(self) -> List[Tuple[float, float, float, float]]:
        """Returns candidates as (local_x, local_y, world_x, world_y)."""
        if self.current_pose is None:
            return []

        px, py, yaw = self.current_pose
        dim = max(2, int(self.action_dim))
        step = (2.0 * self.action_range) / float(dim - 1)

        candidates = []
        for iy in range(dim):
            ly = -self.action_range + step * iy
            for ix in range(dim):
                lx = -self.action_range + step * ix
                wx, wy = self._rotate_to_world(px, py, yaw, lx, ly)
                candidates.append((lx, ly, wx, wy))
        return candidates

    def _is_action_masked(
        self,
        local_x: float,
        local_y: float,
        world_x: float,
        world_y: float,
        obstacles: List[Tuple[float, float, float]],
    ) -> bool:
        if abs(local_x) > self.planner_half_width or abs(local_y) > self.planner_half_height:
            return True

        map_xy = self._point_to_map_frame(world_x, world_y)
        if map_xy is None:
            return False
        map_x, map_y = map_xy

        if self.map_bounds is not None:
            xmin, xmax, ymin, ymax = self.map_bounds
            if map_x < xmin or map_x > xmax or map_y < ymin or map_y > ymax:
                return True

        inflate = self.robot_radius + self.action_mask_clearance
        if self._is_occupied_from_map(map_x, map_y, inflate):
            return True

        for ox, oy, radius in obstacles:
            # Obstacles are represented in map frame; compare in map frame.
            if math.hypot(map_x - ox, map_y - oy) < (inflate + radius):
                return True

        return False

    def _apply_action_mask_to_sub_goal(
        self,
        sub_goal_map: Tuple[float, float],
    ) -> Tuple[Tuple[float, float], bool, float]:
        """Project sub-goal onto the nearest valid discrete action in map frame."""
        candidates = self._generate_action_candidates()
        if not candidates:
            return sub_goal_map, False, 0.0

        dynamic_obstacles = self._scan_only_obstacle_tuples()
        static_obstacles = self._map_to_obstacle_tuples()
        obstacles = dynamic_obstacles + static_obstacles

        best_valid: Optional[Tuple[float, float]] = None
        best_dist = float("inf")

        for lx, ly, wx, wy in candidates:
            if self._is_action_masked(lx, ly, wx, wy, obstacles):
                continue
            map_xy = self._point_to_map_frame(wx, wy)
            if map_xy is None:
                continue
            cx, cy = map_xy
            dist = math.hypot(float(sub_goal_map[0]) - cx, float(sub_goal_map[1]) - cy)
            if dist < best_dist:
                best_dist = dist
                best_valid = (float(cx), float(cy))

        if best_valid is None:
            robot_map = self._current_pose_in_map_frame()
            if robot_map is not None:
                fallback = (float(robot_map[0]), float(robot_map[1]))
                snap_dist = math.hypot(float(sub_goal_map[0]) - fallback[0], float(sub_goal_map[1]) - fallback[1])
                return fallback, True, float(snap_dist)
            return sub_goal_map, True, 0.0

        snap_dist = math.hypot(float(sub_goal_map[0]) - best_valid[0], float(sub_goal_map[1]) - best_valid[1])
        applied = bool(snap_dist > 1e-3)
        return best_valid, applied, float(snap_dist)

    def _publish_action_visualization(self, selected_sub_goal: Optional[Tuple[float, float]]) -> None:
        if self.current_pose is None:
            return
        candidates = self._generate_action_candidates()
        dynamic_obstacles = self._scan_only_obstacle_tuples()
        static_obstacles = self._map_to_obstacle_tuples()
        obstacles = dynamic_obstacles + static_obstacles

        valid_points: List[Tuple[float, float]] = []
        masked_points: List[Tuple[float, float]] = []
        closest_selected: Optional[Tuple[float, float]] = None
        best_dist = float("inf")

        for lx, ly, wx, wy in candidates:
            if self._is_action_masked(lx, ly, wx, wy, obstacles):
                masked_points.append((float(wx), float(wy)))
            else:
                valid_points.append((float(wx), float(wy)))

            if selected_sub_goal is not None:
                cand_map = self._point_to_map_frame(wx, wy)
                if cand_map is None:
                    continue
                dist = math.hypot(selected_sub_goal[0] - cand_map[0], selected_sub_goal[1] - cand_map[1])
                if dist < best_dist:
                    best_dist = dist
                    closest_selected = (wx, wy)

        if self.publish_policy_debug_status:
            status = {
                "action_dim": int(self.action_dim),
                "candidate_count": len(candidates),
                "expected_count": int(self.action_dim) * int(self.action_dim),
                "valid_count": len(valid_points),
                "masked_count": len(masked_points),
                "dynamic_count": len(dynamic_obstacles),
                "static_count": len(static_obstacles),
                "map_walls_count": len(self.map_geometry_walls),
                "map_polygons_count": len(self.map_geometry_polygons),
                "map_local_walls_count": int(self.last_sent_map_walls_count),
                "map_local_polygons_count": int(self.last_sent_map_polygons_count),
                "planner_fail_streak": int(self.consecutive_planner_failures),
                "policy_hz": float(1.0 / self.policy_period),
                "planner_service_hz": float(1.0 / self.mpc_period),
                "goal_source_mode": str(self.goal_source_mode),
                "selected_sub_goal": list(selected_sub_goal) if selected_sub_goal is not None else None,
                "cached_sub_goal": list(self.latest_sub_goal) if self.latest_sub_goal is not None else None,
                "cached_sub_goal_seq": int(self.latest_sub_goal_seq),
                "goal_reached": bool(self.goal_reached),
                "mask_applied": bool(self.last_mask_applied),
                "mask_snap_distance": float(self.last_mask_snap_distance),
            }
            msg = String()
            msg.data = json.dumps(status)
            self.policy_debug_status_pub.publish(msg)

        action_payload = {
            "frame_id": self.action_marker_frame,
            "goal_reached": bool(self.goal_reached),
            "current_pose": [float(self.current_pose[0]), float(self.current_pose[1]), float(self.current_pose[2])],
            "valid_points": valid_points,
            "masked_points": masked_points,
            "selected_point": [float(closest_selected[0]), float(closest_selected[1])] if closest_selected is not None else None,
            "candidate_count": len(candidates),
            "valid_count": len(valid_points),
            "masked_count": len(masked_points),
        }
        msg = String()
        msg.data = json.dumps(action_payload)
        self.action_debug_pub.publish(msg)

    def _scan_to_obstacle_msgs(self) -> List[ObstacleState]:
        # Use both lidar and map-grid circles, then keep only nearest N for ACADOS.
        obs_tuples = self._scan_to_obstacle_tuples()
        pose_map = self._current_pose_in_map_frame()
        if pose_map is not None:
            px, py = pose_map
            obs_tuples.sort(key=lambda o: math.hypot(float(o[0]) - px, float(o[1]) - py))

        # Keep only nearest obstacles to match ACADOS parameter capacity.
        max_supported = 40
        keep_n = min(max_supported, max(1, int(self.map_static_obstacle_limit)))
        obs_tuples = obs_tuples[:keep_n]

        obstacles = []
        for obs_x, obs_y, radius in obs_tuples:
            obstacle = ObstacleState()
            obstacle.px = float(obs_x)
            obstacle.py = float(obs_y)
            obstacle.radius = float(radius)
            obstacles.append(obstacle)
        return obstacles

    def _build_human_fallback_msgs(self, obstacle_msgs: List[ObstacleState]) -> List[HumanState]:
        if not self.human_fallback_enabled:
            return []
        if self.current_pose is None:
            return []

        px, py, _ = self.current_pose
        limit = max(0, int(self.human_fallback_limit))
        if limit == 0:
            return []

        obs_sorted = sorted(
            obstacle_msgs,
            key=lambda o: math.hypot(float(o.px) - px, float(o.py) - py),
        )
        humans: List[HumanState] = []
        for o in obs_sorted[:limit]:
            h = HumanState()
            h.px = float(o.px)
            h.py = float(o.py)
            h.vx = 0.0
            h.vy = 0.0
            h.radius = float(max(0.05, self.human_fallback_radius))
            humans.append(h)
        return humans

    def _wheel_speeds_from_twist(self) -> Optional[Tuple[float, float]]:
        if self.current_twist is None:
            return None

        v, w = self.current_twist
        v_left = v - self.axle_half_width * w
        v_right = v + self.axle_half_width * w
        return v_left, v_right

    def _build_policy_worker_request(self) -> Optional[dict]:
        if self.current_pose is None or self.current_twist is None or self.current_goal is None:
            return None

        pose_map = self._current_pose_with_yaw_in_map_frame()
        goal_map = self._current_goal_in_map_frame()
        if pose_map is None or goal_map is None:
            return None

        px, py, yaw = pose_map
        gx, gy = goal_map

        wheel_speeds = self._wheel_speeds_from_twist()
        if wheel_speeds is None:
            return None
        v_left, v_right = wheel_speeds

        return {
            "px": px,
            "py": py,
            "yaw": yaw,
            "v_left": v_left,
            "v_right": v_right,
            "gx": gx,
            "gy": gy,
            "obstacles": self._scan_to_obstacle_tuples(),
            "walls": [],
        }

    def _send_planner_request(self, sub_goal: Tuple[float, float]) -> None:
        if self.current_pose is None or self.current_twist is None or self.current_goal is None:
            return

        if not self.ocp_client.wait_for_service(timeout_sec=0.01):
            self.get_logger().warn(f"Service {self.planner_service} not available")
            return

        pose_map = self._current_pose_with_yaw_in_map_frame()
        goal_map = self._current_goal_in_map_frame()
        if pose_map is None or goal_map is None:
            self._throttled_log(
                "_last_request_wait_tf_log_ns",
                2.0,
                "[planner_request] waiting TF for pose/goal -> map (pose_frame=%s goal_frame=%s map_frame=%s)"
                % (str(self.pose_frame), str(self.current_goal_frame), str(self.map_frame)),
            )
            return

        px, py, yaw = pose_map
        gx, gy = goal_map
        wheel_speeds = self._wheel_speeds_from_twist()
        if wheel_speeds is None:
            return
        v_left, v_right = wheel_speeds

        req = OcpLocalPlann.Request()
        req.ob.robot_state.pose.x = float(px)
        req.ob.robot_state.pose.y = float(py)
        req.ob.robot_state.pose.theta = float(yaw)
        req.ob.robot_state.vl = float(v_left)
        req.ob.robot_state.vr = float(v_right)
        req.ob.robot_state.gx = float(gx)
        req.ob.robot_state.gy = float(gy)
        req.ob.robot_state.v_pref = float(self.v_pref)
        req.ob.robot_state.radius = float(self.axle_half_width)

        req.ob.obstacle_states = self._scan_to_obstacle_msgs()
        poly_states, walls = self._get_map_geometry_msgs()
        req.ob.walls = walls
        req.ob.poly_states = poly_states
        req.ob.human_states = self._process_human_msgs() if self.enforce_scene_entities else []
        req.ob.header.stamp = self.get_clock().now().to_msg()
        req.ob.header.frame_id = "map"

        if self.publish_debug_joint_state:
            self.debug_joint_state_pub.publish(req.ob)

        req.sub_goal.x = float(sub_goal[0])
        req.sub_goal.y = float(sub_goal[1])

        self._publish_local_goal(sub_goal)
        self.last_wheel_speed = (v_left, v_right)
        self.last_request_for_viz = req

        future = self.ocp_client.call_async(req)
        future.add_done_callback(self._on_planner_response)
        self.pending_future = future

    def _publish_planner_scene_debug(
        self,
        req: OcpLocalPlann.Request,
        astar_path: List[Tuple[float, float]],
    ) -> None:
        x_lim = min(self.effective_half_width, self.planner_x_limit)
        y_lim = min(self.effective_half_height, self.planner_y_limit)
        payload = {
            "frame_id": self.planner_scene_frame,
            "goal_reached": bool(self.goal_reached),
            "bounds": {
                "xmin": float(-x_lim),
                "xmax": float(x_lim),
                "ymin": float(-y_lim),
                "ymax": float(y_lim),
            },
            "robot": {
                "x": float(req.ob.robot_state.pose.x),
                "y": float(req.ob.robot_state.pose.y),
                "theta": float(req.ob.robot_state.pose.theta),
                "radius": float(self.robot_radius),
            },
            "goal": {
                "x": float(req.ob.robot_state.gx),
                "y": float(req.ob.robot_state.gy),
            },
            "sub_goal": {
                "x": float(req.sub_goal.x),
                "y": float(req.sub_goal.y),
            },
            "walls": [
                [float(w.sx), float(w.sy), float(w.ex), float(w.ey)] for w in req.ob.walls
            ],
            "obstacles": [
                [float(o.px), float(o.py), float(o.radius)] for o in req.ob.obstacle_states
            ],
            "polygons": [
                [[float(v.x), float(v.y)] for v in poly.vertices] for poly in req.ob.poly_states
            ],
            "humans": [
                {
                    "px": float(h.px),
                    "py": float(h.py),
                    "vx": float(h.vx),
                    "vy": float(h.vy),
                    "radius": float(h.radius),
                }
                for h in req.ob.human_states
            ],
            "trajectory": [[float(px), float(py)] for px, py in astar_path],
            "path_debug": {
                "astar_points": int(len(astar_path)),
                "goal_source_mode": str(self.goal_source_mode),
                "tracking_controller": "nav2_mppi_controller::MPPIController",
            },
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.planner_scene_debug_pub.publish(msg)

    def _on_planner_response(self, future) -> None:
        self.pending_future = None
        try:
            res = future.result()
        except Exception as exc:
            self.get_logger().error(f"Planner service call failed: {exc}")
            return

        if res.success:
            self.consecutive_planner_failures = 0
            self.success_streak += 1
        else:
            self.success_streak = 0
            self.consecutive_planner_failures += 1
            if self.consecutive_planner_failures % 5 == 1:
                self.get_logger().warn(
                    "Planner reported failure (count=%d, map_walls=%d, map_polygons=%d)"
                    % (
                        int(self.consecutive_planner_failures),
                        int(len(self.map_geometry_walls)),
                        int(len(self.map_geometry_polygons)),
                )
                )

        astar_path = self._path_points_from_response(res)
        self.last_astar_path_for_viz = list(astar_path)
        if self.last_request_for_viz is not None:
            self._publish_planner_scene_debug(self.last_request_for_viz, astar_path)

        if res.success:
            path_msg = self._build_path_msg(res.astar_path, self.map_frame)
            self.local_path_pub.publish(path_msg)
            self._send_follow_path_goal(path_msg)

    def _policy_loop(self) -> None:
        if self.goal_source_mode == self._GOAL_SOURCE_RL and self.worker is None:
            return

        if self.current_pose is None or self.current_goal is None:
            self._throttled_log(
                "_last_policy_wait_log_ns",
                2.0,
                "[policy_loop] waiting pose/goal (pose=%s goal=%s)"
                % (str(self.current_pose is not None), str(self.current_goal is not None)),
            )
            return

        if self.map_data is None:
            self._throttled_log(
                "_last_policy_wait_map_log_ns",
                2.0,
                "[policy_loop] waiting map data on topic %s (grid sampling not started yet)"
                % str(self.map_topic),
            )
            return

        pose_map = self._current_pose_in_map_frame()
        goal_map = self._current_goal_in_map_frame()
        if pose_map is None or goal_map is None:
            self._throttled_log(
                "_last_policy_wait_tf_log_ns",
                2.0,
                "[policy_loop] waiting TF for pose/goal -> map",
            )
            return

        px, py = pose_map
        gx, gy = goal_map
        if math.hypot(gx - px, gy - py) < self.goal_tolerance:
            self.goal_reached = True
            self.latest_sub_goal = None
            self.last_astar_path_for_viz = []
            self._cancel_follow_path_goal()
            self._publish_action_visualization(None)
            return

        self.goal_reached = False

        if self.goal_source_mode == self._GOAL_SOURCE_RVIZ:
            self.latest_sub_goal = (float(gx), float(gy))
            self.latest_sub_goal_seq += 1
            self.last_mask_applied = False
            self.last_mask_snap_distance = 0.0
            self._publish_action_visualization(self.latest_sub_goal)
            return

        try:
            worker_req = self._build_policy_worker_request()
            if worker_req is None:
                self._publish_action_visualization(None)
                return
            self._submit_policy_request(worker_req)
            latest_policy_result = self._take_latest_policy_result()
            if latest_policy_result is None:
                if self.latest_sub_goal is not None:
                    self._publish_action_visualization(self.latest_sub_goal)
                self._throttled_log(
                    "_last_policy_worker_wait_result_log_ns",
                    2.0,
                    "[policy_loop] waiting non-blocking response from policy worker",
                )
                return
            sub_goal, _ = latest_policy_result
            masked_sub_goal, mask_applied, snap_dist = self._apply_action_mask_to_sub_goal(sub_goal)
            self.last_mask_applied = bool(mask_applied)
            self.last_mask_snap_distance = float(snap_dist)
            self.latest_sub_goal = masked_sub_goal
            self.latest_sub_goal_seq += 1
            self._publish_action_visualization(masked_sub_goal)
        except Exception as exc:
            self.get_logger().error(f"Policy loop failed: {exc}")

    def _planner_loop(self) -> None:
        if self.current_pose is None or self.current_goal is None:
            self._throttled_log(
                "_last_planner_wait_log_ns",
                2.0,
                "[planner_loop] waiting pose/goal (pose=%s goal=%s)"
                % (str(self.current_pose is not None), str(self.current_goal is not None)),
            )
            return

        if self.map_data is None:
            self._throttled_log(
                "_last_planner_wait_map_log_ns",
                2.0,
                "[planner_loop] waiting map data on topic %s"
                % str(self.map_topic),
            )
            return

        pose_map_with_yaw = self._current_pose_with_yaw_in_map_frame()
        goal_map = self._current_goal_in_map_frame()
        if pose_map_with_yaw is None or goal_map is None:
            self._throttled_log(
                "_last_planner_wait_tf_log_ns",
                2.0,
                "[planner_loop] waiting TF for pose/goal -> map",
            )
            return

        px, py, _ = pose_map_with_yaw
        gx, gy = goal_map
        if math.hypot(gx - px, gy - py) < self.goal_tolerance:
            self.goal_reached = True
            # In cascaded mode, keep sending state updates so the C++ planner can
            # apply planner.goal_stop_distance and hold cmd_vel at zero near goal.
            self.last_sent_follow_path = None
            self.pending_follow_path_msg = None
            self._cancel_follow_path_goal()
            if self.pending_future is None:
                try:
                    self._send_planner_request((px, py))
                except Exception as exc:
                    self.get_logger().error(f"Planner goal-stop update failed: {exc}")
            return

        self.goal_reached = False
        if self.pending_future is not None:
            return

        if self.goal_source_mode == self._GOAL_SOURCE_RVIZ:
            try:
                self._send_planner_request((float(gx), float(gy)))
            except Exception as exc:
                self.get_logger().error(f"Planner loop failed (rviz_global_goal mode): {exc}")
            return

        if self.latest_sub_goal is None:
            self._throttled_log(
                "_last_planner_wait_subgoal_log_ns",
                2.0,
                "[planner_loop] waiting latest_sub_goal from policy worker",
            )
            return

        try:
            self._send_planner_request(self.latest_sub_goal)
            self.last_requested_sub_goal_seq = self.latest_sub_goal_seq
        except Exception as exc:
            self.get_logger().error(f"Planner loop failed: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = RlOcpPolicyBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.worker_stop_event.set()
        with node.worker_request_cond:
            node.worker_request_cond.notify_all()
        if node.worker is not None and node.worker.poll() is None:
            node.worker.terminate()
            try:
                node.worker.wait(timeout=2.0)
            except Exception:
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
