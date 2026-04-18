#!/usr/bin/python3
import json
import math
import os
import shlex
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import rclpy
from interfaces.msg import ObstacleState, WallState, PolyState, Point as PolyPoint
from interfaces.msg import JointState as InterfaceJointState
from interfaces.srv import OcpLocalPlann
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class RlOcpPolicyBridge(Node):
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
        self.declare_parameter("timer_period", 0.1)
        self.declare_parameter("policy_period", 0.2)
        self.declare_parameter("mpc_period", 0.05)
        self.declare_parameter("sync_mpc_to_policy", True)
        self.declare_parameter("control_dt", 0.25)
        self.declare_parameter("mpc_horizon_steps", 10)
        self.declare_parameter("mpc_stability_warn_threshold", 0.35)
        self.declare_parameter("max_linear_speed", 1.0)
        self.declare_parameter("max_angular_speed", 3.0)

        self.declare_parameter("planner_half_width", 5.8)
        self.declare_parameter("planner_half_height", 9.8)
        self.declare_parameter("planner_hard_half_width", 5.99)
        self.declare_parameter("planner_hard_half_height", 9.99)
        self.declare_parameter("use_wall_constraints", True)
        self.declare_parameter("use_demo_poly_obstacle", True)
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
        self.declare_parameter("cmd_topic", "/diff_cont/cmd_vel")
        self.declare_parameter("planner_service", "/ocp_plann")

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
        self.policy_period = max(0.001, self.get_parameter("policy_period").get_parameter_value().double_value)
        self.mpc_period = max(0.001, self.get_parameter("mpc_period").get_parameter_value().double_value)
        self.sync_mpc_to_policy = self.get_parameter("sync_mpc_to_policy").get_parameter_value().bool_value
        self.control_dt = self.get_parameter("control_dt").get_parameter_value().double_value
        self.mpc_horizon_steps = max(1, self.get_parameter("mpc_horizon_steps").get_parameter_value().integer_value)
        self.mpc_stability_warn_threshold = max(
            0.0,
            self.get_parameter("mpc_stability_warn_threshold").get_parameter_value().double_value,
        )
        self.max_linear_speed = self.get_parameter("max_linear_speed").get_parameter_value().double_value
        self.max_angular_speed = self.get_parameter("max_angular_speed").get_parameter_value().double_value
        self.planner_half_width = self.get_parameter("planner_half_width").get_parameter_value().double_value
        self.planner_half_height = self.get_parameter("planner_half_height").get_parameter_value().double_value
        self.planner_hard_half_width = self.get_parameter("planner_hard_half_width").get_parameter_value().double_value
        self.planner_hard_half_height = self.get_parameter("planner_hard_half_height").get_parameter_value().double_value
        self.effective_half_width = min(self.planner_half_width, self.planner_hard_half_width)
        self.effective_half_height = min(self.planner_half_height, self.planner_hard_half_height)
        self.use_wall_constraints = self.get_parameter("use_wall_constraints").get_parameter_value().bool_value
        self.use_demo_poly_obstacle = self.get_parameter("use_demo_poly_obstacle").get_parameter_value().bool_value
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
        self.cmd_topic = self.get_parameter("cmd_topic").get_parameter_value().string_value
        self.planner_service = self.get_parameter("planner_service").get_parameter_value().string_value

        self.current_pose: Optional[Tuple[float, float, float]] = None
        self.current_twist: Optional[Tuple[float, float]] = None
        self.current_goal: Optional[Tuple[float, float]] = None
        self.latest_scan: Optional[LaserScan] = None
        self.map_bounds: Optional[Tuple[float, float, float, float]] = None

        # Keep Python-side request geometry aligned with hard-coded C++ planner map limits.
        self.planner_x_limit = 5.5
        self.planner_y_limit = 9.8

        self.pending_future = None
        self.last_wheel_speed: Optional[Tuple[float, float]] = None
        self.latest_sub_goal: Optional[Tuple[float, float]] = None
        self.latest_sub_goal_seq = 0
        self.last_requested_sub_goal_seq = -1
        self.worker: Optional[subprocess.Popen] = None
        self.last_request_for_viz: Optional[OcpLocalPlann.Request] = None
        self.last_pose_for_viz: Optional[Tuple[float, float, float]] = None
        self.last_wheels_for_viz: Optional[Tuple[float, float]] = None
        self.consecutive_planner_failures = 0
        self.runtime_use_walls = self.use_wall_constraints
        self.runtime_use_poly = self.use_demo_poly_obstacle
        self.success_streak = 0
        self.goal_reached = False
        self.prev_predicted_traj: Optional[List[Tuple[float, float]]] = None
        self.last_stability_delta_rms = 0.0

        self._start_policy_worker()

        self.create_subscription(Odometry, self.odom_topic, self._odom_callback, 10)
        self.create_subscription(LaserScan, self.scan_topic, self._scan_callback, 10)
        self.create_subscription(PoseStamped, self.goal_topic, self._goal_callback, 10)
        self.create_subscription(OccupancyGrid, self.map_topic, self._map_callback, 10)

        self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_topic, 10)
        self.debug_joint_state_pub = self.create_publisher(InterfaceJointState, self.debug_joint_state_topic, 10)
        self.policy_debug_status_pub = self.create_publisher(String, self.policy_debug_status_topic, 10)
        self.action_debug_pub = self.create_publisher(String, self.action_debug_topic, 10)
        self.planner_scene_debug_pub = self.create_publisher(String, self.planner_scene_debug_topic, 10)
        self.ocp_client = self.create_client(OcpLocalPlann, self.planner_service)
        self.add_on_set_parameters_callback(self._on_parameters_changed)

        # Keep legacy timer_period declared for backward compatibility,
        # but run policy and MPC loops on dedicated timers.
        self.create_timer(self.policy_period, self._policy_loop)
        self.create_timer(self.mpc_period, self._planner_loop)

        self.get_logger().info("RL OCP policy bridge started")
        self.get_logger().info(f"Model: {self.model_path}")
        self.get_logger().info(f"Service: {self.planner_service}")
        self.get_logger().info(
            f"Loop rates: policy={1.0 / self.policy_period:.2f} Hz, mpc={1.0 / self.mpc_period:.2f} Hz"
        )
        self.get_logger().info(f"Sync MPC requests to policy updates: {self.sync_mpc_to_policy}")
        self.get_logger().info(f"Action visualization: {self.visualize_actions} ({self.action_marker_topic})")
        self.get_logger().info(
            f"JointState debug topic: {self.publish_debug_joint_state} ({self.debug_joint_state_topic})"
        )
        self.get_logger().info(
            f"Policy status topic: {self.publish_policy_debug_status} ({self.policy_debug_status_topic})"
        )

    def _on_parameters_changed(self, params):
        update_effective_bounds = False
        for p in params:
            if p.name == "use_demo_poly_obstacle":
                self.use_demo_poly_obstacle = bool(p.value)
                self.runtime_use_poly = self.use_demo_poly_obstacle
            elif p.name == "use_wall_constraints":
                self.use_wall_constraints = bool(p.value)
                self.runtime_use_walls = self.use_wall_constraints
            elif p.name == "visualize_actions":
                self.visualize_actions = bool(p.value)
            elif p.name == "publish_debug_joint_state":
                self.publish_debug_joint_state = bool(p.value)
            elif p.name == "publish_policy_debug_status":
                self.publish_policy_debug_status = bool(p.value)
            elif p.name == "visualize_planner_scene":
                self.visualize_planner_scene = bool(p.value)
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
            elif p.name == "mpc_horizon_steps":
                self.mpc_horizon_steps = max(1, int(p.value))
            elif p.name == "mpc_stability_warn_threshold":
                self.mpc_stability_warn_threshold = max(0.0, float(p.value))

        if update_effective_bounds:
            self.effective_half_width = min(self.planner_half_width, self.planner_hard_half_width)
            self.effective_half_height = min(self.planner_half_height, self.planner_hard_half_height)

        return SetParametersResult(successful=True)

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

    def _query_policy_worker(self, subgoal_req: dict) -> Optional[Tuple[float, float]]:
        if self.worker is None:
            return None
        if self.worker.poll() is not None:
            err = ""
            if self.worker.stderr is not None:
                err = self.worker.stderr.read().strip()
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

    def _build_wall_tuples(self) -> List[Tuple[float, float, float, float]]:
        if self.map_bounds is None:
            if self.current_pose is None:
                xmin, xmax, ymin, ymax = -10.0, 10.0, -10.0, 10.0
            else:
                px, py, _ = self.current_pose
                xmin, xmax, ymin, ymax = px - 10.0, px + 10.0, py - 10.0, py + 10.0
        else:
            xmin, xmax, ymin, ymax = self.map_bounds

        return [
            (xmin, ymin, xmin, ymax),
            (xmax, ymin, xmax, ymax),
            (xmin, ymin, xmax, ymin),
            (xmin, ymax, xmax, ymax),
        ]

    def _build_wall_msgs(self) -> List[WallState]:
        if not self.use_wall_constraints:
            return []

        # Intersect optional map bounds with planner limits to avoid ClipWall errors.
        x_lim = min(self.effective_half_width, self.planner_x_limit)
        y_lim = min(self.effective_half_height, self.planner_y_limit)

        xmin, xmax = -x_lim, x_lim
        ymin, ymax = -y_lim, y_lim
        if self.map_bounds is not None:
            map_xmin, map_xmax, map_ymin, map_ymax = self.map_bounds
            xmin = max(xmin, map_xmin)
            xmax = min(xmax, map_xmax)
            ymin = max(ymin, map_ymin)
            ymax = min(ymax, map_ymax)

        # Fallback to planner box if map intersection is degenerate.
        if xmin >= xmax or ymin >= ymax:
            xmin, xmax = -x_lim, x_lim
            ymin, ymax = -y_lim, y_lim

        eps = 1e-3
        # MPC expects exactly 2 vertical walls with specific orientation:
        # left wall: top -> bottom, right wall: bottom -> top.
        wall_segments = [
            (xmin, ymax, xmin, ymin),
            (xmax, ymin, xmax, ymax),
        ]

        walls = []
        for sx, sy, ex, ey in wall_segments:
            wall = WallState()
            wall.sx = self._clamp(float(sx), -x_lim + eps, x_lim - eps)
            wall.sy = self._clamp(float(sy), -y_lim + eps, y_lim - eps)
            wall.ex = self._clamp(float(ex), -x_lim + eps, x_lim - eps)
            wall.ey = self._clamp(float(ey), -y_lim + eps, y_lim - eps)
            walls.append(wall)

        return walls

    def _build_poly_msgs(self) -> List[PolyState]:
        """Optional demo polygon obstacle for debugging planner constraints."""
        if not self.use_demo_poly_obstacle:
            return []

        # Build one small square near a corner, avoiding the robot/goal corridor.
        x_lim = min(self.effective_half_width, self.planner_x_limit)
        y_lim = min(self.effective_half_height, self.planner_y_limit)
        margin = 0.9
        size = 0.8

        cx = x_lim - margin
        cy = y_lim - margin
        if self.current_pose is not None:
            px, py, _ = self.current_pose
            if math.hypot(cx - px, cy - py) < 1.5:
                cy = -y_lim + margin

        if self.current_goal is not None:
            gx, gy = self.current_goal
            if math.hypot(cx - gx, cy - gy) < 1.5:
                cx = -x_lim + margin

        poly = PolyState()
        poly.is_clockwise = True
        poly.vertices = [
            PolyPoint(x=cx - size, y=cy - size),
            PolyPoint(x=cx + size, y=cy - size),
            PolyPoint(x=cx + size, y=cy + size),
            PolyPoint(x=cx - size, y=cy + size),
        ]

        return [poly]

    def _scan_to_obstacle_tuples(self) -> List[Tuple[float, float, float]]:
        obstacles = []
        if self.latest_scan is None or self.current_pose is None:
            return obstacles

        px, py, yaw = self.current_pose
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
            if abs(ox) >= x_lim or abs(oy) >= y_lim:
                continue
            obstacles.append((ox, oy, self.obstacle_radius))

        return obstacles

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

        if self.map_bounds is not None:
            xmin, xmax, ymin, ymax = self.map_bounds
            if world_x < xmin or world_x > xmax or world_y < ymin or world_y > ymax:
                return True

        inflate = self.robot_radius + self.action_mask_clearance
        for ox, oy, radius in obstacles:
            if math.hypot(world_x - ox, world_y - oy) < (inflate + radius):
                return True

        return False

    def _publish_action_visualization(self, selected_sub_goal: Optional[Tuple[float, float]]) -> None:
        if self.current_pose is None:
            return
        candidates = self._generate_action_candidates()
        obstacles = self._scan_to_obstacle_tuples()

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
                dist = math.hypot(selected_sub_goal[0] - wx, selected_sub_goal[1] - wy)
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
                "walls_enabled": bool(self.runtime_use_walls),
                "poly_enabled": bool(self.runtime_use_poly),
                "planner_fail_streak": int(self.consecutive_planner_failures),
                "policy_hz": float(1.0 / self.policy_period),
                "mpc_hz": float(1.0 / self.mpc_period),
                "selected_sub_goal": list(selected_sub_goal) if selected_sub_goal is not None else None,
                "cached_sub_goal": list(self.latest_sub_goal) if self.latest_sub_goal is not None else None,
                "cached_sub_goal_seq": int(self.latest_sub_goal_seq),
                "goal_reached": bool(self.goal_reached),
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
        obstacles = []
        if self.latest_scan is None or self.current_pose is None:
            return obstacles

        px, py, yaw = self.current_pose
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
            obs_x = px + r * math.cos(ang)
            obs_y = py + r * math.sin(ang)
            if abs(obs_x - px) > x_lim or abs(obs_y - py) > y_lim:
                continue
            if abs(obs_x) >= x_lim or abs(obs_y) >= y_lim:
                continue
            obstacle = ObstacleState()
            obstacle.px = float(obs_x)
            obstacle.py = float(obs_y)
            obstacle.radius = float(self.obstacle_radius)
            obstacles.append(obstacle)

        return obstacles

    def _publish_stop(self) -> None:
        if not rclpy.ok():
            return
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = 0.0
        msg.twist.angular.z = 0.0
        self.cmd_pub.publish(msg)

    def _wheel_speeds_from_twist(self) -> Optional[Tuple[float, float]]:
        if self.current_twist is None:
            return None

        v, w = self.current_twist
        v_left = v - self.axle_half_width * w
        v_right = v + self.axle_half_width * w
        return v_left, v_right

    def _get_sub_goal_from_policy(self) -> Optional[Tuple[float, float]]:
        if self.current_pose is None or self.current_twist is None or self.current_goal is None:
            return None

        px, py, yaw = self.current_pose
        gx, gy = self.current_goal

        wheel_speeds = self._wheel_speeds_from_twist()
        if wheel_speeds is None:
            return None
        v_left, v_right = wheel_speeds

        worker_req = {
            "px": px,
            "py": py,
            "yaw": yaw,
            "v_left": v_left,
            "v_right": v_right,
            "gx": gx,
            "gy": gy,
            "obstacles": self._scan_to_obstacle_tuples(),
            "walls": self._build_wall_tuples(),
        }
        return self._query_policy_worker(worker_req)

    def _send_planner_request(self, sub_goal: Tuple[float, float]) -> None:
        if self.current_pose is None or self.current_twist is None or self.current_goal is None:
            return

        if not self.ocp_client.wait_for_service(timeout_sec=0.01):
            self.get_logger().warn(f"Service {self.planner_service} not available")
            return

        px, py, yaw = self.current_pose
        gx, gy = self.current_goal
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
        req.ob.walls = self._build_wall_msgs() if self.runtime_use_walls else []
        req.ob.poly_states = self._build_poly_msgs() if self.runtime_use_poly else []
        req.ob.header.stamp = self.get_clock().now().to_msg()
        req.ob.header.frame_id = "map"

        if self.publish_debug_joint_state:
            self.debug_joint_state_pub.publish(req.ob)

        req.sub_goal.x = float(sub_goal[0])
        req.sub_goal.y = float(sub_goal[1])

        self.last_wheel_speed = (v_left, v_right)
        self.last_request_for_viz = req
        self.last_pose_for_viz = (px, py, yaw)
        self.last_wheels_for_viz = (v_left, v_right)

        future = self.ocp_client.call_async(req)
        future.add_done_callback(self._on_planner_response)
        self.pending_future = future

    def _predict_mpc_trajectory(
        self,
        start_pose: Tuple[float, float, float],
        start_wheels: Tuple[float, float],
        control_vars,
    ) -> Tuple[List[Tuple[float, float]], Dict[str, float]]:
        px, py, yaw = start_pose
        vl, vr = start_wheels
        dt = float(self.control_dt)
        horizon_steps = max(1, int(self.mpc_horizon_steps))
        received_steps = len(control_vars)

        points: List[Tuple[float, float]] = [(float(px), float(py))]
        # Force exactly horizon_steps predictions so RViz length reflects MPC horizon.
        for i in range(horizon_steps):
            if i < received_steps:
                al = float(control_vars[i].al)
                ar = float(control_vars[i].ar)
            else:
                # If solver returned fewer controls than expected, keep current wheel speed.
                al = 0.0
                ar = 0.0

            vl += al * dt
            vr += ar * dt
            v = 0.5 * (vl + vr)
            w = (vr - vl) / (2.0 * self.axle_half_width)

            yaw += w * dt
            px += v * math.cos(yaw) * dt
            py += v * math.sin(yaw) * dt
            points.append((float(px), float(py)))

        debug = {
            "expected_horizon_steps": float(horizon_steps),
            "received_control_steps": float(received_steps),
            "predicted_points": float(len(points)),
            "horizon_coverage": float(min(received_steps, horizon_steps)) / float(horizon_steps),
        }
        return points, debug

    @staticmethod
    def _trajectory_length(points: List[Tuple[float, float]]) -> float:
        if len(points) < 2:
            return 0.0
        length = 0.0
        for i in range(1, len(points)):
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
            length += math.hypot(dx, dy)
        return float(length)

    @staticmethod
    def _trajectory_delta_rms(
        current: List[Tuple[float, float]],
        previous: Optional[List[Tuple[float, float]]],
    ) -> float:
        if previous is None:
            return 0.0
        n = min(len(current), len(previous))
        if n <= 1:
            return 0.0

        acc = 0.0
        # Skip index 0 because both trajectories start from the current robot pose.
        for i in range(1, n):
            dx = current[i][0] - previous[i][0]
            dy = current[i][1] - previous[i][1]
            acc += (dx * dx) + (dy * dy)
        return float(math.sqrt(acc / float(n - 1)))

    def _publish_planner_scene_debug(
        self,
        req: OcpLocalPlann.Request,
        predicted_traj: List[Tuple[float, float]],
        mpc_debug: Dict[str, float],
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
            "trajectory": [[float(px), float(py)] for px, py in predicted_traj],
            "mpc_debug": mpc_debug,
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
            self._publish_stop()
            return

        if res.success:
            self.consecutive_planner_failures = 0
            self.success_streak += 1

            # If we had to relax constraints, try restoring them after stable success.
            if self.success_streak >= 20:
                if self.use_wall_constraints and not self.runtime_use_walls:
                    self.runtime_use_walls = True
                    self.get_logger().info("Re-enabled wall constraints after stable planner success")
                    self.success_streak = 0
                elif self.use_demo_poly_obstacle and not self.runtime_use_poly:
                    self.runtime_use_poly = True
                    self.get_logger().info("Re-enabled demo polygon after stable planner success")
                    self.success_streak = 0
        else:
            self.success_streak = 0
            self.consecutive_planner_failures += 1
            if self.consecutive_planner_failures >= 6 and self.runtime_use_poly:
                self.runtime_use_poly = False
                self.get_logger().warn("Temporarily disabled demo polygon constraints due to repeated planner failures")
                self.consecutive_planner_failures = 0
            elif self.consecutive_planner_failures >= 10 and self.runtime_use_walls:
                self.runtime_use_walls = False
                self.get_logger().warn("Temporarily disabled wall constraints due to repeated planner failures")
                self.consecutive_planner_failures = 0

        if self.last_request_for_viz is not None and self.last_pose_for_viz is not None and self.last_wheels_for_viz is not None:
            predicted_traj, mpc_debug = self._predict_mpc_trajectory(
                self.last_pose_for_viz,
                self.last_wheels_for_viz,
                res.control_vars,
            )
            traj_len = self._trajectory_length(predicted_traj)
            delta_rms = self._trajectory_delta_rms(predicted_traj, self.prev_predicted_traj)
            self.last_stability_delta_rms = float(delta_rms)
            self.prev_predicted_traj = list(predicted_traj)

            is_stable = bool(delta_rms <= self.mpc_stability_warn_threshold)
            mpc_debug.update(
                {
                    "trajectory_length_m": float(traj_len),
                    "stability_delta_rms_m": float(delta_rms),
                    "stability_warn_threshold_m": float(self.mpc_stability_warn_threshold),
                    "stable": 1.0 if is_stable else 0.0,
                }
            )

            if mpc_debug["horizon_coverage"] < 0.99:
                self.get_logger().warn(
                    f"MPC returned {int(mpc_debug['received_control_steps'])}/{int(mpc_debug['expected_horizon_steps'])} control steps"
                )

            if not is_stable:
                self.get_logger().warn(
                    f"MPC trajectory jitter detected (delta_rms={delta_rms:.3f} m, threshold={self.mpc_stability_warn_threshold:.3f} m)"
                )

            self._publish_planner_scene_debug(self.last_request_for_viz, predicted_traj, mpc_debug)

        if self.last_wheel_speed is None:
            wheel_speeds = self._wheel_speeds_from_twist()
            if wheel_speeds is None:
                self._publish_stop()
                return
            v_left, v_right = wheel_speeds
        else:
            v_left, v_right = self.last_wheel_speed

        v_left_cmd = v_left + res.al * self.control_dt
        v_right_cmd = v_right + res.ar * self.control_dt

        linear = 0.5 * (v_left_cmd + v_right_cmd)
        angular = (v_right_cmd - v_left_cmd) / (2.0 * self.axle_half_width)

        linear = self._clamp(linear, -self.max_linear_speed, self.max_linear_speed)
        angular = self._clamp(angular, -self.max_angular_speed, self.max_angular_speed)

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = float(linear)
        msg.twist.angular.z = float(angular)
        self.cmd_pub.publish(msg)

    def _policy_loop(self) -> None:
        if self.worker is None:
            return

        if self.current_pose is None or self.current_goal is None:
            return

        px, py, _ = self.current_pose
        gx, gy = self.current_goal
        if math.hypot(gx - px, gy - py) < self.goal_tolerance:
            self.goal_reached = True
            self.latest_sub_goal = None
            self.prev_predicted_traj = None
            self.last_stability_delta_rms = 0.0
            self._publish_action_visualization(None)
            return

        self.goal_reached = False

        try:
            sub_goal = self._get_sub_goal_from_policy()
            if sub_goal is None:
                self._publish_action_visualization(None)
                return
            self.latest_sub_goal = sub_goal
            self.latest_sub_goal_seq += 1
            self._publish_action_visualization(sub_goal)
        except Exception as exc:
            self.get_logger().error(f"Policy loop failed: {exc}")

    def _planner_loop(self) -> None:
        if self.current_pose is None or self.current_goal is None:
            return

        px, py, _ = self.current_pose
        gx, gy = self.current_goal
        if math.hypot(gx - px, gy - py) < self.goal_tolerance:
            self.goal_reached = True
            self._publish_stop()
            return

        self.goal_reached = False
        if self.pending_future is not None:
            return

        if self.latest_sub_goal is None:
            return

        if self.sync_mpc_to_policy and self.last_requested_sub_goal_seq >= self.latest_sub_goal_seq:
            return

        try:
            self._send_planner_request(self.latest_sub_goal)
            self.last_requested_sub_goal_seq = self.latest_sub_goal_seq
        except Exception as exc:
            self.get_logger().error(f"Planner loop failed: {exc}")
            self._publish_stop()


def main(args=None):
    rclpy.init(args=args)
    node = RlOcpPolicyBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.worker is not None and node.worker.poll() is None:
            node.worker.terminate()
            try:
                node.worker.wait(timeout=2.0)
            except Exception:
                pass
        node._publish_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
