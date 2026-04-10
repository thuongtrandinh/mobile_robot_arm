#!/usr/bin/python3
import json
import math
import os
import shlex
import subprocess
import sys
from typing import List, Optional, Tuple

import rclpy
from interfaces.msg import ObstacleState, WallState, PolyState, Point as PolyPoint
from interfaces.msg import JointState as InterfaceJointState
from interfaces.srv import OcpLocalPlann
from geometry_msgs.msg import PoseStamped, TwistStamped, Point as RosPoint
from nav_msgs.msg import OccupancyGrid, Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


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
        self.declare_parameter("control_dt", 0.25)
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
        self.control_dt = self.get_parameter("control_dt").get_parameter_value().double_value
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
        self.worker: Optional[subprocess.Popen] = None
        self.last_request_for_viz: Optional[OcpLocalPlann.Request] = None
        self.last_pose_for_viz: Optional[Tuple[float, float, float]] = None
        self.last_wheels_for_viz: Optional[Tuple[float, float]] = None
        self.consecutive_planner_failures = 0
        self.runtime_use_walls = self.use_wall_constraints
        self.runtime_use_poly = self.use_demo_poly_obstacle
        self.success_streak = 0

        self._start_policy_worker()

        self.create_subscription(Odometry, self.odom_topic, self._odom_callback, 10)
        self.create_subscription(LaserScan, self.scan_topic, self._scan_callback, 10)
        self.create_subscription(PoseStamped, self.goal_topic, self._goal_callback, 10)
        self.create_subscription(OccupancyGrid, self.map_topic, self._map_callback, 10)

        self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_topic, 10)
        self.action_viz_pub = self.create_publisher(MarkerArray, self.action_marker_topic, 10)
        self.scene_viz_pub = self.create_publisher(MarkerArray, self.planner_scene_marker_topic, 10)
        self.debug_joint_state_pub = self.create_publisher(InterfaceJointState, self.debug_joint_state_topic, 10)
        self.policy_debug_status_pub = self.create_publisher(String, self.policy_debug_status_topic, 10)
        self.ocp_client = self.create_client(OcpLocalPlann, self.planner_service)
        self.add_on_set_parameters_callback(self._on_parameters_changed)

        timer_period = self.get_parameter("timer_period").get_parameter_value().double_value
        self.create_timer(timer_period, self._control_loop)

        self.get_logger().info("RL OCP policy bridge started")
        self.get_logger().info(f"Model: {self.model_path}")
        self.get_logger().info(f"Service: {self.planner_service}")
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

        now = self.get_clock().now().to_msg()
        candidates = self._generate_action_candidates()
        obstacles = self._scan_to_obstacle_tuples()

        valid_points: List[RosPoint] = []
        masked_points: List[RosPoint] = []
        closest_selected: Optional[Tuple[float, float]] = None
        best_dist = float("inf")

        for lx, ly, wx, wy in candidates:
            p = RosPoint(x=float(wx), y=float(wy), z=0.05)
            if self._is_action_masked(lx, ly, wx, wy, obstacles):
                masked_points.append(p)
            else:
                valid_points.append(p)

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
                "selected_sub_goal": list(selected_sub_goal) if selected_sub_goal is not None else None,
            }
            msg = String()
            msg.data = json.dumps(status)
            self.policy_debug_status_pub.publish(msg)

        if not self.visualize_actions:
            return

        markers = MarkerArray()

        delete_all = Marker()
        delete_all.header.frame_id = self.action_marker_frame
        delete_all.header.stamp = now
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        valid_marker = Marker()
        valid_marker.header.frame_id = self.action_marker_frame
        valid_marker.header.stamp = now
        valid_marker.ns = "policy_actions"
        valid_marker.id = 1
        valid_marker.type = Marker.SPHERE_LIST
        valid_marker.action = Marker.ADD
        valid_marker.scale.x = 0.10
        valid_marker.scale.y = 0.10
        valid_marker.scale.z = 0.10
        valid_marker.color.r = 0.0
        valid_marker.color.g = 1.0
        valid_marker.color.b = 0.25
        valid_marker.color.a = 0.85
        valid_marker.points = valid_points
        markers.markers.append(valid_marker)

        masked_marker = Marker()
        masked_marker.header.frame_id = self.action_marker_frame
        masked_marker.header.stamp = now
        masked_marker.ns = "policy_actions"
        masked_marker.id = 2
        masked_marker.type = Marker.SPHERE_LIST
        masked_marker.action = Marker.ADD
        masked_marker.scale.x = 0.08
        masked_marker.scale.y = 0.08
        masked_marker.scale.z = 0.08
        masked_marker.color.r = 1.0
        masked_marker.color.g = 0.1
        masked_marker.color.b = 0.1
        masked_marker.color.a = 0.65
        masked_marker.points = masked_points
        markers.markers.append(masked_marker)

        selected_marker = Marker()
        selected_marker.header.frame_id = self.action_marker_frame
        selected_marker.header.stamp = now
        selected_marker.ns = "policy_actions"
        selected_marker.id = 3
        selected_marker.type = Marker.SPHERE
        selected_marker.action = Marker.ADD
        selected_marker.scale.x = 0.18
        selected_marker.scale.y = 0.18
        selected_marker.scale.z = 0.18
        selected_marker.color.r = 0.1
        selected_marker.color.g = 0.4
        selected_marker.color.b = 1.0
        selected_marker.color.a = 1.0
        if closest_selected is not None:
            selected_marker.pose.position.x = float(closest_selected[0])
            selected_marker.pose.position.y = float(closest_selected[1])
            selected_marker.pose.position.z = 0.12
        else:
            selected_marker.action = Marker.DELETE
        markers.markers.append(selected_marker)

        info_marker = Marker()
        info_marker.header.frame_id = self.action_marker_frame
        info_marker.header.stamp = now
        info_marker.ns = "policy_actions"
        info_marker.id = 4
        info_marker.type = Marker.TEXT_VIEW_FACING
        info_marker.action = Marker.ADD
        info_marker.scale.z = 0.25
        info_marker.color.r = 1.0
        info_marker.color.g = 1.0
        info_marker.color.b = 1.0
        info_marker.color.a = 0.95
        px, py, _ = self.current_pose
        info_marker.pose.position.x = float(px)
        info_marker.pose.position.y = float(py)
        info_marker.pose.position.z = 0.6
        info_marker.text = f"actions={len(candidates)} valid={len(valid_points)} masked={len(masked_points)}"
        markers.markers.append(info_marker)

        self.action_viz_pub.publish(markers)

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
    ) -> List[Tuple[float, float]]:
        px, py, yaw = start_pose
        vl, vr = start_wheels
        dt = float(self.control_dt)

        points: List[Tuple[float, float]] = [(float(px), float(py))]
        for cv in control_vars:
            vl += float(cv.al) * dt
            vr += float(cv.ar) * dt
            v = 0.5 * (vl + vr)
            w = (vr - vl) / (2.0 * self.axle_half_width)

            yaw += w * dt
            px += v * math.cos(yaw) * dt
            py += v * math.sin(yaw) * dt
            points.append((float(px), float(py)))

        return points

    def _publish_planner_scene_markers(
        self,
        req: OcpLocalPlann.Request,
        predicted_traj: List[Tuple[float, float]],
    ) -> None:
        if not self.visualize_planner_scene:
            return

        now = self.get_clock().now().to_msg()
        markers = MarkerArray()

        delete_all = Marker()
        delete_all.header.frame_id = self.planner_scene_frame
        delete_all.header.stamp = now
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        robot_marker = Marker()
        robot_marker.header.frame_id = self.planner_scene_frame
        robot_marker.header.stamp = now
        robot_marker.ns = "planner_scene"
        robot_marker.id = 100
        robot_marker.type = Marker.CYLINDER
        robot_marker.action = Marker.ADD
        robot_marker.pose.position.x = float(req.ob.robot_state.pose.x)
        robot_marker.pose.position.y = float(req.ob.robot_state.pose.y)
        robot_marker.pose.position.z = 0.05
        robot_marker.pose.orientation.w = 1.0
        robot_marker.scale.x = 2.0 * float(self.robot_radius)
        robot_marker.scale.y = 2.0 * float(self.robot_radius)
        robot_marker.scale.z = 0.10
        robot_marker.color.r = 1.0
        robot_marker.color.g = 0.35
        robot_marker.color.b = 0.35
        robot_marker.color.a = 0.85
        markers.markers.append(robot_marker)

        goal_marker = Marker()
        goal_marker.header.frame_id = self.planner_scene_frame
        goal_marker.header.stamp = now
        goal_marker.ns = "planner_scene"
        goal_marker.id = 101
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        goal_marker.pose.position.x = float(req.ob.robot_state.gx)
        goal_marker.pose.position.y = float(req.ob.robot_state.gy)
        goal_marker.pose.position.z = 0.05
        goal_marker.pose.orientation.w = 1.0
        goal_marker.scale.x = 0.20
        goal_marker.scale.y = 0.20
        goal_marker.scale.z = 0.20
        goal_marker.color.r = 0.10
        goal_marker.color.g = 0.95
        goal_marker.color.b = 0.95
        goal_marker.color.a = 0.95
        markers.markers.append(goal_marker)

        sub_goal_marker = Marker()
        sub_goal_marker.header.frame_id = self.planner_scene_frame
        sub_goal_marker.header.stamp = now
        sub_goal_marker.ns = "planner_scene"
        sub_goal_marker.id = 102
        sub_goal_marker.type = Marker.SPHERE
        sub_goal_marker.action = Marker.ADD
        sub_goal_marker.pose.position.x = float(req.sub_goal.x)
        sub_goal_marker.pose.position.y = float(req.sub_goal.y)
        sub_goal_marker.pose.position.z = 0.05
        sub_goal_marker.pose.orientation.w = 1.0
        sub_goal_marker.scale.x = 0.16
        sub_goal_marker.scale.y = 0.16
        sub_goal_marker.scale.z = 0.16
        sub_goal_marker.color.r = 0.30
        sub_goal_marker.color.g = 0.55
        sub_goal_marker.color.b = 1.0
        sub_goal_marker.color.a = 0.95
        markers.markers.append(sub_goal_marker)

        wall_marker = Marker()
        wall_marker.header.frame_id = self.planner_scene_frame
        wall_marker.header.stamp = now
        wall_marker.ns = "planner_scene"
        wall_marker.id = 110
        wall_marker.type = Marker.LINE_LIST
        wall_marker.action = Marker.ADD
        wall_marker.scale.x = 0.07
        wall_marker.color.r = 0.95
        wall_marker.color.g = 0.90
        wall_marker.color.b = 0.20
        wall_marker.color.a = 0.95
        for wall in req.ob.walls:
            wall_marker.points.append(RosPoint(x=float(wall.sx), y=float(wall.sy), z=0.05))
            wall_marker.points.append(RosPoint(x=float(wall.ex), y=float(wall.ey), z=0.05))
        markers.markers.append(wall_marker)

        poly_marker = Marker()
        poly_marker.header.frame_id = self.planner_scene_frame
        poly_marker.header.stamp = now
        poly_marker.ns = "planner_scene"
        poly_marker.id = 120
        poly_marker.type = Marker.LINE_LIST
        poly_marker.action = Marker.ADD
        poly_marker.scale.x = 0.06
        poly_marker.color.r = 0.75
        poly_marker.color.g = 0.35
        poly_marker.color.b = 1.0
        poly_marker.color.a = 0.95
        for poly in req.ob.poly_states:
            n = len(poly.vertices)
            if n < 2:
                continue
            for i in range(n):
                p1 = poly.vertices[i]
                p2 = poly.vertices[(i + 1) % n]
                poly_marker.points.append(RosPoint(x=float(p1.x), y=float(p1.y), z=0.05))
                poly_marker.points.append(RosPoint(x=float(p2.x), y=float(p2.y), z=0.05))
        markers.markers.append(poly_marker)

        obst_marker = Marker()
        obst_marker.header.frame_id = self.planner_scene_frame
        obst_marker.header.stamp = now
        obst_marker.ns = "planner_scene"
        obst_marker.id = 130
        obst_marker.type = Marker.SPHERE_LIST
        obst_marker.action = Marker.ADD
        obst_marker.scale.x = 2.0 * float(self.obstacle_radius)
        obst_marker.scale.y = 2.0 * float(self.obstacle_radius)
        obst_marker.scale.z = 0.10
        obst_marker.color.r = 1.0
        obst_marker.color.g = 0.10
        obst_marker.color.b = 0.10
        obst_marker.color.a = 0.85
        for obs in req.ob.obstacle_states:
            obst_marker.points.append(RosPoint(x=float(obs.px), y=float(obs.py), z=0.05))
        markers.markers.append(obst_marker)

        human_marker = Marker()
        human_marker.header.frame_id = self.planner_scene_frame
        human_marker.header.stamp = now
        human_marker.ns = "planner_scene"
        human_marker.id = 140
        human_marker.type = Marker.SPHERE_LIST
        human_marker.action = Marker.ADD
        human_marker.scale.x = 0.25
        human_marker.scale.y = 0.25
        human_marker.scale.z = 0.10
        human_marker.color.r = 0.20
        human_marker.color.g = 0.95
        human_marker.color.b = 0.20
        human_marker.color.a = 0.85
        for hum in req.ob.human_states:
            human_marker.points.append(RosPoint(x=float(hum.px), y=float(hum.py), z=0.05))
        markers.markers.append(human_marker)

        traj_marker = Marker()
        traj_marker.header.frame_id = self.planner_scene_frame
        traj_marker.header.stamp = now
        traj_marker.ns = "planner_scene"
        traj_marker.id = 150
        traj_marker.type = Marker.LINE_STRIP
        traj_marker.action = Marker.ADD
        traj_marker.scale.x = 0.08
        traj_marker.color.r = 0.10
        traj_marker.color.g = 0.55
        traj_marker.color.b = 1.00
        traj_marker.color.a = 0.95
        for tx, ty in predicted_traj:
            traj_marker.points.append(RosPoint(x=float(tx), y=float(ty), z=0.08))
        markers.markers.append(traj_marker)

        self.scene_viz_pub.publish(markers)

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
            predicted_traj = self._predict_mpc_trajectory(
                self.last_pose_for_viz,
                self.last_wheels_for_viz,
                res.control_vars,
            )
            self._publish_planner_scene_markers(self.last_request_for_viz, predicted_traj)

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

    def _control_loop(self) -> None:
        if self.worker is None:
            return

        if self.current_pose is None or self.current_goal is None:
            return

        px, py, _ = self.current_pose
        gx, gy = self.current_goal
        if math.hypot(gx - px, gy - py) < self.goal_tolerance:
            self._publish_stop()
            return

        if self.pending_future is not None:
            return

        try:
            sub_goal = self._get_sub_goal_from_policy()
            if sub_goal is None:
                self._publish_action_visualization(None)
                return
            self._publish_action_visualization(sub_goal)
            self._send_planner_request(sub_goal)
        except Exception as exc:
            self.get_logger().error(f"Control loop failed: {exc}")
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
