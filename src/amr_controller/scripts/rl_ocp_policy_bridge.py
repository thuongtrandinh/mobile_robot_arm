#!/usr/bin/python3
import json
import math
import os
import shlex
import subprocess
import sys
from typing import List, Optional, Tuple

import rclpy
from amr_interfaces.msg import ObstacleState, WallState
from amr_interfaces.srv import OcpLocalPlann
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class RlOcpPolicyBridge(Node):
    def __init__(self):
        super().__init__("rl_ocp_policy_bridge")

        self.declare_parameter("policy_env_name", "mpc_rl")
        self.declare_parameter("halo_drl_dir", "/home/thuong/LVTN/amr_ws/HALO_1/drl_moudle")
        self.declare_parameter("config", "configs/mpc_rl.py")
        self.declare_parameter("model_path", "/home/thuong/LVTN/amr_ws/src/amr_controller/policy/best_model.zip")
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

        self.pending_future = None
        self.last_wheel_speed: Optional[Tuple[float, float]] = None
        self.worker: Optional[subprocess.Popen] = None

        self._start_policy_worker()

        self.create_subscription(Odometry, self.odom_topic, self._odom_callback, 10)
        self.create_subscription(LaserScan, self.scan_topic, self._scan_callback, 10)
        self.create_subscription(PoseStamped, self.goal_topic, self._goal_callback, 10)
        self.create_subscription(OccupancyGrid, self.map_topic, self._map_callback, 10)

        self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_topic, 10)
        self.ocp_client = self.create_client(OcpLocalPlann, self.planner_service)

        timer_period = self.get_parameter("timer_period").get_parameter_value().double_value
        self.create_timer(timer_period, self._control_loop)

        self.get_logger().info("RL OCP policy bridge started")
        self.get_logger().info(f"Model: {self.model_path}")
        self.get_logger().info(f"Service: {self.planner_service}")

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
        if self.map_bounds is None:
            if self.current_pose is None:
                xmin, xmax, ymin, ymax = -10.0, 10.0, -10.0, 10.0
            else:
                px, py, _ = self.current_pose
                xmin, xmax, ymin, ymax = px - 10.0, px + 10.0, py - 10.0, py + 10.0
        else:
            xmin, xmax, ymin, ymax = self.map_bounds

        walls = []
        for sx, sy, ex, ey in self._build_wall_tuples():
            wall = WallState()
            wall.sx = float(sx)
            wall.sy = float(sy)
            wall.ex = float(ex)
            wall.ey = float(ey)
            walls.append(wall)

        return walls

    def _scan_to_obstacle_tuples(self) -> List[Tuple[float, float, float]]:
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
            obstacles.append((ox, oy, self.obstacle_radius))

        return obstacles

    def _scan_to_obstacle_msgs(self, origin_x: float, origin_y: float) -> List[ObstacleState]:
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
            rel_x = px + r * math.cos(ang) - origin_x
            rel_y = py + r * math.sin(ang) - origin_y
            if abs(rel_x) > self.planner_half_width or abs(rel_y) > self.planner_half_height:
                continue
            obstacle = ObstacleState()
            obstacle.px = float(rel_x)
            obstacle.py = float(rel_y)
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

        goal_local_x = gx - px
        goal_local_y = gy - py
        sub_goal_local_x = sub_goal[0] - px
        sub_goal_local_y = sub_goal[1] - py

        sub_goal_local_x = self._clamp(sub_goal_local_x, -self.planner_half_width, self.planner_half_width)
        sub_goal_local_y = self._clamp(sub_goal_local_y, -self.planner_half_height, self.planner_half_height)

        req = OcpLocalPlann.Request()
        req.ob.robot_state.pose.x = 0.0
        req.ob.robot_state.pose.y = 0.0
        req.ob.robot_state.pose.theta = float(yaw)
        req.ob.robot_state.vl = float(v_left)
        req.ob.robot_state.vr = float(v_right)
        req.ob.robot_state.gx = float(goal_local_x)
        req.ob.robot_state.gy = float(goal_local_y)
        req.ob.robot_state.v_pref = float(self.v_pref)
        req.ob.robot_state.radius = float(self.axle_half_width)

        req.ob.obstacle_states = self._scan_to_obstacle_msgs(px, py)
        req.ob.walls = []

        req.sub_goal.x = float(sub_goal_local_x)
        req.sub_goal.y = float(sub_goal_local_y)

        self.last_wheel_speed = (v_left, v_right)

        future = self.ocp_client.call_async(req)
        future.add_done_callback(self._on_planner_response)
        self.pending_future = future

    def _on_planner_response(self, future) -> None:
        self.pending_future = None
        try:
            res = future.result()
        except Exception as exc:
            self.get_logger().error(f"Planner service call failed: {exc}")
            self._publish_stop()
            return

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
                return
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
