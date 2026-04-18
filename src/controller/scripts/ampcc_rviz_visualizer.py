#!/usr/bin/python3
import json
import math
from typing import Dict, List, Optional, Tuple

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point as RosPoint


class AmpccRvizVisualizer(Node):
    def __init__(self):
        super().__init__("ampcc_rviz_visualizer")

        self.declare_parameter("visualize_actions", True)
        self.declare_parameter("visualize_planner_scene", True)

        self.declare_parameter("action_marker_topic", "/policy/action_markers")
        self.declare_parameter("action_marker_frame", "odom")
        self.declare_parameter("planner_scene_marker_topic", "/planner/debug_markers")
        self.declare_parameter("planner_scene_frame", "map")

        self.declare_parameter("action_debug_topic", "/debug/policy_actions_scene")
        self.declare_parameter("planner_scene_debug_topic", "/debug/planner_scene")

        self.visualize_actions = self.get_parameter("visualize_actions").get_parameter_value().bool_value
        self.visualize_planner_scene = self.get_parameter("visualize_planner_scene").get_parameter_value().bool_value

        self.action_marker_topic = self.get_parameter("action_marker_topic").get_parameter_value().string_value
        self.action_marker_frame = self.get_parameter("action_marker_frame").get_parameter_value().string_value
        self.planner_scene_marker_topic = self.get_parameter("planner_scene_marker_topic").get_parameter_value().string_value
        self.planner_scene_frame = self.get_parameter("planner_scene_frame").get_parameter_value().string_value

        self.action_debug_topic = self.get_parameter("action_debug_topic").get_parameter_value().string_value
        self.planner_scene_debug_topic = self.get_parameter("planner_scene_debug_topic").get_parameter_value().string_value

        self.last_action_payload: Optional[Dict] = None
        self.last_scene_payload: Optional[Dict] = None

        self.action_pub = self.create_publisher(MarkerArray, self.action_marker_topic, 10)
        self.scene_pub = self.create_publisher(MarkerArray, self.planner_scene_marker_topic, 10)

        self.create_subscription(String, self.action_debug_topic, self._on_action_debug, 10)
        self.create_subscription(String, self.planner_scene_debug_topic, self._on_scene_debug, 10)
        self.add_on_set_parameters_callback(self._on_parameters_changed)

        self.get_logger().info(
            f"RViz visualizer started. action_debug={self.action_debug_topic}, scene_debug={self.planner_scene_debug_topic}"
        )

    def _on_parameters_changed(self, params):
        for p in params:
            if p.name == "visualize_actions":
                self.visualize_actions = bool(p.value)
            elif p.name == "visualize_planner_scene":
                self.visualize_planner_scene = bool(p.value)

        return SetParametersResult(successful=True)

    def _on_action_debug(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"Invalid action debug payload: {exc}")
            return
        self.last_action_payload = payload
        self._publish_action_markers()

    def _on_scene_debug(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"Invalid scene debug payload: {exc}")
            return
        self.last_scene_payload = payload
        self._publish_scene_markers()

    def _delete_all(self, frame_id: str) -> Marker:
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.action = Marker.DELETEALL
        return m

    @staticmethod
    def _point(x: float, y: float, z: float = 0.05) -> RosPoint:
        return RosPoint(x=float(x), y=float(y), z=float(z))

    def _publish_action_markers(self) -> None:
        payload = self.last_action_payload
        if payload is None:
            return

        frame_id = str(payload.get("frame_id", self.action_marker_frame))
        goal_reached = bool(payload.get("goal_reached", False))

        markers = MarkerArray()
        markers.markers.append(self._delete_all(frame_id))

        if goal_reached or not self.visualize_actions:
            self.action_pub.publish(markers)
            return

        now = self.get_clock().now().to_msg()
        valid_points = [self._point(p[0], p[1]) for p in payload.get("valid_points", [])]
        masked_points = [self._point(p[0], p[1]) for p in payload.get("masked_points", [])]
        selected_point = payload.get("selected_point", None)
        current_pose = payload.get("current_pose", [0.0, 0.0, 0.0])

        valid_marker = Marker()
        valid_marker.header.frame_id = frame_id
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
        masked_marker.header.frame_id = frame_id
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
        selected_marker.header.frame_id = frame_id
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
        if selected_point is not None and len(selected_point) == 2:
            selected_marker.pose.position.x = float(selected_point[0])
            selected_marker.pose.position.y = float(selected_point[1])
            selected_marker.pose.position.z = 0.12
        else:
            selected_marker.action = Marker.DELETE
        markers.markers.append(selected_marker)

        info_marker = Marker()
        info_marker.header.frame_id = frame_id
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
        info_marker.pose.position.x = float(current_pose[0])
        info_marker.pose.position.y = float(current_pose[1])
        info_marker.pose.position.z = 0.6
        info_marker.text = (
            f"actions={int(payload.get('candidate_count', 0))} "
            f"valid={int(payload.get('valid_count', 0))} "
            f"masked={int(payload.get('masked_count', 0))}"
        )
        markers.markers.append(info_marker)

        self.action_pub.publish(markers)

    def _publish_scene_markers(self) -> None:
        payload = self.last_scene_payload
        if payload is None:
            return

        frame_id = str(payload.get("frame_id", self.planner_scene_frame))
        goal_reached = bool(payload.get("goal_reached", False))

        markers = MarkerArray()
        markers.markers.append(self._delete_all(frame_id))

        if goal_reached or not self.visualize_planner_scene:
            self.scene_pub.publish(markers)
            return

        now = self.get_clock().now().to_msg()
        robot = payload.get("robot", {})
        goal = payload.get("goal", {})
        sub_goal = payload.get("sub_goal", {})
        bounds = payload.get("bounds", {})
        walls = payload.get("walls", [])
        obstacles = payload.get("obstacles", [])
        polygons = payload.get("polygons", [])
        humans = payload.get("humans", [])
        trajectory = payload.get("trajectory", [])
        mpc_debug = payload.get("mpc_debug", {})

        robot_radius = float(robot.get("radius", 0.25))

        robot_marker = Marker()
        robot_marker.header.frame_id = frame_id
        robot_marker.header.stamp = now
        robot_marker.ns = "planner_scene"
        robot_marker.id = 100
        robot_marker.type = Marker.CYLINDER
        robot_marker.action = Marker.ADD
        robot_marker.pose.position.x = float(robot.get("x", 0.0))
        robot_marker.pose.position.y = float(robot.get("y", 0.0))
        robot_marker.pose.position.z = 0.05
        robot_marker.pose.orientation.w = 1.0
        robot_marker.scale.x = 2.0 * robot_radius
        robot_marker.scale.y = 2.0 * robot_radius
        robot_marker.scale.z = 0.10
        robot_marker.color.r = 1.0
        robot_marker.color.g = 0.35
        robot_marker.color.b = 0.35
        robot_marker.color.a = 0.85
        markers.markers.append(robot_marker)

        heading = Marker()
        heading.header.frame_id = frame_id
        heading.header.stamp = now
        heading.ns = "planner_scene"
        heading.id = 103
        heading.type = Marker.ARROW
        heading.action = Marker.ADD
        heading.scale.x = 0.05
        heading.scale.y = 0.10
        heading.scale.z = 0.10
        heading.color.r = 1.0
        heading.color.g = 0.8
        heading.color.b = 0.2
        heading.color.a = 0.95
        x = float(robot.get("x", 0.0))
        y = float(robot.get("y", 0.0))
        theta = float(robot.get("theta", 0.0))
        heading.points = [self._point(x, y, 0.1), self._point(x + 0.5 * math.cos(theta), y + 0.5 * math.sin(theta), 0.1)]
        markers.markers.append(heading)

        goal_marker = Marker()
        goal_marker.header.frame_id = frame_id
        goal_marker.header.stamp = now
        goal_marker.ns = "planner_scene"
        goal_marker.id = 101
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        goal_marker.pose.position.x = float(goal.get("x", 0.0))
        goal_marker.pose.position.y = float(goal.get("y", 0.0))
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
        sub_goal_marker.header.frame_id = frame_id
        sub_goal_marker.header.stamp = now
        sub_goal_marker.ns = "planner_scene"
        sub_goal_marker.id = 102
        sub_goal_marker.type = Marker.SPHERE
        sub_goal_marker.action = Marker.ADD
        sub_goal_marker.pose.position.x = float(sub_goal.get("x", 0.0))
        sub_goal_marker.pose.position.y = float(sub_goal.get("y", 0.0))
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

        bounds_marker = Marker()
        bounds_marker.header.frame_id = frame_id
        bounds_marker.header.stamp = now
        bounds_marker.ns = "planner_scene"
        bounds_marker.id = 104
        bounds_marker.type = Marker.LINE_STRIP
        bounds_marker.action = Marker.ADD
        bounds_marker.scale.x = 0.04
        bounds_marker.color.r = 0.80
        bounds_marker.color.g = 0.85
        bounds_marker.color.b = 1.0
        bounds_marker.color.a = 0.75
        if bounds:
            xmin = float(bounds.get("xmin", -5.0))
            xmax = float(bounds.get("xmax", 5.0))
            ymin = float(bounds.get("ymin", -5.0))
            ymax = float(bounds.get("ymax", 5.0))
            bounds_marker.points = [
                self._point(xmin, ymin),
                self._point(xmin, ymax),
                self._point(xmax, ymax),
                self._point(xmax, ymin),
                self._point(xmin, ymin),
            ]
            markers.markers.append(bounds_marker)

        wall_marker = Marker()
        wall_marker.header.frame_id = frame_id
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
        for w in walls:
            if len(w) != 4:
                continue
            wall_marker.points.append(self._point(w[0], w[1]))
            wall_marker.points.append(self._point(w[2], w[3]))
        markers.markers.append(wall_marker)

        poly_marker = Marker()
        poly_marker.header.frame_id = frame_id
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
        for poly in polygons:
            if len(poly) < 2:
                continue
            for i in range(len(poly)):
                p1 = poly[i]
                p2 = poly[(i + 1) % len(poly)]
                poly_marker.points.append(self._point(p1[0], p1[1]))
                poly_marker.points.append(self._point(p2[0], p2[1]))
        markers.markers.append(poly_marker)

        obst_marker = Marker()
        obst_marker.header.frame_id = frame_id
        obst_marker.header.stamp = now
        obst_marker.ns = "planner_scene"
        obst_marker.id = 130
        obst_marker.type = Marker.SPHERE_LIST
        obst_marker.action = Marker.ADD
        obst_marker.scale.x = 0.20
        obst_marker.scale.y = 0.20
        obst_marker.scale.z = 0.10
        obst_marker.color.r = 1.0
        obst_marker.color.g = 0.10
        obst_marker.color.b = 0.10
        obst_marker.color.a = 0.85
        for obs in obstacles:
            if len(obs) < 3:
                continue
            radius = max(0.05, float(obs[2]))
            obst_marker.scale.x = 2.0 * radius
            obst_marker.scale.y = 2.0 * radius
            obst_marker.points.append(self._point(obs[0], obs[1]))
        markers.markers.append(obst_marker)

        human_marker = Marker()
        human_marker.header.frame_id = frame_id
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
        for hum in humans:
            human_marker.points.append(self._point(hum.get("px", 0.0), hum.get("py", 0.0)))
        markers.markers.append(human_marker)

        human_dir_marker = Marker()
        human_dir_marker.header.frame_id = frame_id
        human_dir_marker.header.stamp = now
        human_dir_marker.ns = "planner_scene"
        human_dir_marker.id = 141
        human_dir_marker.type = Marker.LINE_LIST
        human_dir_marker.action = Marker.ADD
        human_dir_marker.scale.x = 0.03
        human_dir_marker.color.r = 0.10
        human_dir_marker.color.g = 1.0
        human_dir_marker.color.b = 0.10
        human_dir_marker.color.a = 0.95
        for hum in humans:
            px = float(hum.get("px", 0.0))
            py = float(hum.get("py", 0.0))
            vx = float(hum.get("vx", 0.0))
            vy = float(hum.get("vy", 0.0))
            norm = math.hypot(vx, vy)
            if norm < 1e-3:
                continue
            scale = 0.6
            ex = px + scale * vx / norm
            ey = py + scale * vy / norm
            human_dir_marker.points.append(self._point(px, py, 0.09))
            human_dir_marker.points.append(self._point(ex, ey, 0.09))
        markers.markers.append(human_dir_marker)

        traj_marker = Marker()
        traj_marker.header.frame_id = frame_id
        traj_marker.header.stamp = now
        traj_marker.ns = "planner_scene"
        traj_marker.id = 150
        traj_marker.type = Marker.LINE_STRIP
        traj_marker.action = Marker.ADD
        traj_marker.scale.x = 0.08
        delta_rms = float(mpc_debug.get("stability_delta_rms_m", 0.0))
        jitter_threshold = max(1e-6, float(mpc_debug.get("stability_warn_threshold_m", 0.35)))
        stable = bool(float(mpc_debug.get("stable", 1.0)) > 0.5)
        jitter_ratio = min(1.0, delta_rms / jitter_threshold)

        if stable:
            traj_marker.color.r = 0.10
            traj_marker.color.g = 0.75
            traj_marker.color.b = 1.00
        else:
            traj_marker.color.r = 1.00
            traj_marker.color.g = max(0.0, 0.65 - 0.35 * jitter_ratio)
            traj_marker.color.b = 0.15
        traj_marker.color.a = 0.95
        for pt in trajectory:
            if len(pt) != 2:
                continue
            traj_marker.points.append(self._point(pt[0], pt[1], 0.08))
        markers.markers.append(traj_marker)

        mpc_info = Marker()
        mpc_info.header.frame_id = frame_id
        mpc_info.header.stamp = now
        mpc_info.ns = "planner_scene"
        mpc_info.id = 151
        mpc_info.type = Marker.TEXT_VIEW_FACING
        mpc_info.action = Marker.ADD
        mpc_info.scale.z = 0.25
        if stable:
            mpc_info.color.r = 0.20
            mpc_info.color.g = 1.0
            mpc_info.color.b = 0.25
        else:
            mpc_info.color.r = 1.0
            mpc_info.color.g = 0.20
            mpc_info.color.b = 0.20
        mpc_info.color.a = 0.95

        rx = float(robot.get("x", 0.0))
        ry = float(robot.get("y", 0.0))
        mpc_info.pose.position.x = rx
        mpc_info.pose.position.y = ry
        mpc_info.pose.position.z = 1.0

        expected_h = int(float(mpc_debug.get("expected_horizon_steps", 0.0)))
        received_h = int(float(mpc_debug.get("received_control_steps", 0.0)))
        pred_pts = int(float(mpc_debug.get("predicted_points", 0.0)))
        coverage = 100.0 * float(mpc_debug.get("horizon_coverage", 0.0))
        traj_len = float(mpc_debug.get("trajectory_length_m", 0.0))
        stability_tag = "STABLE" if stable else "UNSTABLE"
        mpc_info.text = (
            f"MPC H={expected_h} ctrl={received_h} pts={pred_pts} cov={coverage:.0f}%\\n"
            f"len={traj_len:.2f}m dRMS={delta_rms:.3f}m -> {stability_tag}"
        )
        markers.markers.append(mpc_info)

        horizon_end = Marker()
        horizon_end.header.frame_id = frame_id
        horizon_end.header.stamp = now
        horizon_end.ns = "planner_scene"
        horizon_end.id = 152
        horizon_end.type = Marker.SPHERE
        horizon_end.action = Marker.ADD
        horizon_end.scale.x = 0.18
        horizon_end.scale.y = 0.18
        horizon_end.scale.z = 0.18
        horizon_end.color.r = 1.0
        horizon_end.color.g = 0.85 if stable else 0.25
        horizon_end.color.b = 0.15
        horizon_end.color.a = 0.95
        if trajectory and len(trajectory[-1]) == 2:
            horizon_end.pose.position.x = float(trajectory[-1][0])
            horizon_end.pose.position.y = float(trajectory[-1][1])
            horizon_end.pose.position.z = 0.14
        else:
            horizon_end.action = Marker.DELETE
        markers.markers.append(horizon_end)

        self.scene_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = AmpccRvizVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
