#!/usr/bin/python3
import json
import math
import importlib
import io
from typing import Dict, List, Optional, Tuple
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point as RosPoint
from interfaces.msg import JointState as InterfaceJointState
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan


class AmpccRvizVisualizer(Node):
    def __init__(self):
        super().__init__("ampcc_rviz_visualizer")

        self.declare_parameter("visualize_actions", True)
        self.declare_parameter("visualize_planner_scene", True)
        self.declare_parameter("visualize_local_costmap", True)
        self.declare_parameter("visualize_scan_obstacles", True)
        self.declare_parameter("visualize_joint_state_geometry", True)
        self.declare_parameter("visualize_astar_path", True)
        self.declare_parameter("visualize_astar_local_map", True)

        self.declare_parameter("action_marker_topic", "/policy/action_markers")
        self.declare_parameter("action_marker_frame", "odom")
        self.declare_parameter("planner_scene_marker_topic", "/planner/debug_markers")
        self.declare_parameter("planner_scene_frame", "map")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("joint_state_topic", "/debug/joint_state_req")
        self.declare_parameter("astar_path_topic", "/a_star_path")
        self.declare_parameter("astar_local_map_topic", "/map")
        self.declare_parameter("local_costmap_topic", "/local_costmap/costmap")
        self.declare_parameter("local_costmap_sample_step", 2)
        self.declare_parameter("local_costmap_max_cells", 12000)
        self.declare_parameter("local_costmap_min_cost", 1)
        self.declare_parameter("astar_local_map_sample_step", 2)
        self.declare_parameter("astar_local_map_max_cells", 12000)
        self.declare_parameter("astar_local_map_min_cost", 1)

        self.declare_parameter("map_obstacle_threshold", 90)
        self.declare_parameter("poly_epsilon_ratio", 0.02)
        self.declare_parameter("cluster_distance", 0.12)
        self.declare_parameter("dynamic_marker_z", 0.05)
        self.declare_parameter("use_opencv", True)
        self.declare_parameter("min_cluster_points", 3)
        self.declare_parameter("max_cluster_points", 60)
        self.declare_parameter("max_cluster_span", 0.80)
        self.declare_parameter("min_circle_radius", 0.05)
        self.declare_parameter("max_circle_radius", 0.45)
        self.declare_parameter("linearity_reject_ratio", 20.0)
        self.declare_parameter("linearity_reject_span", 0.50)

        self.declare_parameter("action_debug_topic", "/debug/policy_actions_scene")
        self.declare_parameter("planner_scene_debug_topic", "/debug/planner_scene")

        self.visualize_actions = self.get_parameter("visualize_actions").get_parameter_value().bool_value
        self.visualize_planner_scene = self.get_parameter("visualize_planner_scene").get_parameter_value().bool_value
        self.visualize_local_costmap = self.get_parameter("visualize_local_costmap").get_parameter_value().bool_value
        self.visualize_scan_obstacles = self.get_parameter("visualize_scan_obstacles").get_parameter_value().bool_value
        self.visualize_joint_state_geometry = self.get_parameter("visualize_joint_state_geometry").get_parameter_value().bool_value
        self.visualize_astar_path = self.get_parameter("visualize_astar_path").get_parameter_value().bool_value
        self.visualize_astar_local_map = self.get_parameter("visualize_astar_local_map").get_parameter_value().bool_value

        self.action_marker_topic = self.get_parameter("action_marker_topic").get_parameter_value().string_value
        self.action_marker_frame = self.get_parameter("action_marker_frame").get_parameter_value().string_value
        self.planner_scene_marker_topic = self.get_parameter("planner_scene_marker_topic").get_parameter_value().string_value
        self.planner_scene_frame = self.get_parameter("planner_scene_frame").get_parameter_value().string_value
        self.map_topic = self.get_parameter("map_topic").get_parameter_value().string_value
        self.scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.joint_state_topic = self.get_parameter("joint_state_topic").get_parameter_value().string_value
        self.astar_path_topic = self.get_parameter("astar_path_topic").get_parameter_value().string_value
        self.astar_local_map_topic = self.get_parameter("astar_local_map_topic").get_parameter_value().string_value
        self.local_costmap_topic = self.get_parameter("local_costmap_topic").get_parameter_value().string_value
        self.local_costmap_sample_step = max(
            1, self.get_parameter("local_costmap_sample_step").get_parameter_value().integer_value
        )
        self.local_costmap_max_cells = max(
            100, self.get_parameter("local_costmap_max_cells").get_parameter_value().integer_value
        )
        self.local_costmap_min_cost = max(
            0, min(100, self.get_parameter("local_costmap_min_cost").get_parameter_value().integer_value)
        )
        self.astar_local_map_sample_step = max(
            1, self.get_parameter("astar_local_map_sample_step").get_parameter_value().integer_value
        )
        self.astar_local_map_max_cells = max(
            100, self.get_parameter("astar_local_map_max_cells").get_parameter_value().integer_value
        )
        self.astar_local_map_min_cost = max(
            0, min(100, self.get_parameter("astar_local_map_min_cost").get_parameter_value().integer_value)
        )

        self.map_obstacle_threshold = int(
            self.get_parameter("map_obstacle_threshold").get_parameter_value().integer_value
        )
        self.poly_epsilon_ratio = float(
            self.get_parameter("poly_epsilon_ratio").get_parameter_value().double_value
        )
        self.cluster_distance = float(
            self.get_parameter("cluster_distance").get_parameter_value().double_value
        )
        self.dynamic_marker_z = float(
            self.get_parameter("dynamic_marker_z").get_parameter_value().double_value
        )
        self.use_opencv = bool(self.get_parameter("use_opencv").get_parameter_value().bool_value)
        self.min_cluster_points = int(
            self.get_parameter("min_cluster_points").get_parameter_value().integer_value
        )
        self.max_cluster_points = int(
            self.get_parameter("max_cluster_points").get_parameter_value().integer_value
        )
        self.max_cluster_span = float(
            self.get_parameter("max_cluster_span").get_parameter_value().double_value
        )
        self.min_circle_radius = float(
            self.get_parameter("min_circle_radius").get_parameter_value().double_value
        )
        self.max_circle_radius = float(
            self.get_parameter("max_circle_radius").get_parameter_value().double_value
        )
        self.linearity_reject_ratio = float(
            self.get_parameter("linearity_reject_ratio").get_parameter_value().double_value
        )
        self.linearity_reject_span = float(
            self.get_parameter("linearity_reject_span").get_parameter_value().double_value
        )

        self.action_debug_topic = self.get_parameter("action_debug_topic").get_parameter_value().string_value
        self.planner_scene_debug_topic = self.get_parameter("planner_scene_debug_topic").get_parameter_value().string_value

        self.last_action_payload: Optional[Dict] = None
        self.last_scene_payload: Optional[Dict] = None
        self.last_costmap: Optional[OccupancyGrid] = None
        self.last_astar_local_map: Optional[OccupancyGrid] = None

        self.static_polygons: List[List[Tuple[float, float]]] = []
        self.dynamic_circles: List[Tuple[float, float, float]] = []
        self.scene_polygons: List[List[Tuple[float, float]]] = []
        self.scene_walls: List[Tuple[float, float, float, float]] = []
        self.astar_path_points: List[Tuple[float, float]] = []

        self.map_resolution: float = 0.0
        self.map_origin_x: float = 0.0
        self.map_origin_y: float = 0.0
        self.map_height: int = 0

        self.map_frame: str = self.planner_scene_frame
        self.scan_frame: str = "base_scan"
        self.scene_frame: str = self.planner_scene_frame
        self.astar_path_frame: str = self.planner_scene_frame

        self._reported_cv2_fallback = False
        self.cv2 = None

        if self.use_opencv:
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.cv2 = importlib.import_module("cv2")
            except Exception as exc:
                self.cv2 = None
                self.get_logger().warn(
                    "OpenCV is enabled but import failed, using NumPy fallback: %s" % str(exc)
                )

        self.action_pub = self.create_publisher(MarkerArray, self.action_marker_topic, 10)
        self.scene_pub = self.create_publisher(MarkerArray, self.planner_scene_marker_topic, 10)

        self.create_subscription(String, self.action_debug_topic, self._on_action_debug, 10)
        self.create_subscription(String, self.planner_scene_debug_topic, self._on_scene_debug, 10)
        self.create_subscription(OccupancyGrid, self.local_costmap_topic, self._on_local_costmap, 10)
        self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, 10)
        self.create_subscription(OccupancyGrid, self.astar_local_map_topic, self._on_astar_local_map, 10)
        self.create_subscription(LaserScan, self.scan_topic, self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Path, self.astar_path_topic, self._on_astar_path, 10)
        if self.visualize_joint_state_geometry:
            self.create_subscription(InterfaceJointState, self.joint_state_topic, self._on_joint_state, 10)
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
            elif p.name == "visualize_local_costmap":
                self.visualize_local_costmap = bool(p.value)
            elif p.name == "visualize_scan_obstacles":
                self.visualize_scan_obstacles = bool(p.value)
            elif p.name == "visualize_joint_state_geometry":
                self.visualize_joint_state_geometry = bool(p.value)
            elif p.name == "visualize_astar_path":
                self.visualize_astar_path = bool(p.value)
            elif p.name == "visualize_astar_local_map":
                self.visualize_astar_local_map = bool(p.value)
            elif p.name == "local_costmap_sample_step":
                self.local_costmap_sample_step = max(1, int(p.value))
            elif p.name == "local_costmap_max_cells":
                self.local_costmap_max_cells = max(100, int(p.value))
            elif p.name == "local_costmap_min_cost":
                self.local_costmap_min_cost = max(0, min(100, int(p.value)))
            elif p.name == "astar_local_map_sample_step":
                self.astar_local_map_sample_step = max(1, int(p.value))
            elif p.name == "astar_local_map_max_cells":
                self.astar_local_map_max_cells = max(100, int(p.value))
            elif p.name == "astar_local_map_min_cost":
                self.astar_local_map_min_cost = max(0, min(100, int(p.value)))

        return SetParametersResult(successful=True)

    def _on_local_costmap(self, msg: OccupancyGrid) -> None:
        self.last_costmap = msg
        self._publish_scene_markers()

    def _on_astar_local_map(self, msg: OccupancyGrid) -> None:
        self.last_astar_local_map = msg
        self._publish_scene_markers()

    def _on_map(self, msg: OccupancyGrid) -> None:
        self.map_frame = msg.header.frame_id if msg.header.frame_id else self.map_frame
        self.map_resolution = float(msg.info.resolution)
        self.map_origin_x = float(msg.info.origin.position.x)
        self.map_origin_y = float(msg.info.origin.position.y)
        self.map_height = int(msg.info.height)

        if msg.info.width == 0 or msg.info.height == 0:
            self.static_polygons = []
            self._publish_scene_markers()
            return

        grid = np.array(msg.data, dtype=np.int16).reshape((msg.info.height, msg.info.width))
        obstacle_img = np.zeros_like(grid, dtype=np.uint8)
        obstacle_img[grid > self.map_obstacle_threshold] = 255

        polygons: List[List[Tuple[float, float]]] = []
        if self.cv2 is not None:
            img_for_cv = np.flipud(obstacle_img)
            contours, _ = self.cv2.findContours(
                img_for_cv, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                peri = self.cv2.arcLength(contour, True)
                eps = self.poly_epsilon_ratio * peri
                approx = self.cv2.approxPolyDP(contour, eps, True)

                if len(approx) < 3:
                    continue

                poly_pts: List[Tuple[float, float]] = []
                for vertex in approx:
                    u = int(vertex[0][0])
                    v = int(vertex[0][1])
                    x_m, y_m = self._pixel_to_meter(u, v)
                    poly_pts.append((x_m, y_m))

                polygons.append(poly_pts)
        else:
            if not self._reported_cv2_fallback:
                self.get_logger().info("Using NumPy fallback for map polygon extraction")
                self._reported_cv2_fallback = True

            obstacle_mask = obstacle_img > 0
            polygons = self._extract_polygons_numpy_fallback(obstacle_mask)

        self.static_polygons = polygons
        self._publish_scene_markers()

    def _on_scan(self, msg: LaserScan) -> None:
        self.scan_frame = msg.header.frame_id if msg.header.frame_id else self.scan_frame

        points: List[Tuple[float, float]] = []
        angle = msg.angle_min
        for r in msg.ranges:
            if math.isfinite(r) and msg.range_min <= r <= msg.range_max:
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                points.append((x, y))
            angle += msg.angle_increment

        if not points:
            self.dynamic_circles = []
            self._publish_scene_markers()
            return

        clusters: List[List[Tuple[float, float]]] = []
        current_cluster: List[Tuple[float, float]] = [points[0]]

        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            dist = math.hypot(x2 - x1, y2 - y1)

            if dist < self.cluster_distance:
                current_cluster.append(points[i + 1])
            else:
                clusters.append(current_cluster)
                current_cluster = [points[i + 1]]

        clusters.append(current_cluster)

        circles: List[Tuple[float, float, float]] = []
        for cluster in clusters:
            if len(cluster) < self.min_cluster_points:
                continue
            if len(cluster) > self.max_cluster_points:
                continue

            arr = np.array(cluster, dtype=np.float64)
            span = float(np.linalg.norm(arr[-1] - arr[0]))
            if span > self.max_cluster_span:
                continue

            if self._is_wall_like_cluster(arr, span):
                continue

            cx = float(np.mean(arr[:, 0]))
            cy = float(np.mean(arr[:, 1]))
            dists = np.sqrt((arr[:, 0] - cx) ** 2 + (arr[:, 1] - cy) ** 2)
            radius = float(np.max(dists)) if len(dists) > 0 else 0.0

            if self.min_circle_radius <= radius <= self.max_circle_radius:
                circles.append((cx, cy, radius))

        self.dynamic_circles = circles
        self._publish_scene_markers()

    def _on_joint_state(self, msg: InterfaceJointState) -> None:
        self.scene_frame = msg.header.frame_id if msg.header.frame_id else self.map_frame

        polygons: List[List[Tuple[float, float]]] = []
        for poly in msg.poly_states:
            if len(poly.vertices) < 2:
                continue
            poly_pts: List[Tuple[float, float]] = []
            for v in poly.vertices:
                poly_pts.append((float(v.x), float(v.y)))
            polygons.append(poly_pts)

        walls: List[Tuple[float, float, float, float]] = []
        for wall in msg.walls:
            walls.append((float(wall.sx), float(wall.sy), float(wall.ex), float(wall.ey)))

        self.scene_polygons = polygons
        self.scene_walls = walls
        self._publish_scene_markers()

    def _on_astar_path(self, msg: Path) -> None:
        self.astar_path_frame = msg.header.frame_id if msg.header.frame_id else self.planner_scene_frame
        pts: List[Tuple[float, float]] = []
        for pose_stamped in msg.poses:
            pts.append((float(pose_stamped.pose.position.x), float(pose_stamped.pose.position.y)))
        self.astar_path_points = pts
        self._publish_scene_markers()

    def _on_action_debug(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"Invalid action debug payload: {exc}")
            return
        self.last_action_payload = payload
        self._publish_action_markers()
        self._publish_scene_markers()

    def _on_scene_debug(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"Invalid scene debug payload: {exc}")
            return
        self.last_scene_payload = payload
        self._publish_scene_markers()

    def _pixel_to_meter(self, u: int, v: int) -> Tuple[float, float]:
        row_occ = self.map_height - 1 - v
        x_world = self.map_origin_x + (float(u) + 0.5) * self.map_resolution
        y_world = self.map_origin_y + (float(row_occ) + 0.5) * self.map_resolution
        return x_world, y_world

    def _extract_polygons_numpy_fallback(self, obstacle_mask: np.ndarray) -> List[List[Tuple[float, float]]]:
        h, w = obstacle_mask.shape
        visited = np.zeros((h, w), dtype=np.uint8)
        polygons: List[List[Tuple[float, float]]] = []
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for row in range(h):
            for col in range(w):
                if not obstacle_mask[row, col] or visited[row, col]:
                    continue

                stack = [(row, col)]
                visited[row, col] = 1
                component: List[Tuple[int, int]] = []

                while stack:
                    r, c = stack.pop()
                    component.append((r, c))
                    for dr, dc in neighbors:
                        nr = r + dr
                        nc = c + dc
                        if nr < 0 or nr >= h or nc < 0 or nc >= w:
                            continue
                        if obstacle_mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = 1
                            stack.append((nr, nc))

                boundary_world: List[Tuple[float, float]] = []
                for r, c in component:
                    is_boundary = False
                    for dr, dc in neighbors:
                        nr = r + dr
                        nc = c + dc
                        if nr < 0 or nr >= h or nc < 0 or nc >= w or not obstacle_mask[nr, nc]:
                            is_boundary = True
                            break
                    if is_boundary:
                        x_m = self.map_origin_x + (float(c) + 0.5) * self.map_resolution
                        y_m = self.map_origin_y + (float(r) + 0.5) * self.map_resolution
                        boundary_world.append((x_m, y_m))

                if len(boundary_world) < 3:
                    continue

                hull = self._convex_hull(boundary_world)
                if len(hull) >= 3:
                    polygons.append(hull)

        return polygons

    def _convex_hull(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        pts = sorted(set(points))
        if len(pts) <= 2:
            return pts

        def cross(o: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower: List[Tuple[float, float]] = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
                lower.pop()
            lower.append(p)

        upper: List[Tuple[float, float]] = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
                upper.pop()
            upper.append(p)

        return lower[:-1] + upper[:-1]

    def _is_wall_like_cluster(self, arr: np.ndarray, span: float) -> bool:
        if arr.shape[0] < 3:
            return False

        centered = arr - np.mean(arr, axis=0)
        cov = np.cov(centered, rowvar=False)

        if cov.shape != (2, 2):
            return False

        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(np.abs(eigvals))

        small = float(eigvals[0])
        large = float(eigvals[1])
        linearity = large / (small + 1e-9)

        return linearity >= self.linearity_reject_ratio and span >= self.linearity_reject_span

    def _delete_all(self, frame_id: str) -> Marker:
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = rclpy.time.Time().to_msg()
        m.action = Marker.DELETEALL
        return m

    @staticmethod
    def _point(x: float, y: float, z: float = 0.05) -> RosPoint:
        return RosPoint(x=float(x), y=float(y), z=float(z))

    def _append_circle_segments(
        self,
        points: List[RosPoint],
        cx: float,
        cy: float,
        radius: float,
        z: float = 0.08,
        segments: int = 32,
    ) -> None:
        if radius <= 0.0:
            return
        for i in range(segments):
            a0 = 2.0 * math.pi * float(i) / float(segments)
            a1 = 2.0 * math.pi * float(i + 1) / float(segments)
            points.append(self._point(cx + radius * math.cos(a0), cy + radius * math.sin(a0), z))
            points.append(self._point(cx + radius * math.cos(a1), cy + radius * math.sin(a1), z))

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

        now = rclpy.time.Time().to_msg()
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
        payload = self.last_scene_payload or {}

        frame_id = str(payload.get("frame_id", self.planner_scene_frame))
        goal_reached = bool(payload.get("goal_reached", False))

        markers = MarkerArray()
        markers.markers.append(self._delete_all(frame_id))

        if goal_reached or not self.visualize_planner_scene:
            self.scene_pub.publish(markers)
            return

        now = rclpy.time.Time().to_msg()
        robot = payload.get("robot", {})
        goal = payload.get("goal", {})
        sub_goal = payload.get("sub_goal", {})
        bounds = payload.get("bounds", {})
        walls = payload.get("walls", [])
        obstacles = payload.get("obstacles", [])
        polygons = payload.get("polygons", [])
        humans = payload.get("humans", [])
        trajectory = payload.get("trajectory", [])
        reference_debug = payload.get("reference_debug", {})
        costmap_msg = self.last_costmap
        action_payload = self.last_action_payload or {}

        robot_radius = float(robot.get("radius", 0.25))

        robot_marker = Marker()
        # Gắn thẳng vào hệ quy chiếu của xe (base_footprint)
        robot_marker.header.frame_id = "base_footprint" 
        robot_marker.header.stamp = now
        robot_marker.ns = "planner_scene"
        robot_marker.id = 100
        robot_marker.type = Marker.CYLINDER
        robot_marker.action = Marker.ADD
        # Tâm của base_link luôn là (0, 0)
        robot_marker.pose.position.x = 0.0
        robot_marker.pose.position.y = 0.0
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

        # =========================================================
        # 2. VẼ MŨI TÊN HƯỚNG ĐI (BÁM THEO BASE_LINK)
        # =========================================================
        heading = Marker()
        heading.header.frame_id = "base_link" # Gắn vào xe
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
        # Mũi tên chỉ thẳng về phía trước theo trục X của xe
        heading.points = [self._point(0.0, 0.0, 0.1), self._point(0.5, 0.0, 0.1)]
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
        if self.visualize_joint_state_geometry:
            for w in self.scene_walls:
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
        if self.visualize_joint_state_geometry:
            for poly in self.scene_polygons:
                if len(poly) < 2:
                    continue
                for i in range(len(poly)):
                    p1 = poly[i]
                    p2 = poly[(i + 1) % len(poly)]
                    poly_marker.points.append(self._point(p1[0], p1[1]))
                    poly_marker.points.append(self._point(p2[0], p2[1]))
        markers.markers.append(poly_marker)

        map_poly_marker = Marker()
        map_poly_marker.header.frame_id = self.map_frame
        map_poly_marker.header.stamp = now
        map_poly_marker.ns = "planner_scene"
        map_poly_marker.id = 125
        map_poly_marker.type = Marker.LINE_LIST
        map_poly_marker.action = Marker.ADD
        map_poly_marker.scale.x = 0.04
        map_poly_marker.color.r = 0.0
        map_poly_marker.color.g = 1.0
        map_poly_marker.color.b = 1.0
        map_poly_marker.color.a = 0.85
        for poly in self.static_polygons:
            if len(poly) < 2:
                continue
            for i in range(len(poly)):
                p1 = poly[i]
                p2 = poly[(i + 1) % len(poly)]
                map_poly_marker.points.append(self._point(p1[0], p1[1], 0.03))
                map_poly_marker.points.append(self._point(p2[0], p2[1], 0.03))
        markers.markers.append(map_poly_marker)

        scan_marker = Marker()
        scan_marker.header.frame_id = self.scan_frame
        scan_marker.header.stamp = now
        scan_marker.ns = "planner_scene"
        scan_marker.id = 131
        scan_marker.type = Marker.SPHERE_LIST
        scan_marker.action = Marker.ADD
        scan_marker.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()
        scan_marker.scale.x = 0.16
        scan_marker.scale.y = 0.16
        scan_marker.scale.z = 0.10
        scan_marker.color.r = 1.0
        scan_marker.color.g = 0.35
        scan_marker.color.b = 0.20
        scan_marker.color.a = 0.75
        if self.visualize_scan_obstacles:
            for cx, cy, _ in self.dynamic_circles:
                scan_marker.points.append(self._point(cx, cy, self.dynamic_marker_z))
        markers.markers.append(scan_marker)

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

        action_frame = str(action_payload.get("frame_id", frame_id))
        action_obstacle_frame = str(action_payload.get("obstacle_frame", frame_id))
        valid_points = action_payload.get("valid_points", [])
        masked_points = action_payload.get("masked_points", [])
        selected_point = action_payload.get("selected_point", None)
        action_obstacles = action_payload.get("obstacles", [])
        mask_margin = float(action_payload.get("mask_margin", 0.35))
        human_safety_margin = float(action_payload.get("human_safety_margin", 0.0))
        map_margin = 0.5 * mask_margin

        rl_valid_marker = Marker()
        rl_valid_marker.header.frame_id = action_frame
        rl_valid_marker.header.stamp = now
        rl_valid_marker.ns = "rl_grid_valid"
        rl_valid_marker.id = 160
        rl_valid_marker.type = Marker.SPHERE_LIST
        rl_valid_marker.action = Marker.ADD
        rl_valid_marker.scale.x = 0.11
        rl_valid_marker.scale.y = 0.11
        rl_valid_marker.scale.z = 0.05
        rl_valid_marker.color.r = 0.0
        rl_valid_marker.color.g = 1.0
        rl_valid_marker.color.b = 0.25
        rl_valid_marker.color.a = 0.95
        for p in valid_points:
            if len(p) >= 2:
                rl_valid_marker.points.append(self._point(p[0], p[1], 0.12))
        markers.markers.append(rl_valid_marker)

        rl_masked_marker = Marker()
        rl_masked_marker.header.frame_id = action_frame
        rl_masked_marker.header.stamp = now
        rl_masked_marker.ns = "rl_grid_masked"
        rl_masked_marker.id = 161
        rl_masked_marker.type = Marker.SPHERE_LIST
        rl_masked_marker.action = Marker.ADD
        rl_masked_marker.scale.x = 0.10
        rl_masked_marker.scale.y = 0.10
        rl_masked_marker.scale.z = 0.05
        rl_masked_marker.color.r = 1.0
        rl_masked_marker.color.g = 0.05
        rl_masked_marker.color.b = 0.05
        rl_masked_marker.color.a = 0.85
        for p in masked_points:
            if len(p) >= 2:
                rl_masked_marker.points.append(self._point(p[0], p[1], 0.13))
        markers.markers.append(rl_masked_marker)

        rl_selected_marker = Marker()
        rl_selected_marker.header.frame_id = action_frame
        rl_selected_marker.header.stamp = now
        rl_selected_marker.ns = "rl_grid_selected"
        rl_selected_marker.id = 162
        rl_selected_marker.type = Marker.SPHERE
        rl_selected_marker.action = Marker.ADD
        rl_selected_marker.scale.x = 0.24
        rl_selected_marker.scale.y = 0.24
        rl_selected_marker.scale.z = 0.12
        rl_selected_marker.color.r = 0.1
        rl_selected_marker.color.g = 0.45
        rl_selected_marker.color.b = 1.0
        rl_selected_marker.color.a = 1.0
        if selected_point is not None and len(selected_point) >= 2:
            rl_selected_marker.pose.position.x = float(selected_point[0])
            rl_selected_marker.pose.position.y = float(selected_point[1])
            rl_selected_marker.pose.position.z = 0.18
        else:
            rl_selected_marker.action = Marker.DELETE
        markers.markers.append(rl_selected_marker)

        margin_marker = Marker()
        margin_obstacles = obstacles if obstacles else action_obstacles
        margin_marker.header.frame_id = frame_id if obstacles else action_obstacle_frame
        margin_marker.header.stamp = now
        margin_marker.ns = "planner_margins"
        margin_marker.id = 163
        margin_marker.type = Marker.LINE_LIST
        margin_marker.action = Marker.ADD
        margin_marker.scale.x = 0.035
        margin_marker.color.r = 1.0
        margin_marker.color.g = 0.15
        margin_marker.color.b = 0.05
        margin_marker.color.a = 0.9
        for obs in margin_obstacles:
            if len(obs) >= 3:
                radius = max(0.05, float(obs[2])) + mask_margin
                self._append_circle_segments(margin_marker.points, float(obs[0]), float(obs[1]), radius)
        for hum in humans:
            radius = max(0.05, float(hum.get("radius", 0.25)))
            self._append_circle_segments(
                margin_marker.points,
                float(hum.get("px", 0.0)),
                float(hum.get("py", 0.0)),
                radius,
                z=0.10,
            )
        markers.markers.append(margin_marker)

        wall_margin_marker = Marker()
        wall_margin_marker.header.frame_id = frame_id
        wall_margin_marker.header.stamp = now
        wall_margin_marker.ns = "wall_margins"
        wall_margin_marker.id = 164
        wall_margin_marker.type = Marker.LINE_LIST
        wall_margin_marker.action = Marker.ADD
        wall_margin_marker.scale.x = max(0.03, 2.0 * map_margin)
        wall_margin_marker.color.r = 1.0
        wall_margin_marker.color.g = 0.1
        wall_margin_marker.color.b = 0.0
        wall_margin_marker.color.a = 0.25
        for w in walls:
            if len(w) == 4:
                wall_margin_marker.points.append(self._point(w[0], w[1], 0.04))
                wall_margin_marker.points.append(self._point(w[2], w[3], 0.04))
        if self.visualize_joint_state_geometry:
            for w in self.scene_walls:
                wall_margin_marker.points.append(self._point(w[0], w[1], 0.04))
                wall_margin_marker.points.append(self._point(w[2], w[3], 0.04))
        markers.markers.append(wall_margin_marker)

        map_margin_marker = Marker()
        map_margin_marker.header.frame_id = self.map_frame
        map_margin_marker.header.stamp = now
        map_margin_marker.ns = "map_margins"
        map_margin_marker.id = 165
        map_margin_marker.type = Marker.LINE_LIST
        map_margin_marker.action = Marker.ADD
        map_margin_marker.scale.x = max(0.03, 2.0 * map_margin)
        map_margin_marker.color.r = 1.0
        map_margin_marker.color.g = 0.18
        map_margin_marker.color.b = 0.0
        map_margin_marker.color.a = 0.22
        for poly in self.static_polygons:
            if len(poly) < 2:
                continue
            for i in range(len(poly)):
                p1 = poly[i]
                p2 = poly[(i + 1) % len(poly)]
                map_margin_marker.points.append(self._point(p1[0], p1[1], 0.035))
                map_margin_marker.points.append(self._point(p2[0], p2[1], 0.035))
        markers.markers.append(map_margin_marker)

        costmap_marker = Marker()
        costmap_marker.header.frame_id = frame_id
        costmap_marker.header.stamp = now
        costmap_marker.ns = "planner_scene"
        costmap_marker.id = 145
        costmap_marker.type = Marker.CUBE_LIST
        costmap_marker.action = Marker.ADD
        costmap_marker.scale.x = 0.10
        costmap_marker.scale.y = 0.10
        costmap_marker.scale.z = 0.02
        costmap_marker.color.a = 0.0

        if self.visualize_local_costmap and costmap_msg is not None:
            cframe = costmap_msg.header.frame_id if costmap_msg.header.frame_id else frame_id
            costmap_marker.header.frame_id = cframe
            res = float(costmap_msg.info.resolution)
            width = int(costmap_msg.info.width)
            height = int(costmap_msg.info.height)
            ox = float(costmap_msg.info.origin.position.x)
            oy = float(costmap_msg.info.origin.position.y)
            data = costmap_msg.data

            if res > 0.0 and width > 0 and height > 0 and len(data) == width * height:
                costmap_marker.scale.x = res
                costmap_marker.scale.y = res
                costmap_marker.scale.z = max(0.01, 0.6 * res)

                step = max(1, int(self.local_costmap_sample_step))
                points: List[RosPoint] = []
                colors: List[ColorRGBA] = []
                max_cells = int(self.local_costmap_max_cells)
                min_cost = int(self.local_costmap_min_cost)

                for my in range(0, height, step):
                    row_offset = my * width
                    for mx in range(0, width, step):
                        idx = row_offset + mx
                        cost = int(data[idx])
                        if cost < min_cost:
                            continue

                        wx = ox + (mx + 0.5) * res
                        wy = oy + (my + 0.5) * res
                        points.append(self._point(wx, wy, 0.01))

                        norm = max(0.0, min(1.0, float(cost) / 100.0))
                        c = ColorRGBA()
                        c.r = float(norm)
                        c.g = float(1.0 - norm)
                        c.b = 0.15
                        c.a = float(0.15 + 0.50 * norm)
                        colors.append(c)

                        if len(points) >= max_cells:
                            break
                    if len(points) >= max_cells:
                        break

                costmap_marker.points = points
                costmap_marker.colors = colors

        markers.markers.append(costmap_marker)

        astar_local_map_marker = Marker()
        astar_local_map_marker.header.frame_id = frame_id
        astar_local_map_marker.header.stamp = now
        astar_local_map_marker.ns = "planner_scene"
        astar_local_map_marker.id = 146
        astar_local_map_marker.type = Marker.CUBE_LIST
        astar_local_map_marker.action = Marker.ADD
        astar_local_map_marker.scale.x = 0.10
        astar_local_map_marker.scale.y = 0.10
        astar_local_map_marker.scale.z = 0.02
        astar_local_map_marker.color.a = 0.0

        if self.visualize_astar_local_map and self.last_astar_local_map is not None:
            amap = self.last_astar_local_map
            aframe = amap.header.frame_id if amap.header.frame_id else frame_id
            astar_local_map_marker.header.frame_id = aframe
            ares = float(amap.info.resolution)
            awidth = int(amap.info.width)
            aheight = int(amap.info.height)
            aox = float(amap.info.origin.position.x)
            aoy = float(amap.info.origin.position.y)
            adata = amap.data

            if ares > 0.0 and awidth > 0 and aheight > 0 and len(adata) == awidth * aheight:
                astar_local_map_marker.scale.x = ares
                astar_local_map_marker.scale.y = ares
                astar_local_map_marker.scale.z = max(0.01, 0.5 * ares)

                astep = max(1, int(self.astar_local_map_sample_step))
                amax_cells = int(self.astar_local_map_max_cells)
                amin_cost = int(self.astar_local_map_min_cost)

                apts: List[RosPoint] = []
                acols: List[ColorRGBA] = []
                for my in range(0, aheight, astep):
                    row_offset = my * awidth
                    for mx in range(0, awidth, astep):
                        idx = row_offset + mx
                        cost = int(adata[idx])
                        if cost < amin_cost:
                            continue

                        wx = aox + (mx + 0.5) * ares
                        wy = aoy + (my + 0.5) * ares
                        apts.append(self._point(wx, wy, 0.02))

                        norm = max(0.0, min(1.0, float(cost) / 100.0))
                        c = ColorRGBA()
                        c.r = 0.15
                        c.g = float(0.30 + 0.70 * (1.0 - norm))
                        c.b = 1.0
                        c.a = float(0.10 + 0.35 * norm)
                        acols.append(c)

                        if len(apts) >= amax_cells:
                            break
                    if len(apts) >= amax_cells:
                        break

                astar_local_map_marker.points = apts
                astar_local_map_marker.colors = acols

        markers.markers.append(astar_local_map_marker)

        traj_marker = Marker()
        traj_marker.header.frame_id = frame_id
        traj_marker.header.stamp = now
        traj_marker.ns = "planner_scene"
        traj_marker.id = 150
        traj_marker.type = Marker.LINE_STRIP
        traj_marker.action = Marker.ADD
        traj_marker.scale.x = 0.08
        delta_rms = float(reference_debug.get("stability_delta_rms_m", 0.0))
        jitter_threshold = max(1e-6, float(reference_debug.get("stability_warn_threshold_m", 0.35)))
        stable = bool(float(reference_debug.get("stable", 1.0)) > 0.5)
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

        reference_info = Marker()
        reference_info.header.frame_id = frame_id
        reference_info.header.stamp = now
        reference_info.ns = "planner_scene"
        reference_info.id = 151
        reference_info.type = Marker.TEXT_VIEW_FACING
        reference_info.action = Marker.ADD
        reference_info.scale.z = 0.25
        if stable:
            reference_info.color.r = 0.20
            reference_info.color.g = 1.0
            reference_info.color.b = 0.25
        else:
            reference_info.color.r = 1.0
            reference_info.color.g = 0.20
            reference_info.color.b = 0.20
        reference_info.color.a = 0.95

        rx = float(robot.get("x", 0.0))
        ry = float(robot.get("y", 0.0))
        reference_info.pose.position.x = rx
        reference_info.pose.position.y = ry
        reference_info.pose.position.z = 1.0

        expected_h = int(float(reference_debug.get("expected_horizon_steps", 0.0)))
        received_h = int(float(reference_debug.get("received_control_steps", 0.0)))
        pred_pts = int(float(reference_debug.get("predicted_points", 0.0)))
        coverage = 100.0 * float(reference_debug.get("horizon_coverage", 0.0))
        traj_len = float(reference_debug.get("trajectory_length_m", 0.0))
        stability_tag = "STABLE" if stable else "UNSTABLE"
        reference_info.text = (
            f"REF H={expected_h} ctrl={received_h} pts={pred_pts} cov={coverage:.0f}%\\n"
            f"len={traj_len:.2f}m dRMS={delta_rms:.3f}m -> {stability_tag}"
        )
        markers.markers.append(reference_info)

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

        astar_path_marker = Marker()
        astar_path_marker.header.frame_id = self.astar_path_frame
        astar_path_marker.header.stamp = now
        astar_path_marker.ns = "planner_scene"
        astar_path_marker.id = 153
        astar_path_marker.type = Marker.LINE_STRIP
        astar_path_marker.action = Marker.ADD
        astar_path_marker.scale.x = 0.05
        astar_path_marker.color.r = 1.0
        astar_path_marker.color.g = 0.95
        astar_path_marker.color.b = 0.10
        astar_path_marker.color.a = 0.95
        if self.visualize_astar_path:
            for px, py in self.astar_path_points:
                astar_path_marker.points.append(self._point(px, py, 0.12))
        markers.markers.append(astar_path_marker)

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
