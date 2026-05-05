#!/usr/bin/python3
import json
import math
import importlib
import io
from typing import Dict, List, Optional, Tuple
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import rclpy
import rclpy.time
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point as RosPoint
from interfaces.msg import JointState as InterfaceJointState
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan

# Dùng Mutex để chống xung đột luồng khi Timer và Callback cùng truy cập dữ liệu
import threading


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
        self.declare_parameter("scan_grid_range", 5.0)
        self.declare_parameter("scan_marker_lifetime", 0.1)  # Tăng chút xíu để tránh nhấp nháy

        self.declare_parameter("action_debug_topic", "/debug/policy_actions_scene")
        self.declare_parameter("planner_scene_debug_topic", "/debug/planner_scene")

        # Caching parameters
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
        
        self.local_costmap_sample_step = max(1, self.get_parameter("local_costmap_sample_step").get_parameter_value().integer_value)
        self.local_costmap_max_cells = max(100, self.get_parameter("local_costmap_max_cells").get_parameter_value().integer_value)
        self.local_costmap_min_cost = max(0, min(100, self.get_parameter("local_costmap_min_cost").get_parameter_value().integer_value))
        
        self.astar_local_map_sample_step = max(1, self.get_parameter("astar_local_map_sample_step").get_parameter_value().integer_value)
        self.astar_local_map_max_cells = max(100, self.get_parameter("astar_local_map_max_cells").get_parameter_value().integer_value)
        self.astar_local_map_min_cost = max(0, min(100, self.get_parameter("astar_local_map_min_cost").get_parameter_value().integer_value))

        self.map_obstacle_threshold = int(self.get_parameter("map_obstacle_threshold").get_parameter_value().integer_value)
        self.poly_epsilon_ratio = float(self.get_parameter("poly_epsilon_ratio").get_parameter_value().double_value)
        self.cluster_distance = float(self.get_parameter("cluster_distance").get_parameter_value().double_value)
        self.dynamic_marker_z = float(self.get_parameter("dynamic_marker_z").get_parameter_value().double_value)
        self.use_opencv = bool(self.get_parameter("use_opencv").get_parameter_value().bool_value)
        self.min_cluster_points = int(self.get_parameter("min_cluster_points").get_parameter_value().integer_value)
        self.max_cluster_points = int(self.get_parameter("max_cluster_points").get_parameter_value().integer_value)
        self.max_cluster_span = float(self.get_parameter("max_cluster_span").get_parameter_value().double_value)
        self.min_circle_radius = float(self.get_parameter("min_circle_radius").get_parameter_value().double_value)
        self.max_circle_radius = float(self.get_parameter("max_circle_radius").get_parameter_value().double_value)
        self.linearity_reject_ratio = float(self.get_parameter("linearity_reject_ratio").get_parameter_value().double_value)
        self.linearity_reject_span = float(self.get_parameter("linearity_reject_span").get_parameter_value().double_value)
        self.scan_grid_range = float(self.get_parameter("scan_grid_range").get_parameter_value().double_value)
        self.scan_marker_lifetime = float(self.get_parameter("scan_marker_lifetime").get_parameter_value().double_value)

        self.action_debug_topic = self.get_parameter("action_debug_topic").get_parameter_value().string_value
        self.planner_scene_debug_topic = self.get_parameter("planner_scene_debug_topic").get_parameter_value().string_value

        # Data states
        self.data_lock = threading.Lock() # Lock để bảo vệ dữ liệu khi đa luồng truy cập
        self.last_action_payload: Optional[Dict] = None
        self.last_scene_payload: Optional[Dict] = None
        self.last_costmap: Optional[OccupancyGrid] = None
        self.last_astar_local_map: Optional[OccupancyGrid] = None
        self.last_scan_msg: Optional[LaserScan] = None # Bộ đệm lưu trữ scan thô
        
        self.static_polygons: List[List[Tuple[float, float]]] = []
        self.scene_polygons: List[List[Tuple[float, float]]] = []
        self.scene_walls: List[Tuple[float, float, float, float]] = []
        self.astar_path_points: List[Tuple[float, float]] = []

        self.map_resolution: float = 0.0
        self.map_origin_x: float = 0.0
        self.map_origin_y: float = 0.0
        self.map_height: int = 0

        self.map_frame: str = self.planner_scene_frame
        self.scan_frame: str = "base_scan"

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

        # Publishers
        self.action_pub = self.create_publisher(MarkerArray, self.action_marker_topic, 10)
        self.scene_pub = self.create_publisher(MarkerArray, self.planner_scene_marker_topic, 10)

        # Subscribers
        self.create_subscription(String, self.action_debug_topic, self._on_action_debug, 10)
        self.create_subscription(String, self.planner_scene_debug_topic, self._on_scene_debug, 10)
        self.create_subscription(OccupancyGrid, self.local_costmap_topic, self._on_local_costmap, 10)
        self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, 10)
        self.create_subscription(OccupancyGrid, self.astar_local_map_topic, self._on_astar_local_map, 10)
        
        # [QUAN TRỌNG]: Tách Callback Scan thành việc cực nhẹ: Chỉ lưu msg, không tính toán
        self.create_subscription(LaserScan, self.scan_topic, self._on_scan, qos_profile_sensor_data)
        
        self.create_subscription(Path, self.astar_path_topic, self._on_astar_path, 10)
        if self.visualize_joint_state_geometry:
            self.create_subscription(InterfaceJointState, self.joint_state_topic, self._on_joint_state, 10)
        self.add_on_set_parameters_callback(self._on_parameters_changed)

        # [QUAN TRỌNG]: Tạo Timer độc lập ở tốc độ 15Hz để gom xử lý và Publish mượt mà
        self.publish_timer = self.create_timer(1.0 / 15.0, self._process_and_publish_scene)

        self.get_logger().info(
            f"RViz visualizer started. action_debug={self.action_debug_topic}, scene_debug={self.planner_scene_debug_topic}"
        )

    def _on_parameters_changed(self, params):
        with self.data_lock:
            for p in params:
                if hasattr(self, p.name):
                    # Tự động parse kiểu dữ liệu dựa trên thuộc tính hiện tại
                    current_val = getattr(self, p.name)
                    if isinstance(current_val, bool):
                        setattr(self, p.name, bool(p.value))
                    elif isinstance(current_val, int):
                        setattr(self, p.name, int(p.value))
                    elif isinstance(current_val, float):
                        setattr(self, p.name, float(p.value))
        return SetParametersResult(successful=True)

    def _on_local_costmap(self, msg: OccupancyGrid) -> None:
        with self.data_lock:
            self.last_costmap = msg

    def _on_astar_local_map(self, msg: OccupancyGrid) -> None:
        with self.data_lock:
            self.last_astar_local_map = msg

    def _on_map(self, msg: OccupancyGrid) -> None:
        with self.data_lock:
            self.map_frame = msg.header.frame_id if msg.header.frame_id else self.map_frame
            self.map_resolution = float(msg.info.resolution)
            self.map_origin_x = float(msg.info.origin.position.x)
            self.map_origin_y = float(msg.info.origin.position.y)
            self.map_height = int(msg.info.height)

            if msg.info.width == 0 or msg.info.height == 0:
                self.static_polygons = []
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

    def _on_scan(self, msg: LaserScan) -> None:
        """
        [FIX]: Callback này CHỈ THU THẬP dữ liệu thô. KHÔNG tính toán gì ở đây để tránh nghẽn.
        """
        with self.data_lock:
            self.last_scan_msg = msg

    def _on_joint_state(self, msg: InterfaceJointState) -> None:
        with self.data_lock:
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

    def _on_astar_path(self, msg: Path) -> None:
        with self.data_lock:
            pts: List[Tuple[float, float]] = []
            for pose_stamped in msg.poses:
                pts.append((float(pose_stamped.pose.position.x), float(pose_stamped.pose.position.y)))
            self.astar_path_points = pts

    def _on_action_debug(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"Invalid action debug payload: {exc}")
            return
        with self.data_lock:
            self.last_action_payload = payload

    def _on_scene_debug(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"Invalid scene debug payload: {exc}")
            return
        with self.data_lock:
            self.last_scene_payload = payload

    # =========================================================================
    # CORE PROCESSING & PUBLISHING LOUPE
    # =========================================================================
    def _process_and_publish_scene(self) -> None:
        """
        Hàm được gọi bởi Timer (15Hz). Đồng bộ hóa việc lấy dữ liệu, 
        tính toán cluster và vẽ RViz ở ĐÚNG MỘT NƠI DUY NHẤT.
        """
        with self.data_lock:
            # Copy snapshot của dữ liệu hiện tại để tránh race condition
            scan_msg = self.last_scan_msg
            action_payload = self.last_action_payload
            scene_payload = self.last_scene_payload
            
        # 1. Vẽ lưới Action (Các hạt màu xanh, đỏ)
        if action_payload is not None:
            self._publish_action_markers(action_payload)

        # 2. Xử lý Scan và Vẽ Map/Obstacles
        if scene_payload is not None:
            self._publish_scene_markers(scene_payload)

    def _process_scan_clusters(self, msg: LaserScan) -> List[Tuple[float, float]]:
        """Tính toán Clustering từ dữ liệu LaserScan thô, lọc bỏ điểm quá xa"""
        points_laser_frame: List[Tuple[float, float]] = []
        angle = msg.angle_min
        grid_range_sq = self.scan_grid_range * self.scan_grid_range 
        
        # Downsample mảng Lidar (nhảy 2 điểm một) để giảm tải tính toán
        step = 2 
        for i in range(0, len(msg.ranges), step):
            r_dist = msg.ranges[i]
            if math.isfinite(r_dist) and msg.range_min <= r_dist <= msg.range_max:
                if r_dist * r_dist <= grid_range_sq:
                    lx = r_dist * math.cos(angle)
                    ly = r_dist * math.sin(angle)
                    points_laser_frame.append((lx, ly))
            angle += msg.angle_increment * step

        if not points_laser_frame:
            return []

        # Tối ưu Clustering nội bộ
        clusters: List[List[Tuple[float, float]]] = []
        current_cluster: List[Tuple[float, float]] = [points_laser_frame[0]]

        for i in range(len(points_laser_frame) - 1):
            x1, y1 = points_laser_frame[i]
            x2, y2 = points_laser_frame[i + 1]
            dist = math.hypot(x2 - x1, y2 - y1)

            if dist < self.cluster_distance:
                current_cluster.append(points_laser_frame[i + 1])
            else:
                if len(current_cluster) >= self.min_cluster_points:
                    clusters.append(current_cluster)
                current_cluster = [points_laser_frame[i + 1]]

        if len(current_cluster) >= self.min_cluster_points:
            clusters.append(current_cluster)

        circles: List[Tuple[float, float]] = []
        for cluster in clusters:
            if len(cluster) > self.max_cluster_points:
                continue

            arr = np.array(cluster, dtype=np.float32)
            span = float(np.linalg.norm(arr[-1] - arr[0]))
            if span > self.max_cluster_span:
                continue

            if self._is_wall_like_cluster(arr.astype(np.float64), span):
                continue

            cx = float(np.mean(arr[:, 0]))
            cy = float(np.mean(arr[:, 1]))
            dists = np.sqrt((arr[:, 0] - cx) ** 2 + (arr[:, 1] - cy) ** 2)
            radius = float(np.max(dists)) if len(dists) > 0 else 0.0

            if self.min_circle_radius <= radius <= self.max_circle_radius:
                circles.append((cx, cy))

        return circles

    # =========================================================================
    # PUBLISHERS
    # =========================================================================
    def _publish_action_markers(self, payload: dict) -> None:
        frame_id = str(payload.get("frame_id", self.action_marker_frame))
        goal_reached = bool(payload.get("goal_reached", False))

        markers = MarkerArray()
        if goal_reached or not self.visualize_actions:
            self.action_pub.publish(markers)
            return
        
        # [QUAN TRỌNG]: Dùng chung một Timestamp duy nhất (Now) cho các marker thuộc Action
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

    def _publish_scene_markers(self, payload: dict) -> None:
        frame_id = str(payload.get("frame_id", self.planner_scene_frame))
        goal_reached = bool(payload.get("goal_reached", False))

        markers = MarkerArray()
        markers.markers.append(self._delete_all(frame_id))

        if goal_reached or not self.visualize_planner_scene:
            self.scene_pub.publish(markers)
            return

        # Dùng chung Timestamp của Action/Scene thay vì scan_stamp cho các vật cản tĩnh
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

        robot_radius = float(robot.get("radius", 0.25))

        robot_marker = Marker()
        robot_marker.header.frame_id = "base_link" 
        robot_marker.header.stamp = now
        robot_marker.ns = "planner_scene"
        robot_marker.id = 100
        robot_marker.type = Marker.CYLINDER
        robot_marker.action = Marker.ADD
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

        heading = Marker()
        heading.header.frame_id = "base_link"
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

        self.scene_pub.publish(markers)

    # Các hàm toán hình học (giữ nguyên không thay đổi)
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

    def _delete_all(self, frame_id: str, ns: str = "planner_scene") -> Marker:
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = rclpy.time.Time().to_msg()
        m.ns = ns
        m.action = Marker.DELETEALL
        return m

    @staticmethod
    def _point(x: float, y: float, z: float = 0.05) -> RosPoint:
        return RosPoint(x=float(x), y=float(y), z=float(z))


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