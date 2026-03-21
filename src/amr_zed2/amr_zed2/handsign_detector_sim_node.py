"""
ZED 2 Hand Sign Detection and Person Tracking Node - SIMULATION VERSION
Uses ROS topics instead of ZED SDK for Gazebo simulation compatibility
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import message_filters

import cv2
from ultralytics import YOLO
from cv_bridge import CvBridge

import tf2_ros
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion, Point, Vector3
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from amr_interfaces.msg import HumanState, Obstacles, CircleObstacle
import math

from .botsort_handler import BoTSortTracker
from .velocity_filter import VelocityFilter
from .appearance_feature_extractor import FeatureExtractorLightweight
from .deep_nms import DeepNMSOptimizer
from .ctrv_ekf import MultiObjectCTRVTracker


@dataclass(slots=True)
class TrackedObject:
    """Tracked object state in global frame"""
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in image
    center_3d_camera: Tuple[float, float, float]  # 3D position in camera frame
    center_3d_global: Tuple[float, float, float]  # 3D position in global frame
    velocity_global: Tuple[float, float, float]  # velocity in global frame
    confidence: float


class HandSignDetectorSimNode(Node):
    """
    ROS2 Node for hand sign detection and person tracking - SIMULATION VERSION

    Uses ROS image topics instead of ZED SDK for Gazebo simulation.
    Works with simulated stereo camera in Gazebo.

    Subscribed Topics:
    - /zed2/left/image_raw (Image)
    - /zed2/left/camera_info (CameraInfo)
    - /camera/points (PointCloud2) - for depth information

    Published Topics:
    - /tracking/main_person (HumanState)
    - /tracking/obstacles (ObstacleStateArray)
    - /navigation/goal (PoseStamped)
    """

    def __init__(self):
        super().__init__('handsign_detector_sim')

        # Declare parameters
        self.declare_parameter('hand_model_path', 'handsign.pt')
        self.declare_parameter('person_model_path', 'yolov8n.pt')
        self.declare_parameter('detection_conf', 0.5)
        self.declare_parameter('nms_threshold', 0.45)
        self.declare_parameter('person_radius', 0.3)
        self.declare_parameter('enable_visualization', False)
        self.declare_parameter('camera_frame', 'zed2_left_camera_optical_frame')
        self.declare_parameter('global_frame', 'odom')
        self.declare_parameter('gesture_velocity_threshold', 0.3)  # Ignore gesture if person moving faster

        # Get parameters
        hand_model = self.get_parameter('hand_model_path').value
        person_model = self.get_parameter('person_model_path').value
        self._detect_conf = self.get_parameter('detection_conf').value
        self._nms_threshold = self.get_parameter('nms_threshold').value
        self._person_radius = self.get_parameter('person_radius').value
        self._enable_viz = self.get_parameter('enable_visualization').value
        self._camera_frame = self.get_parameter('camera_frame').value
        self._global_frame = self.get_parameter('global_frame').value
        self._gesture_velocity_threshold = self.get_parameter('gesture_velocity_threshold').value

        # Track person velocity for gesture filtering
        self._current_person_velocity = 0.0

        # TF2 for frame transformation
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # CV Bridge
        self._bridge = CvBridge()

        # Camera calibration (will be updated from CameraInfo)
        self._cam_fx = 1000.0
        self._cam_fy = 1000.0
        self._cam_cx = 640.0
        self._cam_cy = 360.0
        self._camera_info_received = False

        # Depth data from PointCloud2
        self._depth_data = None
        self._depth_timestamp = None

        # Load YOLO models
        self.get_logger().info('Loading YOLO models for simulation...')
        self._hand_model = YOLO(self._resolve_model_path(hand_model))
        self._person_model = YOLO(self._resolve_model_path(person_model))

        # Initialize BoT-SORT tracker
        self.get_logger().info('Initializing BoT-SORT tracker...')
        self._person_tracker = BoTSortTracker(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
            use_appearance=True,
            appearance_weight=0.5
        )

        # Hand sign detection state machine
        self._hand_sign_frame_skip = 10
        self._frame_counter = 0
        self._gesture_state = "IDLE"
        self._gesture_timer_start = None
        self._gesture_hold_threshold = 3.0

        # Tracking state machine
        self._tracking_state = "IDLE"
        self._frames_without_detection = 0
        self._lost_threshold = 30

        # CTRV EKF tracker
        self._ctrv_tracker = MultiObjectCTRVTracker(dt=1/30.0)
        self._prediction_horizon = 5.0

        # Last known state
        self._last_known_position_global = None
        self._last_known_velocity = None

        # Feature extractor and NMS
        self._feature_extractor = FeatureExtractorLightweight(n_bins=16)
        self._nms = DeepNMSOptimizer()

        # Velocity filters
        self._velocity_filters: Dict[int, VelocityFilter] = {}
        self._prev_positions: Dict[int, np.ndarray] = {}
        self._last_timestamps: Dict[int, float] = {}

        # QoS for simulation
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self._sub_camera_info = self.create_subscription(
            CameraInfo,
            '/zed2/left/camera_info',
            self._camera_info_callback,
            qos_sensor
        )

        self._sub_image = self.create_subscription(
            Image,
            '/zed2/left/image_raw',
            self._image_callback,
            qos_sensor
        )

        self._sub_pointcloud = self.create_subscription(
            PointCloud2,
            '/camera/points',
            self._pointcloud_callback,
            qos_sensor
        )

        # Publishers
        self._pub_main_person = self.create_publisher(
            HumanState, '/tracking/main_person', qos_reliable)
        self._pub_obstacles = self.create_publisher(
            Obstacles, '/tracking/obstacles', qos_reliable)
        self._pub_lost_goal = self.create_publisher(
            PoseStamped, '/navigation/goal', qos_reliable)

        self.get_logger().info('HandSign Detector (SIM) initialized - waiting for camera data...')

    def _resolve_model_path(self, model_path: str) -> str:
        """Resolve model path"""
        import os

        if os.path.isabs(model_path) and os.path.exists(model_path):
            return model_path

        if os.path.exists(model_path):
            return model_path

        try:
            from ament_index_python.packages import get_package_share_directory
            pkg_dir = get_package_share_directory('amr_zed2')
            models_path = os.path.join(pkg_dir, 'models', model_path)
            if os.path.exists(models_path):
                return models_path
        except Exception:
            pass

        raise FileNotFoundError(f'Model not found: {model_path}')

    def _camera_info_callback(self, msg: CameraInfo):
        """Update camera calibration from CameraInfo"""
        if not self._camera_info_received:
            self._cam_fx = msg.k[0]
            self._cam_fy = msg.k[4]
            self._cam_cx = msg.k[2]
            self._cam_cy = msg.k[5]
            self._camera_info_received = True
            self.get_logger().info(
                f'Camera calibration received: fx={self._cam_fx:.1f}, '
                f'fy={self._cam_fy:.1f}, cx={self._cam_cx:.1f}, cy={self._cam_cy:.1f}'
            )

    def _pointcloud_callback(self, msg: PointCloud2):
        """Process PointCloud2 to extract depth information"""
        import struct

        # Parse PointCloud2 to get depth at each pixel
        width = msg.width
        height = msg.height
        point_step = msg.point_step
        row_step = msg.row_step

        # Create depth image from point cloud
        depth_image = np.zeros((height, width), dtype=np.float32)

        # Find xyz field offsets
        x_offset = y_offset = z_offset = 0
        for field in msg.fields:
            if field.name == 'x':
                x_offset = field.offset
            elif field.name == 'y':
                y_offset = field.offset
            elif field.name == 'z':
                z_offset = field.offset

        data = np.frombuffer(msg.data, dtype=np.uint8)

        for v in range(height):
            for u in range(width):
                idx = v * row_step + u * point_step
                if idx + z_offset + 4 <= len(data):
                    z = struct.unpack('f', data[idx + z_offset:idx + z_offset + 4])[0]
                    if not np.isnan(z) and not np.isinf(z):
                        depth_image[v, u] = z

        self._depth_data = depth_image
        self._depth_timestamp = msg.header.stamp

    def _image_callback(self, msg: Image):
        """Process incoming image"""
        if not self._camera_info_received:
            return

        # Convert ROS Image to OpenCV
        try:
            # Use passthrough and handle encoding manually
            if msg.encoding == 'rgb8':
                frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            elif msg.encoding == 'rgba8':
                frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif msg.encoding == 'bgra8':
                frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:
                # Try passthrough for other encodings
                frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                if len(frame.shape) == 2:  # Grayscale
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:  # 4 channels
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            if frame is None or frame.size == 0:
                self.get_logger().warn('Received empty frame')
                return

        except Exception as e:
            self.get_logger().error(f'Failed to convert image (encoding={msg.encoding}): {e}')
            return

        timestamp = msg.header.stamp
        current_time = timestamp.sec + timestamp.nanosec / 1e9

        # Person Detection
        person_results = self._person_model.predict(
            frame, conf=self._detect_conf, verbose=False)[0]

        # Extract person detections (class 0 = Person)
        person_detections = []
        for det in person_results.boxes.data:
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            if int(cls) == 0:
                person_detections.append([x1, y1, x2, y2, conf])

        if person_detections:
            person_detections = np.array(person_detections)
        else:
            person_detections = np.empty((0, 5))

        # Apply NMS
        if len(person_detections) > 0:
            person_detections = self._nms.deep_nms_spatial(
                person_detections, self._nms_threshold)

        # Hand sign detection
        self._frame_counter += 1
        detected_gesture_class = None

        # Only detect gestures if person is standing still or moving slowly
        # This prevents false positives from walking arm swing
        person_is_stationary = self._current_person_velocity < self._gesture_velocity_threshold

        if self._frame_counter % self._hand_sign_frame_skip == 0 and person_is_stationary:
            hand_results = self._hand_model.predict(
                frame, conf=self._detect_conf, verbose=False)[0]
            if len(hand_results.boxes) > 0:
                best_det = hand_results.boxes.data[hand_results.boxes.conf.argmax()]
                detected_gesture_class = int(best_det[5].cpu().numpy())

        # Gesture state machine
        elapsed = current_time - self._gesture_timer_start if self._gesture_timer_start else 0

        if detected_gesture_class == 0:  # START
            if self._gesture_state != "START_HELD":
                self._gesture_state = "START_HELD"
                self._gesture_timer_start = current_time
                self.get_logger().info('START gesture detected')
            elif elapsed >= self._gesture_hold_threshold and self._tracking_state == "IDLE":
                self._tracking_state = "TRACKING"
                self._gesture_state = "IDLE"
                self._gesture_timer_start = None
                self._velocity_filters.clear()
                self._prev_positions.clear()
                self._last_timestamps.clear()
                self._ctrv_tracker = MultiObjectCTRVTracker(dt=1/30.0)
                self.get_logger().warn('TRACKING STARTED')

        elif detected_gesture_class == 1:  # STOP
            if self._gesture_state != "STOP_HELD":
                self._gesture_state = "STOP_HELD"
                self._gesture_timer_start = current_time
                self.get_logger().info('STOP gesture detected')
            elif elapsed >= self._gesture_hold_threshold and self._tracking_state == "TRACKING":
                self._tracking_state = "STOPPED"
                self._gesture_state = "IDLE"
                self._gesture_timer_start = None
                self._ctrv_tracker = MultiObjectCTRVTracker(dt=1/30.0)
                self.get_logger().warn('TRACKING STOPPED')
        else:
            if self._gesture_state != "IDLE":
                self._gesture_state = "IDLE"
                self._gesture_timer_start = None

        # Tracking state machine
        tracked_objects = []

        if len(person_detections) > 0 and self._tracking_state in ["TRACKING", "RE_TRACKING"]:
            self._frames_without_detection = 0
            if self._tracking_state == "RE_TRACKING":
                self._tracking_state = "TRACKING"
                self.get_logger().warn('Person re-detected - resuming tracking')

        elif self._tracking_state in ["TRACKING", "RE_TRACKING"]:
            self._frames_without_detection += 1

            if self._frames_without_detection == 1 and self._tracking_state == "TRACKING":
                self._tracking_state = "LOST"
                self.get_logger().warn('Person LOST')

            if self._frames_without_detection >= self._lost_threshold and self._tracking_state == "LOST":
                if self._last_known_position_global is not None:
                    self._publish_lost_goal(self._last_known_position_global.copy(), timestamp)

        # Extract appearance features
        if len(person_detections) > 0:
            person_features = self._feature_extractor.extract(
                frame, person_detections[:, :4])
        else:
            person_features = None

        # Update tracker
        if self._tracking_state in ["TRACKING", "RE_TRACKING"]:
            tracks = self._person_tracker.update(
                person_detections, features=person_features)
        else:
            tracks = []

        # Estimate 3D positions
        for track in tracks:
            x1, y1, x2, y2, tid = track
            person_center_x = (x1 + x2) / 2
            person_center_y = (y1 + y2) / 2
            person_bbox = (x1, y1, x2, y2)

            # Get depth
            depth_val = self._get_depth_at_pixel(
                int(person_center_x), int(person_center_y))
            if depth_val is None or depth_val <= 0 or depth_val > 10:
                continue

            # Backproject to 3D
            x3d_cam = (person_center_x - self._cam_cx) * depth_val / self._cam_fx
            y3d_cam = (person_center_y - self._cam_cy) * depth_val / self._cam_fy
            z3d_cam = depth_val

            # Transform to global frame
            result = self._transform_to_global(x3d_cam, y3d_cam, z3d_cam, timestamp)
            if result is None:
                continue
            x3d_global, y3d_global, z3d_global = result

            pos_global = np.array([x3d_global, y3d_global, z3d_global])

            # Velocity estimation
            if tid not in self._velocity_filters:
                self._velocity_filters[tid] = VelocityFilter(alpha=0.3)
                self._prev_positions[tid] = pos_global.copy()
                self._last_timestamps[tid] = current_time

            dt = current_time - self._last_timestamps[tid]
            if dt > 0:
                vel_raw = (pos_global - self._prev_positions[tid]) / dt
                vel_filtered = self._velocity_filters[tid].update(vel_raw)
            else:
                vel_filtered = self._velocity_filters[tid].filtered_velocity or np.zeros(3)

            self._prev_positions[tid] = pos_global.copy()
            self._last_timestamps[tid] = current_time

            # Update CTRV tracker
            if self._tracking_state in ["TRACKING", "RE_TRACKING"]:
                pos_2d = np.array([pos_global[0], pos_global[1]])
                vel_2d = np.array([vel_filtered[0], vel_filtered[1]])
                self._ctrv_tracker.update(int(tid), pos_2d, vel_2d)

                self._last_known_position_global = pos_global.copy()
                self._last_known_velocity = vel_filtered.copy()

            tracked_objects.append(TrackedObject(
                track_id=int(tid),
                bbox=person_bbox,
                center_3d_camera=(x3d_cam, y3d_cam, z3d_cam),
                center_3d_global=(x3d_global, y3d_global, z3d_global),
                velocity_global=tuple(vel_filtered),
                confidence=0.8
            ))

        # Update current person velocity for gesture filtering
        if tracked_objects:
            # Use velocity of closest person (main target)
            main_person = min(tracked_objects, key=lambda obj: obj.center_3d_global[2])
            vel = main_person.velocity_global
            self._current_person_velocity = math.sqrt(vel[0]**2 + vel[1]**2)
        else:
            self._current_person_velocity = 0.0

        # Publish tracking data
        self._publish_tracking(tracked_objects, timestamp)

        # Visualization
        if self._enable_viz:
            self._visualize(frame, tracked_objects)

    def _get_depth_at_pixel(self, u: int, v: int) -> Optional[float]:
        """Get depth value at pixel (u, v) from point cloud"""
        if self._depth_data is None:
            return None

        h, w = self._depth_data.shape
        if 0 <= u < w and 0 <= v < h:
            depth = self._depth_data[v, u]
            if depth > 0:
                return float(depth)
        return None

    def _transform_to_global(self, x_cam, y_cam, z_cam, timestamp):
        """Transform 3D point from camera frame to global frame"""
        try:
            transform = self._tf_buffer.lookup_transform(
                self._global_frame,
                self._camera_frame,
                timestamp,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )

            from tf2_geometry_msgs import do_transform_point

            point_stamped = PointStamped()
            point_stamped.header.frame_id = self._camera_frame
            point_stamped.header.stamp = timestamp
            point_stamped.point.x = float(x_cam)
            point_stamped.point.y = float(y_cam)
            point_stamped.point.z = float(z_cam)

            transformed_point = do_transform_point(point_stamped, transform)

            return (
                transformed_point.point.x,
                transformed_point.point.y,
                transformed_point.point.z
            )

        except Exception as e:
            self.get_logger().debug(f"TF2 transform failed: {str(e)[:50]}")
            return None

    def _estimate_person_radius(self, obj: TrackedObject) -> float:
        """Estimate person radius from bounding box"""
        bbox_w = obj.bbox[2] - obj.bbox[0]
        bbox_h = obj.bbox[3] - obj.bbox[1]
        radius = (bbox_w + bbox_h) / (4.0 * self._cam_fx) * obj.center_3d_global[2]
        return max(0.2, min(1.0, radius))

    def _publish_tracking(self, tracked_objects: list, timestamp):
        """Publish tracking data"""
        if not tracked_objects:
            return

        main_person = min(tracked_objects, key=lambda obj: obj.center_3d_global[2])

        # Publish main person
        human_msg = HumanState()
        human_msg.header.stamp = timestamp
        human_msg.header.frame_id = self._global_frame
        human_msg.id = int(main_person.track_id)
        human_msg.px = float(main_person.center_3d_global[0])
        human_msg.py = float(main_person.center_3d_global[1])
        human_msg.vx = float(main_person.velocity_global[0])
        human_msg.vy = float(main_person.velocity_global[1])
        human_msg.radius = self._estimate_person_radius(main_person)
        self._pub_main_person.publish(human_msg)

        # Publish obstacles
        obstacles_msg = Obstacles()
        obstacles_msg.header.stamp = timestamp
        obstacles_msg.header.frame_id = self._global_frame

        for obj in tracked_objects:
            if obj.track_id == main_person.track_id:
                continue

            circle = CircleObstacle()
            circle.uid = int(obj.track_id)
            circle.center = Point(
                x=float(obj.center_3d_global[0]),
                y=float(obj.center_3d_global[1]),
                z=0.0
            )
            circle.velocity = Vector3(
                x=float(obj.velocity_global[0]),
                y=float(obj.velocity_global[1]),
                z=0.0
            )
            circle.radius = self._estimate_person_radius(obj)
            circle.true_radius = circle.radius
            circle.confidence = obj.confidence
            obstacles_msg.circles.append(circle)

        self._pub_obstacles.publish(obstacles_msg)

    def _publish_lost_goal(self, position_global, timestamp):
        """Publish navigation goal when person is lost"""
        goal_msg = PoseStamped()
        goal_msg.header.stamp = timestamp
        goal_msg.header.frame_id = self._global_frame
        goal_msg.pose.position.x = float(position_global[0])
        goal_msg.pose.position.y = float(position_global[1])
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        self._pub_lost_goal.publish(goal_msg)

    def _visualize(self, frame, tracked_objects):
        """Visualization"""
        viz_frame = frame.copy()

        # Draw tracking state
        state_color = {
            "IDLE": (128, 128, 128),
            "TRACKING": (0, 255, 0),
            "LOST": (0, 0, 255),
            "RE_TRACKING": (255, 255, 0),
            "STOPPED": (255, 0, 0),
        }
        cv2.putText(viz_frame, f"State: {self._tracking_state}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                   state_color.get(self._tracking_state, (255, 255, 255)), 2)

        if not tracked_objects:
            cv2.imshow('ZED2 Detection (SIM)', viz_frame)
            cv2.waitKey(1)
            return

        main_person = min(tracked_objects, key=lambda obj: obj.center_3d_global[2])

        for obj in tracked_objects:
            x1, y1, x2, y2 = [int(v) for v in obj.bbox]
            color = (0, 255, 0) if obj.track_id == main_person.track_id else (0, 180, 180)
            cv2.rectangle(viz_frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(viz_frame, f'ID:{obj.track_id}', (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            pos_text = f"({obj.center_3d_global[0]:.1f}, {obj.center_3d_global[1]:.1f})"
            cv2.putText(viz_frame, pos_text, (x1, y2+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            vel_mag = math.sqrt(obj.velocity_global[0]**2 + obj.velocity_global[1]**2)
            vel_text = f"v:{vel_mag:.2f}m/s"
            cv2.putText(viz_frame, vel_text, (x1, y2+35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.imshow('ZED2 Detection (SIM)', viz_frame)
        cv2.waitKey(1)

    def destroy_node(self):
        """Cleanup"""
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HandSignDetectorSimNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
