"""
ZED 2 Hand Sign Detection and Person Tracking Node
Enhanced with BoT-SORT tracker + optimized NMS + appearance features
"""

import os
os.environ['ZED_LOG_LEVEL'] = 'ERROR'

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import cv2
import pyzed.sl as sl
from ultralytics import YOLO
from cv_bridge import CvBridge

import tf2_ros
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
from std_msgs.msg import Header
from amr_interfaces.msg import HumanState, ObstacleState, ObstacleStateArray
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
    velocity_global: Tuple[float, float, float]  # velocity in global frame (vx, vy, vz)
    confidence: float


class HandSignDetectorNode(Node):
    """
    ROS2 Node for hand sign detection and person tracking using ZED 2
    
    Enhanced Features:
    - BoT-SORT tracker: Better than ByteTrack with appearance features
    - Optimized NMS: Proper deduplication of overlapping detections
    - Re-ID appearance features: Robust tracking in crowded scenes
    - Dual model detection: YOLO for persons + YOLO for hand signs
    
    Frame transformation strategy:
    - URDF defines static transforms (camera frame hierarchy)
    - robot_state_publisher broadcasts TF transforms from URDF
    - This node uses TF2 to transform points from camera -> global frame
    - No computation of transforms in code - rely on URDF definition
    
    Topics Published:
    - /tracking/main_person (HumanState) - Primary tracked person
    - /tracking/obstacles (ObstacleStateArray) - Other detected objects
    """

    def __init__(self):
        super().__init__('handsign_detector')
        
        # Declare parameters
        self.declare_parameter('hand_model_path', 'handsign.pt')
        self.declare_parameter('person_model_path', 'yolov8n.pt')
        self.declare_parameter('detection_conf', 0.5)
        self.declare_parameter('nms_threshold', 0.45)
        self.declare_parameter('person_radius', 0.3)
        self.declare_parameter('enable_visualization', False)
        self.declare_parameter('camera_frame', 'zed2_left_camera')
        self.declare_parameter('global_frame', 'odom')
        
        # Get parameters
        hand_model = self.get_parameter('hand_model_path').value
        person_model = self.get_parameter('person_model_path').value
        self._detect_conf = self.get_parameter('detection_conf').value
        self._nms_threshold = self.get_parameter('nms_threshold').value
        self._person_radius = self.get_parameter('person_radius').value
        self._enable_viz = self.get_parameter('enable_visualization').value
        self._camera_frame = self.get_parameter('camera_frame').value
        self._global_frame = self.get_parameter('global_frame').value
        
        # TF2 - for frame transformation
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        
        # Initialize ZED camera
        self.get_logger().info('Initializing ZED 2 camera...')
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.sdk_verbose = False
        
        self._zed = sl.Camera()
        status = self._zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f'Failed to open ZED camera: {status}')
        
        # Camera calibration
        cal_params = self._zed.get_camera_information().camera_configuration.calibration_parameters
        self._cam_fx = cal_params.focal_length.x
        self._cam_fy = cal_params.focal_length.y
        self._cam_cx = cal_params.principal_point.x
        self._cam_cy = cal_params.principal_point.y
        
        # Load models
        self.get_logger().info(f'Loading YOLO models...')
        self._hand_model = YOLO(self._resolve_model_path(hand_model))
        self._person_model = YOLO(self._resolve_model_path(person_model))
        
        # Initialize BoT-SORT tracker for PERSON TRACKING ONLY
        # Hand sign detection is ONE-TIME EVENT, not continuous tracking
        self.get_logger().info('Initializing BoT-SORT tracker for person tracking...')
        self._person_tracker = BoTSortTracker(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
            use_appearance=True,
            appearance_weight=0.5
        )
        
        # Hand sign detection with 2 states: START / STOP, 3-second hold to trigger
        # State machine: IDLE -> START (hold 3s) -> TRACKING
        #              TRACKING -> STOP (hold 3s) -> STOPPED
        #              TRACKING -> (person lost) -> LOST (publish goal, wait for re-entry)
        #              LOST -> (person re-detected) -> RE_TRACKING
        self._hand_sign_frame_skip = 10  # Detect every 10 frames (~3 Hz)
        self._frame_counter = 0
        
        # Hand sign gesture state machine (2 classes: START=0, STOP=1)
        self._gesture_state = "IDLE"  # IDLE | START_HELD | STOP_HELD
        self._gesture_timer_start = None
        self._gesture_hold_threshold = 3.0  # 3 seconds to confirm gesture
        
        # Tracking state machine
        self._tracking_state = "IDLE"  # IDLE | TRACKING | LOST | RE_TRACKING | STOPPED
        self._frames_without_detection = 0
        self._lost_threshold = 30
        
        # CTRV EKF-based trajectory prediction (curvilinear motion model)
        # Superior to linear prediction: handles turning/curved motion
        self._ctrv_tracker = MultiObjectCTRVTracker(dt=1/30.0)
        self._prediction_horizon = 5.0  # Predict 5 seconds ahead
        
        # Last known state for loss recovery (used as fallback)
        self._last_known_position_global = None
        self._last_known_velocity = None
        
        # Feature extractor (lightweight, CPU-friendly)
        self._feature_extractor = FeatureExtractorLightweight(n_bins=16)
        
        # NMS optimizer (Deep-Aware NMS with spatial decay)
        self._nms = DeepNMSOptimizer()
        
        # Simple velocity filters (for initial state estimation only)
        self._velocity_filters: Dict[int, VelocityFilter] = {}
        self._prev_positions: Dict[int, np.ndarray] = {}
        self._last_timestamps: Dict[int, float] = {}
        
        # Publishers
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self._pub_main_person = self.create_publisher(HumanState, '/tracking/main_person', qos)
        self._pub_obstacles = self.create_publisher(ObstacleStateArray, '/tracking/obstacles', qos)
        
        # Publisher for navigation goal (when person is lost, robot should go to last known position)
        self._pub_lost_goal = self.create_publisher(PoseStamped, '/navigation/goal', qos)
        
        # OpenCV bridge
        self._bridge = CvBridge()
        
        # Processing loop
        self._timer = self.create_timer(1.0/30.0, self._process_frame)
        self.get_logger().info('HandSign Detector initialized - BoT-SORT + Optimized NMS ready')
    
    def _resolve_model_path(self, model_path: str) -> str:
        """Resolve model path - check absolute, relative, and models/ directory"""
        # Try absolute path
        if os.path.isabs(model_path) and os.path.exists(model_path):
            return model_path
        
        # Try relative path
        if os.path.exists(model_path):
            return model_path
        
        # Try models/ directory relative to package
        import rclpy.logging
        try:
            from ament_index_python.packages import get_package_share_directory
            pkg_dir = get_package_share_directory('amr_zed2')
            models_path = os.path.join(pkg_dir, 'models', model_path)
            if os.path.exists(models_path):
                return models_path
        except:
            pass
        
        raise FileNotFoundError(f'Model not found: {model_path} (checked: absolute, relative, models/)')
    
    def _process_frame(self):
        """Main processing loop: detect hand signs + persons, track, estimate velocity in global frame"""
        if self._zed.grab() != sl.ERROR_CODE.SUCCESS:
            return
        
        # Get image and depth
        image = sl.Mat()
        depth = sl.Mat()
        self._zed.retrieve_image(image, sl.VIEW.LEFT)
        self._zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        
        frame = cv2.cvtColor(image.get_data(), cv2.COLOR_RGBA2BGR)
        depth_data = depth.get_data()
        
        # Get timestamp
        timestamp = self.get_clock().now().to_msg()
        current_time = timestamp.sec + timestamp.nanosec / 1e9
        
        # Person Detection
        person_results = self._person_model.predict(frame, conf=self._detect_conf, verbose=False)[0]
        
        # Extract person detections (class 0 = Person)
        person_detections = []
        for det in person_results.boxes.data:
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            if int(cls) == 0:  # Person class
                person_detections.append([x1, y1, x2, y2, conf])
        
        if person_detections:
            person_detections = np.array(person_detections)
        else:
            person_detections = np.empty((0, 5))
        
        # APPLY DEEP-AWARE NMS to person detections (Spatial decay model)
        if len(person_detections) > 0:
            person_detections = self._nms.deep_nms_spatial(person_detections, self._nms_threshold)
            self.get_logger().debug(f'Persons after NMS: {len(person_detections)}')
        
        # HAND SIGN DETECTION: 2 STATES (START/STOP) with 3-SECOND HOLD TRIGGER
        # Classes: 0 = START, 1 = STOP
        self._frame_counter += 1
        detected_gesture_class = None
        
        if self._frame_counter % self._hand_sign_frame_skip == 0:
            hand_results = self._hand_model.predict(frame, conf=self._detect_conf, verbose=False)[0]
            if len(hand_results.boxes) > 0:
                best_det = hand_results.boxes.data[hand_results.boxes.conf.argmax()]
                detected_gesture_class = int(best_det[5].cpu().numpy())  # Class
        
        # State machine: START/STOP hold for 3 seconds
        elapsed = current_time - self._gesture_timer_start if self._gesture_timer_start else 0
        
        if detected_gesture_class == 0:  # START
            if self._gesture_state != "START_HELD":
                self._gesture_state = "START_HELD"
                self._gesture_timer_start = current_time
                self.get_logger().info('👋 START gesture detected')
            elif elapsed >= self._gesture_hold_threshold and self._tracking_state == "IDLE":
                self._tracking_state = "TRACKING"
                self._gesture_state = "IDLE"
                self._gesture_timer_start = None
                # Reset velocity filters and CTRV tracker on new tracking start
                self._velocity_filters.clear()
                self._prev_positions.clear()
                self._last_timestamps.clear()
                self._ctrv_tracker = MultiObjectCTRVTracker(dt=1/30.0)
                self.get_logger().warn('▶️ TRACKING STARTED')
                    
        elif detected_gesture_class == 1:  # STOP
            if self._gesture_state != "STOP_HELD":
                self._gesture_state = "STOP_HELD"
                self._gesture_timer_start = current_time
                self.get_logger().info('🛑 STOP gesture detected')
            elif elapsed >= self._gesture_hold_threshold and self._tracking_state == "TRACKING":
                self._tracking_state = "STOPPED"
                self._gesture_state = "IDLE"
                self._gesture_timer_start = None
                self._ctrv_tracker = MultiObjectCTRVTracker(dt=1/30.0)  # Reset CTRV filters on stop
                self.get_logger().warn('⏹️ TRACKING STOPPED')
        else:
            # No gesture - reset
            if self._gesture_state != "IDLE":
                self._gesture_state = "IDLE"
                self._gesture_timer_start = None
        
        # PERSON DETECTION STATE MACHINE: Loss & recovery
        if len(person_detections) > 0 and self._tracking_state in ["TRACKING", "RE_TRACKING"]:
            self._frames_without_detection = 0
            if self._tracking_state == "RE_TRACKING":
                self._tracking_state = "TRACKING"
                self.get_logger().warn('✅ Person re-detected - resuming tracking')
                
        elif self._tracking_state in ["TRACKING", "RE_TRACKING"]:
            self._frames_without_detection += 1
            
            if self._frames_without_detection == 1 and self._tracking_state == "TRACKING":
                self._tracking_state = "LOST"
                self.get_logger().warn('❌ Person LOST - fetching CTRV trajectory prediction')
            
            # Publish navigation goal when confirmed lost (after ~1 second)
            if self._frames_without_detection >= self._lost_threshold and self._tracking_state == "LOST":
                # Try CTRV EKF prediction first (handles curved motion)
                if len(tracked_objects) > 0:
                    main_person = tracked_objects[0]
                    goal_pos_ctrv = self._ctrv_tracker.predict(int(main_person.track_id), self._prediction_horizon)
                    if goal_pos_ctrv is not None:
                        goal_pos_3d = np.array([goal_pos_ctrv[0], goal_pos_ctrv[1], 0.0])
                        self._publish_lost_goal(goal_pos_3d, timestamp)
                        ctrv_state = self._ctrv_tracker.get_state(int(main_person.track_id))
                        self.get_logger().warn(
                            f'⚠️ CTRV Prediction [{self._prediction_horizon}s] >> '
                            f'({goal_pos_ctrv[0]:.2f}, {goal_pos_ctrv[1]:.2f}) '
                            f'[v={ctrv_state.v:.2f}, ω={ctrv_state.omega:.3f}]'
                        )
                    else:
                        # Fallback to linear if CTRV fails
                        if self._last_known_velocity is not None:
                            fallback_pos = self._last_known_position_global + self._last_known_velocity * self._prediction_horizon
                            self._publish_lost_goal(fallback_pos, timestamp)
                            self.get_logger().warn(f'⚠️ Using linear fallback >> ({fallback_pos[0]:.2f}, {fallback_pos[1]:.2f})')
                elif self._last_known_position_global is not None:
                    # Final fallback: last known position only
                    self._publish_lost_goal(self._last_known_position_global.copy(), timestamp)
                    self.get_logger().warn(f'⚠️ No trajectory - using last position')
        
        # EXTRACT APPEARANCE FEATURES for BoT-SORT Re-ID (PERSON TRACKING ONLY)
        if len(person_detections) > 0:
            person_features = self._feature_extractor.extract(frame, person_detections[:, :4])
        else:
            person_features = None
        
        # UPDATE BoT-SORT TRACKER: Only active when TRACKING or RE_TRACKING
        # When IDLE/STOPPED/LOST: pause tracking, allow clean restart
        if self._tracking_state in ["TRACKING", "RE_TRACKING"]:
            tracks = self._person_tracker.update(person_detections, features=person_features)
        else:
            # Tracking paused - keep tracker internal state but don't add new detections
            tracks = []
        
        # Estimate 3D positions and velocities in global frame
        tracked_objects = []
        
        for track in tracks:
            x1, y1, x2, y2, tid = track
            person_center_x = (x1 + x2) / 2
            person_center_y = (y1 + y2) / 2
            person_bbox = (x1, y1, x2, y2)
            
            # Get depth at person center
            depth_val = depth_data[int(person_center_y), int(person_center_x), 0]
            if depth_val <= 0 or depth_val > 10:  # Invalid or too far
                continue
            
            # Backproject to 3D (camera frame)
            x3d_cam = (person_center_x - self._cam_cx) * depth_val / self._cam_fx
            y3d_cam = (person_center_y - self._cam_cy) * depth_val / self._cam_fy
            z3d_cam = depth_val
            
            # Transform to global frame
            x3d_global, y3d_global, z3d_global = self._transform_to_global(
                x3d_cam, y3d_cam, z3d_cam, timestamp
            )
            
            if x3d_global is None:  # TF transform failed
                continue
            
            pos_global = np.array([x3d_global, y3d_global, z3d_global])
            
            # Velocity estimation
            if tid not in self._velocity_filters:
                self._velocity_filters[tid] = VelocityFilter(alpha=0.3)
                self._prev_positions[tid] = pos_global.copy()
                self._last_timestamps[tid] = current_time
            
            dt = current_time - self._last_timestamps[tid]
            if dt > 0:
                # Velocity in global frame
                vel_raw = (pos_global - self._prev_positions[tid]) / dt
                vel_filtered = self._velocity_filters[tid].update(vel_raw)
            else:
                vel_filtered = self._velocity_filters[tid].filtered_velocity or np.zeros(3)
            
            self._prev_positions[tid] = pos_global.copy()
            self._last_timestamps[tid] = current_time
            
            # UPDATE CTRV EKF TRACKER with position and velocity
            # This builds curvilinear motion model for better trajectory prediction
            if self._tracking_state in ["TRACKING", "RE_TRACKING"]:
                pos_2d = np.array([pos_global[0], pos_global[1]])
                vel_2d = np.array([vel_filtered[0], vel_filtered[1]])
                self._ctrv_tracker.update(int(tid), pos_2d, vel_2d)
                
                # Save for fallback (if CTRV prediction fails)
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
        
        # Publish tracking data in controller format
        self._publish_tracking(tracked_objects, timestamp)
        
        # Visualization (optional)
        if self._enable_viz:
            self._visualize(frame, tracked_objects)
    
    def _transform_to_global(self, x_cam, y_cam, z_cam, timestamp):
        """Transform 3D point from camera frame to global frame using TF2
        
        Transformation is handled by robot_state_publisher from URDF.
        This method simply looks up the transform and applies it.
        """
        try:
            # Get transform from camera frame to global frame (published by robot_state_publisher)
            transform = self._tf_buffer.lookup_transform(
                self._global_frame,
                self._camera_frame,
                timestamp
            )
            
            # Use tf2_geometry_msgs to transform point
            from geometry_msgs.msg import PointStamped
            from tf2_geometry_msgs import do_transform_point
            
            point_stamped = PointStamped()
            point_stamped.header.frame_id = self._camera_frame
            point_stamped.header.stamp = timestamp
            point_stamped.point.x = float(x_cam)
            point_stamped.point.y = float(y_cam)
            point_stamped.point.z = float(z_cam)
            
            # Transform point using TF2
            transformed_point = do_transform_point(point_stamped, transform)
            
            return (
                transformed_point.point.x,
                transformed_point.point.y,
                transformed_point.point.z
            )
        
        except Exception as e:
            # TF2 transform error - log but continue (graceful degradation)
            self.get_logger().warn(f"TF2 transform failed: {str(e)[:100]}")
            return None
    
    def _estimate_person_radius(self, obj: TrackedObject) -> float:
        """Estimate person radius from bounding box size"""
        bbox_w = obj.bbox[2] - obj.bbox[0]
        bbox_h = obj.bbox[3] - obj.bbox[1]
        radius = (bbox_w + bbox_h) / (4.0 * self._cam_fx) * obj.center_3d_global[2]
        return max(0.2, min(1.0, radius))  # Clamp to [0.2, 1.0] meters

    def _publish_tracking(self, traced_objects: list, timestamp):
        """Publish tracking data in controller format (global frame, correct fields)
        
        Note: Tracking is only published when hand sign trigger indicates tracking is active
        """
        if not traced_objects:
            return
        
        # Find main person (closest in Z direction in global frame)
        main_person = min(traced_objects, key=lambda obj: obj.center_3d_global[2])
        
        # Publish main person as HumanState
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
        
        # Publish other persons as obstacles
        obstacle_array = ObstacleStateArray()
        obstacle_array.header.stamp = timestamp
        obstacle_array.header.frame_id = self._global_frame
        
        for obj in traced_objects:
            if obj.track_id == main_person.track_id:
                continue
            
            obstacle = ObstacleState()
            obstacle.id = int(obj.track_id)
            obstacle.px = float(obj.center_3d_global[0])
            obstacle.py = float(obj.center_3d_global[1])
            obstacle.radius = self._estimate_person_radius(obj)
            obstacle_array.obstacles.append(obstacle)
        
        self._pub_obstacles.publish(obstacle_array)
    
    def _publish_lost_goal(self, position_global, timestamp):
        """Publish navigation goal when person is lost (for robot to move to last known position)
        
        This allows the robot to autonomously navigate to the last known position of the person
        and resume tracking when the person reappears.
        """
        goal_msg = PoseStamped()
        goal_msg.header.stamp = timestamp
        goal_msg.header.frame_id = self._global_frame
        
        # Position: last known (or extrapolated) person position
        goal_msg.pose.position.x = float(position_global[0])
        goal_msg.pose.position.y = float(position_global[1])
        goal_msg.pose.position.z = 0.0  # 2D navigation
        
        # Orientation: default (facing forward)
        goal_msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        
        self._pub_lost_goal.publish(goal_msg)
    
    def _visualize(self, frame, tracked_objects):
        """Visualization of tracked objects"""
        viz_frame = frame.copy()
        
        if not tracked_objects:
            cv2.imshow('ZED2 Detection', viz_frame)
            cv2.waitKey(1)
            return
        
        # Find main person for highlighting
        main_person = min(tracked_objects, key=lambda obj: obj.center_3d_global[2])
        
        # Draw person bboxes
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj.bbox
            # Highlight main person in green, others in yellow
            color = (0, 255, 0) if obj.track_id == main_person.track_id else (0, 180, 180)
            cv2.rectangle(viz_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw track ID
            cv2.putText(viz_frame, f'ID:{obj.track_id}', (int(x1), int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw position info (2D projection)
            pos_text = f"({obj.center_3d_global[0]:.1f}, {obj.center_3d_global[1]:.1f})"
            cv2.putText(viz_frame, pos_text, (int(x1), int(y2)+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw velocity magnitude
            vel_mag = math.sqrt(obj.velocity_global[0]**2 + obj.velocity_global[1]**2)
            vel_text = f"v:{vel_mag:.2f}m/s"
            cv2.putText(viz_frame, vel_text, (int(x1), int(y2)+35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.imshow('ZED2 Detection', viz_frame)
        cv2.waitKey(1)
    
    def destroy_node(self):
        """Cleanup"""
        self._zed.close()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HandSignDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
