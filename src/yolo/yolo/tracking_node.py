#!/usr/bin/env python3
"""
ROS2 Hand Sign Detection Node with BoT-SORT Tracking
OPTIMIZED FOR ROS2 TOPIC-BASED STREAMING

Core optimizations:
  ✅ Frame timing & latency measurement
  ✅ QoS tuned for low-latency ROS2 streaming
  ✅ Frame skipping on CPU overload
  ✅ Memory-efficient numpy buffer reuse
  ✅ Dynamic parameters (tune at runtime)
  ✅ Robust frame drop recovery
  ✅ Performance metrics (FPS, latency tracking)
  ✅ Message filter sync for multi-topic processing
"""

import rclpy
import os
import cv2
import numpy as np
import time
import yaml
import math
from collections import deque
from typing import Dict, List, Tuple, Optional
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ultralytics import YOLO
from cv_bridge import CvBridge, CvBridgeError
import message_filters  # RGB-Depth synchronization for 3D accuracy

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Header
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from interfaces.msg import HumanState, ObstacleArray, DynaObstacle, SystemState
from ament_index_python.packages import get_package_share_directory

# BoT-SORT tracking with appearance features
from yolo.algorithms.botsort_handler import BoTSortTracker
from yolo.algorithms.appearance_feature_extractor import FeatureExtractorDeep
from yolo.algorithms.ctrv_ekf import CTRV_EKF
from yolo.algorithms.velocity_filter import VelocityFilter

# Depth-aware NMS for improved hand detection
try:
    from yolo.algorithms.depth_aware_nms import DepthAwareNMS
    HAS_DEPTH_NMS = True
except ImportError:
    HAS_DEPTH_NMS = False


class TrackingNode(Node):
    """Hand sign detection with BoT-SORT + ROS2 (optimized for streaming)"""

    def __init__(self):
        super().__init__('tracking_node')

        # ===== 1. PARAMETERS (DYNAMIC + YAML CONFIG) =====
        # --- Runtime parameters ---
        self.declare_parameter('camera_image_topic', '/zed/zed_node/rgb/color/rect/image')
        self.declare_parameter('person_model_path', 'yolov8n.pt')
        self.declare_parameter('hand_model_path', 'handsign.pt')
        self.declare_parameter('use_cuda', True)
        self.declare_parameter('use_fp16', True)
        self.declare_parameter('hand_person_match_buffer', 100)
        self.declare_parameter('max_inference_time_ms', 200.0)
        self.declare_parameter('enable_frame_skip', True)
        self.declare_parameter('enable_performance_metrics', True)

        # Load YAML config for tunable parameters
        try:
            config_path = os.path.join(get_package_share_directory('yolo'), 'config', 'params.yaml')
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # YOLO detection parameters
            self.person_conf = config['yolo']['person_conf_thresh']
            self.hand_conf = config['yolo']['handsign_conf_thresh']
            
            # Tracking parameters
            self.gesture_hold_time = config['tracking']['gesture_hold_time']
            self.tracking_timeout = config['tracking']['tracking_timeout']
            self.gesture_buffer_size = config['tracking']['gesture_buffer_size']
            self.gesture_confirmation_ratio = config['tracking']['gesture_confirmation_ratio']
            
            # ROI optimization
            self.roi_margin_x_ratio = config['tracking']['roi_margin_x_ratio']
            self.roi_margin_y_top_ratio = config['tracking']['roi_margin_y_top_ratio']
            self.roi_body_height_ratio = config['tracking']['roi_body_height_ratio']
            
            # Depth filtering
            self.max_hand_person_depth_diff = config['depth_filter']['max_hand_person_depth_diff']
            
            # Histogram equalization
            self.enable_histogram_equalization = config['image_processing']['enable_histogram_equalization']
            self.clahe_clip_limit = config['image_processing']['clahe_clip_limit']
            self.clahe_tile_size = config['image_processing']['clahe_tile_size']
            
            # ===== CAMERA INTRINSICS (WVGA - 672x376) =====
            # Load directly from config dict (no ROS parameter declaration needed)
            self.cam_w = config['camera']['width']
            self.cam_h = config['camera']['height']
            self.fx = config['camera']['fx']
            self.fy = config['camera']['fy']
            self.cx = config['camera']['cx']
            self.cy = config['camera']['cy']
            
            self.get_logger().info(f'✅ Loaded configuration from {config_path}')
            self.get_logger().info(
                f'✅ Loaded ZED2 Camera Config: {self.cam_w}x{self.cam_h}, '
                f'fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}'
            )
        except Exception as e:
            self.get_logger().warn(f'⚠️  Failed to load params.yaml: {e}. Using defaults.')

        camera_topic = self.get_parameter('camera_image_topic').value
        person_model_name = self.get_parameter('person_model_path').value
        hand_model_name = self.get_parameter('hand_model_path').value
        use_cuda = self.get_parameter('use_cuda').value
        self.use_fp16 = self.get_parameter('use_fp16').value
        self.hand_person_buffer = self.get_parameter('hand_person_match_buffer').value
        self.gesture_y_min_ratio = 0.10
        self.gesture_y_max_ratio = 0.90
        self.max_inference_time = self.get_parameter('max_inference_time_ms').value / 1000.0
        self.enable_frame_skip = self.get_parameter('enable_frame_skip').value
        self.enable_perf_metrics = self.get_parameter('enable_performance_metrics').value

        self._device = 'cuda' if use_cuda else 'cpu'
        
        # Initialize CLAHE for histogram equalization (if enabled)
        if self.enable_histogram_equalization:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=(self.clahe_tile_size, self.clahe_tile_size)
            )
            self.get_logger().info(f'✅ CLAHE histogram equalization enabled (clip={self.clahe_clip_limit}, tile={self.clahe_tile_size})')
        else:
            self._clahe = None

        # ===== 2. LOAD YOLO MODELS =====
        try:
            pkg_share_dir = get_package_share_directory('yolo')
            person_model_path = os.path.join(pkg_share_dir, 'models', person_model_name)
            hand_model_path = os.path.join(pkg_share_dir, 'models', hand_model_name)

            if not os.path.isfile(person_model_path):
                raise FileNotFoundError(f'Person model not found: {person_model_path}')
            if not os.path.isfile(hand_model_path):
                raise FileNotFoundError(f'Hand model not found: {hand_model_path}')
        except Exception as e:
            self.get_logger().warn(f'⚠️  Package models not found: {e}, using pretrained')
            person_model_path = person_model_name
            hand_model_path = hand_model_name

        self.get_logger().info(f'🚀 Loading models from {person_model_path}, {hand_model_path}')

        try:
            self._person_yolo = YOLO(person_model_path)
            if use_cuda:
                self._person_yolo.to('cuda')
            self.get_logger().info('✅ Person model loaded')
        except Exception as e:
            self.get_logger().error(f'❌ Failed to load person model: {e}')
            raise

        try:
            self._hand_yolo = YOLO(hand_model_path)
            if use_cuda:
                self._hand_yolo.to('cuda')
            self.get_logger().info('✅ Hand model loaded')
        except Exception as e:
            self.get_logger().error(f'❌ Failed to load hand model: {e}')
            raise

        # ===== 2b. WARM-UP MODELS (Prevent 10s Cold Start) =====
        # CUDA compilation and VRAM allocation happens FIRST inference run
        # Without warm-up, first person detection causes 10s freeze
        self.get_logger().info('🔥 Warming up GPU (preventing 10s cold start on first detection)...')
        try:
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            self._person_yolo.predict(dummy_img, device=self._device, verbose=False, half=self.use_fp16)
            self._hand_yolo.predict(dummy_img, device=self._device, verbose=False, half=self.use_fp16)
            self.get_logger().info('✅ GPU warm-up complete (cold start eliminated)')
        except Exception as e:
            self.get_logger().warn(f'⚠️  Warm-up failed (non-critical): {e}')

        # ===== 3. INITIALIZE TRACKERS =====
        # CRITICAL FOR FLICKERING FIX: Lower track_thresh for patience + higher max_age_before_predict
        # This ensures person bbox stays visible even when YOLO temporarily misses detection
        # Parameters loaded from config/params.yaml

        self._person_tracker = BoTSortTracker(
            track_thresh=0.45,          # High threshold: require 45% confidence for new tracks
            track_low_thresh=0.10,      # LOW THRESHOLD: keep tracks alive with only 10% confidence (BYTE Association)
            track_buffer=500,           # 500 frames ≈ 5s at 100 FPS
            match_thresh=0.8,
            use_appearance=True,        # ENABLED: Deep learning Re-ID features
            appearance_weight=0.6,      # Trust ResNet18 features 60% (improved from 0.2)
            max_age_before_predict=150
        )
        
        self.get_logger().info(
            f'✅ Person Tracker initialized (FLICKERING FIX v3 - BYTE Association):'
            f' track_thresh=0.45, track_low_thresh=0.10, track_buffer=500 frames (~5s@100FPS), '
            f'max_age_before_predict=150 (Two-Stage matching + EMA smoothing)'
        )

        self._hand_tracker = BoTSortTracker(
            track_thresh=0.35,
            track_low_thresh=0.05,      # Lower threshold for hands (weaker detections)
            track_buffer=100,
            match_thresh=0.7,
            use_appearance=True,
            appearance_weight=0.25,
            max_age_before_predict=20
        )

        # ===== 4. APPEARANCE FEATURES (DEEP LEARNING ON RTX A4000) =====
        # ResNet18 backbone extracts 512-dim features from person crops
        # RTX A4000 Tensor Cores (FP16) accelerate extraction to ~1.5ms/person
        # Enables robust Re-ID after occlusion/re-entrance
        self._appearance_extractor = FeatureExtractorDeep(
            device=self._device,
            half=self.use_fp16  # Tensor Core FP16 acceleration
        )

        # ===== 5. GESTURE STATE MACHINE (ROBUST - GLOBAL TIMERS + CONSECUTIVE FRAMES + HYSTERESIS BUFFER) =====
        # NO PER-TRACK TIMING: Use global timers instead (like ref_yolo does)
        # This avoids BoTSort tracking jitter from resetting gesture detection
        # OPTIMIZATION: Add consecutive frame counter to eliminate noise (per Gemini)
        # NEW: Add gesture buffer (deque voting) for hysteresis - eliminate flickering
        
        self.gesture_buffer = deque(maxlen=self.gesture_buffer_size)  # Keep last N gesture detections
        self.confirmed_gesture = "NONE"  # Current confirmed gesture state
        
        self.start_signal_detected = False
        self.start_signal_time = None
        self.last_start_seen_time = None
        self.start_consecutive_count = 0
        
        self.stop_detected_time = None
        self.last_stop_seen_time = None
        self.stop_consecutive_count = 0
        self.signal_timeout = 1.0  # Allow 1s signal loss without timer reset

        # ===== 5b. SYSTEM STATE (from state_machine_node) =====
        # Manage detection logic based on robot state
        self.current_robot_state = 'IDLE'  # IDLE | TRACKING | RE_TRACKING

        # ===== 5c. TARGET PERSON LOCKING (START/STOP control) =====
        # When START gesture is detected, robot locks onto this person ID
        self.main_target_id: Optional[int] = None
        self.target_lock_time: float = 0.0  # Timestamp when target was locked (for START 1s detection)
        self.target_lost_time: Optional[float] = None  # Timestamp when target disappeared (for 5s Re-TRACKING)
        
        # ===== 5c. STATE MACHINE =====
        # IDLE: No target locked
        # TRACKING: Target locked and visible
        # Re-TRACKING: Target disappeared, predicting position for 5s
        self.current_state = 'IDLE'
        self.target_visible = False  # Whether main target is in current frame
        self.state_changed_at = time.time()  # For state transition logging
        
        # ===== 5d. MOTION FILTERING & ESTIMATION =====
        # CTRV Extended Kalman Filters: one per tracked person
        self.ekfs: Dict[int, CTRV_EKF] = {}
        # Velocity filters: one per tracked person (Exponential Moving Average)
        self.velocity_filters: Dict[int, VelocityFilter] = {}
        # Store position history for velocity calculation
        self.position_history: Dict[int, deque] = {}
        self.max_position_history = 10  # Keep last 10 positions for velocity estimation

        # ===== 5e. TIMESTAMP TRACKING FOR REAL DT =====
        # Track frame timestamps to compute real dt instead of assuming 1/30.0
        self.last_frame_timestamp_per_track: Dict[int, float] = {}  # {track_id: ts_seconds}

        # ===== 6. FRAME STORAGE =====
        self._current_persons = []
        self._current_gestures = []
        self._current_depth_map = None  # Store latest depth map
        
        # ===== 6a. CAMERA INTRINSICS =====
        # fx, cx, cy will be obtained from CameraInfo topic
        # Default values for ZED2 WVGA (672x376) - actual values will be overwritten
        self.fx = 263.127      # Focal length X for WVGA
        self.cy_cam = 187.776  # Principle point Y for WVGA
        self.cx_cam = 346.277  # Principle point X for WVGA
        self.camera_info_received = False  # Flag to track if we got camera info yet
        
        # ===== 6b. EMA SMOOTHING FOR BOUNDING BOXES (REMOVED FOR PERFORMANCE) =====
        # Removed: EMA smoothing reduces responsiveness without significant benefit
        # Keeping raw detections increases FPS at cost of slight jitter

        # ===== 7. ROS2 QoS PROFILE (OPTIMIZED FOR STREAMING) =====
        # LOW LATENCY: Best effort reliability, minimal history
        qos_img = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # Drop old frames if queue full
            history=HistoryPolicy.KEEP_LAST,
            depth=1  # Only keep 1 frame = minimal latency
        )

        # ===== 8. DEPTH-AWARE NMS (OPTIMIZED FOR START/STOP DETECTION) =====
        if HAS_DEPTH_NMS:
            # TỐI ƯU ĐỂ DỄ NHẬN DIỆN TAY: Nới lỏng NMS (tăng threshold) để không vô tình xóa tay thật
            self._depth_nms = DepthAwareNMS(nms_threshold=0.65, depth_threshold=0.70)
            self.get_logger().info('✅ Depth-Aware NMS enabled (Relaxed mode: nms=0.65, depth=0.70)')
        else:
            self._depth_nms = None
            self.get_logger().warn('⚠️  Depth-Aware NMS not available (detection may be less accurate)')

        # ===== 9. PUBLISHERS =====
        self._pub_gesture = self.create_publisher(String, '/yolo/gesture_command', 10)
        # ===== 9b. HUMAN STATE PUBLISHER (for state machine integration) =====
        self._pub_human_state = self.create_publisher(HumanState, '/tracking/main_person', 10)
        # ===== 9c. OBSTACLES PUBLISHER (for perception layer) =====
        # Updated: Use ObstacleArray for multiple dynamic obstacles
        self._pub_obstacles = self.create_publisher(ObstacleArray, '/tracking/obstacles', 10)
        # ===== 9d. ANNOTATED IMAGE PUBLISHER (for debugging/visualization) =====
        self._pub_annotated_image = self.create_publisher(Image, '/tracking/annotated_image', 10)

        # ===== 10. SUBSCRIBERS (RGB-DEPTH SYNCHRONIZED & SYSTEM STATE) =====
        # Subscribe to system state (IDLE | TRACKING | RE_TRACKING) using SystemState interface
        self.create_subscription(
            SystemState,
            '/system_state',
            self._system_state_callback,
            10
        )

        # Create subscribers without immediate callbacks
        self.image_sub = message_filters.Subscriber(
            self, Image, camera_topic, qos_profile=qos_img
        )
        
        if HAS_DEPTH_NMS:
            # 🔗 SYNCHRONIZER: Wait for both RGB and Depth with max 0.1s tolerance
            # Ensures 3D coordinates are computed from perfectly aligned sensor readings
            self.depth_sub = message_filters.Subscriber(
                self, Image, '/zed/zed_node/depth/depth_registered', qos_profile=qos_img
            )
            
            # ApproximateTimeSynchronizer: Matches messages within 0.1s window (100ms)
            # At 30 FPS, messages arrive every 33ms, so sync is tight and accurate
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.image_sub, self.depth_sub], queue_size=5, slop=0.1
            )
            self.ts.registerCallback(self._synced_callback)
            self.get_logger().info('✅ RGB-Depth synchronization enabled (message_filters, slop=0.1s)')
        else:
            # Fallback: Use image callback only if no depth
            self.image_sub.registerCallback(self._fallback_image_callback)
        
        # Subscribe to camera info separately (no sync needed)
        self.create_subscription(
            CameraInfo,
            '/zed/zed_node/rgb/camera_info',
            self._camera_info_callback,
            qos_profile=qos_img
        )

        # ===== 11. PERFORMANCE METRICS =====
        self._bridge = CvBridge()
        self._current_state = 'IDLE'  # Track current state for logging
        
        # Latency tracking (deque for rolling average)
        self._inference_times = deque(maxlen=30)  # Last 30 frames
        self._frame_times = deque(maxlen=30)
        self._last_frame_time = time.time()
        self._last_perf_log_time = time.time()
        self._frame_dims_logged = False
        self._last_valid_frame = None  # For frame skip recovery

        # ===== 11b. PERFORMANCE LOGGING TIMER =====
        if self.enable_perf_metrics:
            self.create_timer(5.0, self._log_performance_metrics)

        self.get_logger().info(
            f'✅ Tracking Node initialized (ROS2 OPTIMIZED)\n'
            f'   Camera: {camera_topic}\n'
            f'   Resolution: VGA 640x480 @ 30 FPS\n'
            f'   Person Model: {person_model_name} @ {self.person_conf}\n'
            f'   Hand Model: {hand_model_name} @ {self.hand_conf} (sensitive mode)\n'
            f'   Tracker: BoT-SORT with appearance features\n'
            f'   Device: {self._device}\n'
            f'   QoS: Best Effort, depth=1 (low latency)\n'
            f'   Frame Skip: {"ENABLED" if self.enable_frame_skip else "DISABLED (process all frames)"}\n'
            f'   Max Inference: {self.max_inference_time*1000:.0f}ms\n'
            f'   Gesture Hold Time: {self.gesture_hold_time}s (fast response)\n'
            f'   Y-Position Range: {self.gesture_y_min_ratio}-{self.gesture_y_max_ratio} (extended)\n'
            f'   Training Mode: OPTIMIZED (No visualization, state-based logging)\n'
            f'   Performance Metrics: {"ENABLED" if self.enable_perf_metrics else "DISABLED"}'
        )

    def _system_state_callback(self, msg: SystemState):
        """Update robot state from state_machine_node using SystemState interface"""
        self.current_robot_state = msg.state
        self.get_logger().debug(f'📡 System state updated: {self.current_robot_state}')

    def _get_real_dt(self, track_id: int, current_timestamp: float) -> float:
        """Calculate real dt from frame timestamps instead of assuming 1/30.0

        This accounts for actual frame rate variation due to GPU load, camera sync, etc.
        CRITICAL FIX: Always update timestamp IMMEDIATELY to prevent dt accumulation bug

        Args:
            track_id: Track ID to look up previous timestamp
            current_timestamp: Current frame timestamp (seconds)

        Returns:
            dt: Real time delta in seconds, or 1/30.0 if first measurement
        """
        if track_id not in self.last_frame_timestamp_per_track:
            self.last_frame_timestamp_per_track[track_id] = current_timestamp
            return 1.0 / 30.0
        
        dt = current_timestamp - self.last_frame_timestamp_per_track[track_id]
        
        # ⚠️  CRITICAL: MUST UPDATE TIMESTAMP IMMEDIATELY to prevent dt accumulation (KẸTDT bug)
        # This ensures next frame calculation is fresh, not stale
        self.last_frame_timestamp_per_track[track_id] = current_timestamp
        
        # Accept dt up to 2.0 seconds (covers frame drops, multi-frame occlusion)
        if 0.005 < dt < 2.0:
            return dt
        
        # If dt outside range (clock jump, extreme frame skip), use default
        # No warning printed anymore since timestamp is now always fresh
        return 1.0 / 30.0

    def get_3d_coordinates(self, u: float, v: float, z: float) -> Tuple[float, float, float]:
        """Convert 2D pixel coordinates to 3D world coordinates using pinhole camera model.
        
        This uses the camera intrinsics (fx, fy, cx, cy) to transform from image space
        to 3D camera-relative coordinates. Essential for MPC control loop that needs
        world coordinates of detected person.
        
        Args:
            u: Pixel X coordinate (0 to 672)
            v: Pixel Y coordinate (0 to 376)
            z: Depth value from ZED2 depth map (meters)
        
        Returns:
            (X, Y, Z): 3D coordinates in camera frame (meters)
                X: Horizontal offset from camera principal axis
                Y: Vertical offset from camera principal axis  
                Z: Depth along camera Z axis
        """
        X = (u - self.cx) * z / self.fx
        Y = (v - self.cy) * z / self.fy
        return X, Y, z

    def _get_weighted_depth_in_bbox(self, bbox: Tuple[int, int, int, int],
                                    depth_map: np.ndarray, h_img: int, w_img: int) -> Optional[float]:
        """Extract robust depth value using weighted average across entire bbox

        Instead of sampling at just the center or upper chest, this method:
        1. Samples depth at multiple points across the bbox
        2. Uses weighted average (closer to center = higher weight)
        3. Filters out invalid values (NaN, outliers)

        This prevents accidental depth changes when person raises arm/hand in front.

        Args:
            bbox: (x1, y1, x2, y2) bounding box
            depth_map: ZED2 depth registration image
            h_img, w_img: Image dimensions

        Returns:
            depth_val: Weighted median depth value in meters, or None if invalid
        """
        if depth_map is None or depth_map.size == 0:
            return None

        x1, y1, x2, y2 = bbox
        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

        # Ensure within bounds
        if not (0 <= x1 < w_img and 0 <= x2 <= w_img and
                0 <= y1 < h_img and 0 <= y2 <= h_img):
            return None

        # Extract bbox region
        roi_depth = depth_map[y1:y2, x1:x2]
        if roi_depth.size == 0:
            return None

        # Filter valid depths: no NaN, positive, reasonable range (0.3-8.0m)
        valid_mask = ~np.isnan(roi_depth) & (roi_depth > 0.3) & (roi_depth < 8.0)
        valid_depths = roi_depth[valid_mask]

        if len(valid_depths) == 0:
            return None

        # Calculate weighted average using distance from center as weight
        # Closer to center = higher weight (inverse squared distance)
        bbox_h = y2 - y1
        bbox_w = x2 - x1
        cy_center = bbox_h / 2.0
        cx_center = bbox_w / 2.0

        # Create coordinate grids for weighted calculation
        y_coords, x_coords = np.where(valid_mask)
        distances = np.sqrt((x_coords - cx_center)**2 + (y_coords - cy_center)**2)

        # Inverse squared distance weighting (closer points have more influence)
        # Add small epsilon to prevent division by zero
        weights = 1.0 / (1.0 + distances**2)
        weights = weights / np.sum(weights)  # Normalize

        # Weighted median: sort by depth and accumulate weights
        sorted_indices = np.argsort(valid_depths)
        sorted_depths = valid_depths[sorted_indices]
        sorted_weights = weights[sorted_indices]

        cumsum_weights = np.cumsum(sorted_weights)
        median_idx = np.searchsorted(cumsum_weights, 0.5)
        median_idx = min(median_idx, len(sorted_depths) - 1)

        return float(sorted_depths[median_idx])

    def _update_gesture_buffer(self, raw_gesture: str) -> str:
        """Update gesture buffer with voting mechanism (Hysteresis - Anti-Flickering)
        
        Per Gemini recommendation: Use buffer voting to eliminate gesture flickering.
        Only confirm gesture when it appears in ≥60% of buffer frames (9/15 at default).
        
        This prevents "detect then immediately disappear" issue by requiring consensus
        across multiple frames before confirming a command.
        
        Args:
            raw_gesture: Current frame's detected gesture ("START", "STOP", or "NONE")
            
        Returns:
            confirmed_gesture: Gesture confirmed by voting ("START", "STOP", or "NONE")
        """
        # Add current gesture to buffer
        self.gesture_buffer.append(raw_gesture)
        
        # Count occurrences in buffer
        start_count = self.gesture_buffer.count("START")
        stop_count = self.gesture_buffer.count("STOP")
        
        # Determine confirmed gesture based on majority voting threshold (Gemini's voting mechanism)
        # Need 60% consensus (9/15 frames by default) for confirmation
        min_required = int(self.gesture_buffer_size * self.gesture_confirmation_ratio)
        
        if start_count >= min_required:
            self.confirmed_gesture = "START"
        elif stop_count >= min_required:
            self.confirmed_gesture = "STOP"
        else:
            self.confirmed_gesture = "NONE"
        
        # Debug: Log voting state occasionally
        if len(self.gesture_buffer) == self.gesture_buffer_size:  # Only when buffer full
            if raw_gesture != "NONE":
                self.get_logger().debug(
                    f'📊 Gesture Voting: START={start_count}/{self.gesture_buffer_size}, '
                    f'STOP={stop_count}/{self.gesture_buffer_size}, '
                    f'CONFIRMED={self.confirmed_gesture} (need {min_required})'
                )
        
        return self.confirmed_gesture

    def _synced_callback(self, img_msg: Image, depth_msg: Image):
        """Process RGB and Depth simultaneously (perfectly synchronized by message_filters)
        
        This callback is triggered when RGB and Depth messages arrive within the
        synchronization window (0.1s), ensuring 3D coordinates are computed from
        perfectly time-aligned sensor readings.
        """
        ts_recv = time.time()

        try:
            # 🎬 DECODE RGB IMAGE
            frame = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
            if frame is None or frame.size == 0:
                raise CvBridgeError('Empty frame')

            # Validate BGR format (H, W, 3)
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                raise CvBridgeError(f'Invalid BGR frame shape: {frame.shape}, expected (H, W, 3)')
            
            # 🌊 DECODE DEPTH MAP (synchronized with RGB)
            # This depth map is guaranteed to be from the same instant as RGB
            depth_map = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            if depth_map is not None and depth_map.size > 0:
                self._current_depth_map = depth_map
            
            # Log frame info on first frame
            if not self._frame_dims_logged:
                self._frame_dims_logged = True
                self.get_logger().info(
                    f'📸 First synced frame: RGB {frame.shape} + Depth {depth_map.shape if depth_map is not None else "None"} '
                    f'(timestamp diff: {abs(img_msg.header.stamp.sec - depth_msg.header.stamp.sec)} ns)'
                )

        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error: {e}', throttle_duration_sec=2.0)
            return

        # ===== FRAME SKIP LOGIC (if CPU overloaded) =====
        if self.enable_frame_skip:
            time_since_last = time.time() - self._last_frame_time
            avg_inference_time = np.mean(self._inference_times) if self._inference_times else 0
            
            # Skip frame if inference is slower than 1/FPS (30ms at 30 FPS)
            if avg_inference_time > self.max_inference_time and time_since_last < 0.033:
                self._last_valid_frame = frame.copy()
                return

        self._process_frame(frame, img_msg.header.stamp, ts_recv)

    def _fallback_image_callback(self, msg: Image):
        """Fallback callback if depth synchronization is unavailable
        
        Used only when HAS_DEPTH_NMS is False (no Depth-Aware NMS available).
        Falls back to image-only processing without depth synchronization.
        """
        ts_recv = time.time()

        if msg.width == 0 or msg.height == 0 or not msg.data:
            return

        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if frame is None or frame.size == 0:
                raise CvBridgeError('Empty frame')

            if len(frame.shape) != 3 or frame.shape[2] != 3:
                raise CvBridgeError(f'Invalid BGR frame shape: {frame.shape}')
            
            if not self._frame_dims_logged:
                self._frame_dims_logged = True
                self.get_logger().info(f'📸 First frame (fallback, no depth sync): {frame.shape}')

        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error (fallback): {e}', throttle_duration_sec=2.0)
            return

        if self.enable_frame_skip:
            time_since_last = time.time() - self._last_frame_time
            avg_inference_time = np.mean(self._inference_times) if self._inference_times else 0
            if avg_inference_time > self.max_inference_time and time_since_last < 0.033:
                self._last_valid_frame = frame.copy()
                return

        self._process_frame(frame, msg.header.stamp, ts_recv)

    def _camera_info_callback(self, msg: CameraInfo):
        """Extract camera intrinsics (fx, cx, cy) from camera_info topic"""
        if not self.camera_info_received:
            # Extract from projection matrix K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            # K[0]=fx, K[2]=cx, K[5]=cy (fy in K[4])
            if len(msg.K) >= 6:
                self.fx = float(msg.K[0])
                self.cx_cam = float(msg.K[2])
                self.cy_cam = float(msg.K[5])
                self.camera_info_received = True
                self.get_logger().info(
                    f'✅ Camera INFO received (WVGA 672x376):\n'
                    f'   fx={self.fx:.3f} | cx={self.cx_cam:.3f} | cy={self.cy_cam:.3f}'
                )

    def _process_frame(self, frame: np.ndarray, ts, ts_recv: float):
        """Main processing pipeline - with timing"""
        ts_process_start = time.time()
        h_img, w_img = frame.shape[:2]
        ts_now = self.get_clock().now().nanoseconds * 1e-9

        # ===== DETECT PERSONS =====
        person_detections = []
        try:
            person_results = self._person_yolo.predict(
                frame,
                conf=self.person_conf,
                verbose=False,
                device=self._device,
                half=self.use_fp16
            )[0]

            if person_results is not None and len(person_results.boxes) > 0:
                for idx, box in enumerate(person_results.boxes.data):
                    x1, y1, x2, y2, conf, _ = box.cpu().numpy()
                    
                    # Validate coordinates
                    if not (0 <= x1 < w_img and 0 <= x2 < w_img and 
                            0 <= y1 < h_img and 0 <= y2 < h_img):
                        continue
                    
                    # BoT-SORT with Two-Stage Matching is sufficient for person validation
                    # (No need for height check - triggers frozen bbox when raising arms)
                    
                    person_detections.append({
                        'box': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(conf),
                        'class_id': 0,
                        'label': 'person'
                    })

        except Exception as e:
            self.get_logger().warn(f'Person detection error: {e}', throttle_duration_sec=2.0)

        # ===== DETECT HANDS (FULL FRAME INFERENCE - KHỚP VỚI REF_YOLO) =====
        # Chạy mô hình Hand trực tiếp trên TOÀN BỘ khung hình gốc (giống hệt detect_handsign.py)
        # Full-frame inference: mô hình được huấn luyện trên toàn khung hình, không bị scale distortion
        # Sử dụng self.hand_conf từ params.yaml - NO HARDCODING
        hand_detections = []
        try:
            # Dùng đúng self.hand_conf lấy từ params.yaml (không hardcode 0.30/0.85/0.55)
            hand_results = self._hand_yolo.predict(
                frame, 
                conf=self.hand_conf,  # From params.yaml - allows START gesture (~0.40-0.50) to pass
                verbose=False, 
                device=self._device, 
                half=self.use_fp16
            )[0]
            
            if hand_results is not None and len(hand_results.boxes) > 0:
                for box in hand_results.boxes.data:
                    x1, y1, x2, y2, conf, cls_id = box.cpu().numpy()
                    cls_id_int = int(cls_id)
                    
                    # ❌ REMOVED HARDCODED 0.85/0.55 THRESHOLDS ❌
                    # Thresholds are now set only in params.yaml via self.hand_conf
                    # This allows START gesture (thumbs-up ~0.40-0.50) to pass through
                    
                    # Validate coordinates
                    if not (0 <= x1 < w_img and 0 <= x2 < w_img and 
                            0 <= y1 < h_img and 0 <= y2 < h_img):
                        continue
                    
                    # Filter by Y-position to ignore ceiling/floor noise
                    hcy = (y1 + y2) / 2
                    if not (self.gesture_y_min_ratio * h_img <= hcy <= self.gesture_y_max_ratio * h_img):
                        continue
                    
                    hand_detections.append({
                        'box': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(conf),
                        'conf': float(conf),
                        'class_id': cls_id_int,
                        'label': "START" if cls_id_int == 0 else "STOP"
                    })

            # ===== APPLY DEPTH-AWARE NMS =====
            if HAS_DEPTH_NMS and self._depth_nms and self._current_depth_map is not None and len(hand_detections) > 0:
                try:
                    hand_detections = self._depth_nms.apply(
                        hand_detections, 
                        self._current_depth_map,
                        conf_threshold=0.0
                    )
                except Exception as e:
                    self.get_logger().warn(f'Depth-Aware NMS error: {e}', throttle_duration_sec=5.0)

        except Exception as e:
            self.get_logger().error(f'Hand detection root error: {e}', throttle_duration_sec=2.0)

        # ===== TRÍCH XUẤT ĐẶC TRƯNG ẢNH (DEEP LEARNING RE-ID) =====
        # Extract ResNet18 512-dim features for each person detection via GPU
        features_list = []
        for det in person_detections:
            # Only extract features for confident detections (optimization)
            if det['confidence'] >= self._person_tracker.track_thresh:
                x1, y1, x2, y2 = det['box']
                # Crop person region from frame
                crop_img = frame[max(0, int(y1)):min(h_img, int(y2)), max(0, int(x1)):min(w_img, int(x2))]
                # Extract 512-dim appearance feature using ResNet18 on RTX A4000 (Tensor Cores FP16)
                feat = self._appearance_extractor.extract(crop_img)
            else:
                # Low confidence detection gets empty feature vector
                feat = self._appearance_extractor.get_empty_feature()
            features_list.append(feat)
            
        features_np = np.array(features_list) if len(features_list) > 0 else None

        # ===== TRACKING NGƯỜI (ĐÃ KÍCH HOẠT RE-ID) =====
        # Person tracking via BoT-SORT with deep learning appearance features enabled
        person_dets_np = self._dicts_to_numpy(person_detections)
        # ✅ NOW PASSING FEATURES: ResNet18 enables robust Re-ID after occlusion/re-entrance
        person_tracks_np = self._person_tracker.update(person_dets_np, features=features_np)
        person_tracks = self._numpy_to_dicts(person_tracks_np, person_detections)
        
        # ĐỐI VỚI TAY: Dùng trực tiếp kết quả YOLO, bỏ qua BoTSort để không bị mất nhãn START/STOP
        # (BoTSort tracking causes label mismatch when hand detection flickers)
        hand_tracks = []
        for i, det in enumerate(hand_detections):
            hand_tracks.append({
                'track_id': 9990 + i,  # Dummy ID (not used for gesture timing)
                'bbox': det['box'],
                'confidence': det.get('confidence', det.get('conf', 0.5)),
                'label': det['label'],
                'class_id': det['class_id']
            })

        # ===== GẮN TAY VÀO NGƯỜI =====
        hand_to_person = self._associate_hands_to_persons(hand_tracks, person_tracks)

        # Update current state (for visualization and logging)
        self._current_persons = [
            {
                'idx': t['track_id'],
                'bbox': t['bbox'],
                'conf': t.get('confidence', 0.8),
                'track_id': t['track_id'],
                'gestures': []
            }
            for t in person_tracks
        ]
        self._current_gestures = []

        # ===== KIỂM TRA MỤC TIÊU CÓ BỊ MẤT DẤU KHÔNG =====
        self.target_visible = any(t['track_id'] == self.main_target_id for t in person_tracks)
        
        if self.main_target_id is not None and not self.target_visible:
            if self.current_state == 'TRACKING':
                self.current_state = 'Re-TRACKING'
                self.target_lost_time = ts_now
                self.get_logger().info('🔄 Target lost, switching to Re-TRACKING (5s prediction)')
        
        if self.current_state == 'Re-TRACKING' and self.target_lost_time is not None:
            if ts_now - self.target_lost_time > 5.0:
                self.current_state = 'IDLE'
                self.main_target_id = None
                self.target_lost_time = None
                self.get_logger().info('⏱️ Re-TRACKING timeout, returning to IDLE')
                
        if self.current_state == 'Re-TRACKING' and self.target_visible:
            self.current_state = 'TRACKING'
            self.target_lost_time = None
            self.get_logger().info('✅ Target reacquired, back to TRACKING')

        # ===== GESTURE STATE MACHINE (WITH HYSTERESIS BUFFER VOTING) =====
        # Use gesture buffer voting mechanism per Gemini recommendation:
        # Confirms gesture only when it appears in 70%+ of recent frames
        # This eliminates "detect then disappear" flickering
        
        # Detect raw gesture from current frame
        raw_gesture_this_frame = "NONE"
        matched_person_for_gesture = None
        
        for hand_track in hand_tracks:
            label = hand_track['label'].upper()
            track_id = hand_track['track_id']
            matched_person_id = hand_to_person.get(track_id)
            
            if matched_person_id is None:
                continue
            
            # OPTIMIZATION: Check depth consistency (hand must be reasonably close to person)
            # This eliminates false positives from background objects with similar geometry
            person_bbox = next((p['bbox'] for p in person_tracks if p['track_id'] == matched_person_id), None)
            if person_bbox and self._current_depth_map is not None:
                hand_depth = self._get_depth_at_center(hand_track['bbox'], self._current_depth_map, h_img, w_img)
                person_depth = self._get_depth_at_center(person_bbox, self._current_depth_map, h_img, w_img)
                
                # If hand is more than 60cm away from person, reject it (likely false positive)
                if hand_depth is not None and person_depth is not None:
                    if abs(hand_depth - person_depth) > self.max_hand_person_depth_diff:
                        continue
            
            # Detect which gesture (START or STOP)
            if label == "START" and self.current_state == 'IDLE':
                raw_gesture_this_frame = "START"
                matched_person_for_gesture = matched_person_id
                break  # Take first valid gesture in frame
            elif label == "STOP" and self.current_state in ['TRACKING', 'Re-TRACKING']:
                if matched_person_id == self.main_target_id:
                    raw_gesture_this_frame = "STOP"
                    matched_person_for_gesture = matched_person_id
                    break
        
        # Update gesture buffer with voting mechanism (Hysteresis filter)
        confirmed_gesture = self._update_gesture_buffer(raw_gesture_this_frame)
        
        # Now check if confirmed gesture meets timing requirement
        if confirmed_gesture == "START" and self.current_state == 'IDLE':
            if not self.start_signal_detected:
                self.start_signal_detected = True
                self.start_signal_time = ts_now
                self.last_start_seen_time = ts_now
                self.get_logger().info(f'▶ START signal CONFIRMED via buffer voting (Person #{matched_person_for_gesture}). Holding for {self.gesture_hold_time}s...')
            else:
                self.last_start_seen_time = ts_now
                elapsed = ts_now - self.start_signal_time
                
                # Only time check now (buffer voting already filters noise)
                if elapsed >= self.gesture_hold_time:
                    self.main_target_id = matched_person_for_gesture
                    self.current_state = 'TRACKING'
                    self.target_lock_time = ts_now
                    self.target_visible = True
                    
                    if matched_person_for_gesture not in self.ekfs:
                        self.ekfs[matched_person_for_gesture] = CTRV_EKF(dt=1/30.0)
                        self.velocity_filters[matched_person_for_gesture] = VelocityFilter(alpha=0.3)
                        self.position_history[matched_person_for_gesture] = deque(maxlen=self.max_position_history)
                    
                    self.get_logger().info(f'🔒 LOCKED onto Person #{matched_person_for_gesture} via START gesture')
                    self.start_signal_detected = False
                    self.start_signal_time = None
                    self.gesture_buffer.clear()  # Clear buffer after confirming action
                    
                    msg = String()
                    msg.data = f"START(1000ms)<p={matched_person_for_gesture}>"
                    self._pub_gesture.publish(msg)
        
        elif confirmed_gesture == "STOP" and self.current_state in ['TRACKING', 'Re-TRACKING']:
            if self.stop_detected_time is None:
                self.stop_detected_time = ts_now
                self.last_stop_seen_time = ts_now
                self.get_logger().info(f'⏹ STOP signal CONFIRMED via buffer voting from Target #{matched_person_for_gesture}. Holding for {self.gesture_hold_time}s...')
            else:
                self.last_stop_seen_time = ts_now
                elapsed = ts_now - self.stop_detected_time
                
                # Only time check now (buffer voting already filters noise)
                if elapsed >= self.gesture_hold_time:
                    self.current_state = 'IDLE'
                    self.main_target_id = None
                    self.target_lost_time = None
                    self.get_logger().info(f'🔓 Tracking STOPPED by user via STOP gesture')
                    self.stop_detected_time = None
                    self.gesture_buffer.clear()  # Clear buffer after confirming action
                    
                    msg = String()
                    msg.data = f"STOP(1000ms)<p={matched_person_for_gesture}>"
                    self._pub_gesture.publish(msg)
        
        else:
            # Gesture no longer confirmed via voting buffer
            if self.start_signal_detected and confirmed_gesture != "START":
                if self.last_start_seen_time and (ts_now - self.last_start_seen_time > self.signal_timeout):
                    self.start_signal_detected = False
                    self.start_signal_time = None
                    self.get_logger().info('⏳ START gesture lost (not confirmed by buffer). Timer reset.')
            
            if self.stop_detected_time is not None and confirmed_gesture != "STOP":
                if self.last_stop_seen_time and (ts_now - self.last_stop_seen_time > self.signal_timeout):
                    self.stop_detected_time = None
                    self.get_logger().info('⏳ STOP gesture lost (not confirmed by buffer). Timer reset.')

        # ===== PUBLISH KẾT QUẢ ĐỂ HIỂN THỊ VÀ STATE MACHINE =====
        self._publish_annotated_image(frame, person_tracks, hand_tracks, h_img, w_img)
        self._publish_human_state(person_tracks, h_img, w_img, ts_now)

        # ===== TIMING METRICS =====
        inference_time = time.time() - ts_process_start
        self._inference_times.append(inference_time)
        frame_latency = time.time() - ts_recv
        self._frame_times.append(frame_latency)
        self._last_frame_time = time.time()

    def _dicts_to_numpy(self, detections: List[Dict]) -> np.ndarray:
        """Convert detection dicts to numpy array [x1, y1, x2, y2, conf]"""
        if not detections:
            return np.empty((0, 5), dtype=np.float32)
        dets = []
        for det in detections:
            x1, y1, x2, y2 = det['box']
            conf = det['confidence']
            dets.append([x1, y1, x2, y2, conf])
        return np.array(dets, dtype=np.float32)

    def _numpy_to_dicts(self, tracks_np: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """Convert BoT-SORT output to dict format"""
        tracks = []
        for i, track in enumerate(tracks_np):
            x1, y1, x2, y2, track_id = track
            det_info = detections[i] if i < len(detections) else {}
            tracks.append({
                'track_id': int(track_id),
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': det_info.get('confidence', 0.5),
                'label': det_info.get('label', 'unknown'),
                'class_id': det_info.get('class_id', 0),
            })
        return tracks

    def _get_bbox_center(self, bbox: Tuple) -> Tuple[int, int]:
        """Get bbox center"""
        x1, y1, x2, y2 = bbox
        return ((int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2)

    def _associate_hands_to_persons(self, hand_tracks: List[Dict],
                                    person_tracks: List[Dict]) -> Dict[int, Optional[int]]:
        """Ghép tay vào người dựa trên khoảng cách tâm (Bỏ giới hạn Bounding Box)
        
        Uses Euclidean distance from hand center to person center instead of fixed buffer box.
        This allows detection of extended arms without being blocked by spatial buffer limits.
        """
        hand_to_person = {}

        for hand_track in hand_tracks:
            hcx, hcy = self._get_bbox_center(hand_track['bbox'])
            best_person_id = None
            best_distance = float('inf')

            for person_track in person_tracks:
                pcx, pcy = self._get_bbox_center(person_track['bbox'])
                
                # Tính khoảng cách từ tâm tay đến tâm người (Euclidean distance)
                distance = ((hcx - pcx) ** 2 + (hcy - pcy) ** 2) ** 0.5

                # 350 pixel là khoảng cách an toàn bao trọn sải tay trên khung hình VGA (640x480)
                if distance < best_distance and distance < 350:
                    best_distance = distance
                    best_person_id = person_track['track_id']

            hand_to_person[hand_track['track_id']] = best_person_id

        return hand_to_person

    def _get_depth_at_center(self, bbox: Tuple[int, int, int, int], 
                              depth_map: np.ndarray, h_img: int, w_img: int, 
                              cy_override: Optional[int] = None) -> Optional[float]:
        """Extract robust depth value at bbox center using 5x5 median filtering
        
        Args:
            bbox: (x1, y1, x2, y2) bounding box
            depth_map: ZED2 depth registration image
            h_img, w_img: Image dimensions
            cy_override: Optional override for cy coordinate (for upper chest sampling)
            
        Returns:
            depth_val: Median depth value in meters, or None if invalid
        """
        if depth_map is None or depth_map.size == 0:
            return None
        
        x1, y1, x2, y2 = bbox
        if cy_override is not None:
            cx = (int(x1) + int(x2)) // 2
            cy = cy_override
        else:
            cx, cy = (int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2
        
        # Ensure within bounds
        if not (0 <= cx < w_img and 0 <= cy < h_img):
            return None
        
        # Extract 5x5 ROI around center (robust to single pixel errors)
        roi_y1 = max(0, cy - 2)
        roi_y2 = min(h_img, cy + 3)
        roi_x1 = max(0, cx - 2)
        roi_x2 = min(w_img, cx + 3)
        
        roi_depth = depth_map[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Filter valid depths: no NaN, positive, reasonable range (0.3-8.0m)
        valid_depths = roi_depth[~np.isnan(roi_depth) & (roi_depth > 0.3) & (roi_depth < 8.0)]
        
        if len(valid_depths) > 0:
            return float(np.median(valid_depths))
        
        return None

    def _publish_human_state(self, person_tracks: List[Dict],
                             h_img: int, w_img: int, ts_now: float):
        """Extract 3D coordinates, run EKF filtering, and publish to specific topics
        using custom interfaces (HumanState & ObstacleArray).
        
        Logic:
        - IDLE state: Publish all people as obstacles (ObstacleArray)
        - TRACKING state:
          * Publish main_person to /tracking/main_person (HumanState) with radius=1.5m
          * Publish other people to /tracking/obstacles (ObstacleArray with DynaObstacle items)
        """
        try:
            fx = self.fx
            cx_cam = self.cx_cam
            cy_cam = self.cy_cam

            # Initialize ObstacleArray message for all non-main-target people
            obstacle_array_msg = ObstacleArray()
            obstacle_array_msg.header = Header()
            obstacle_array_msg.header.stamp = self.get_clock().now().to_msg()
            obstacle_array_msg.header.frame_id = "camera_link"

            # ===== HELPER FUNCTION FOR COMMON PROCESSING =====
            def process_person_track(tid: int, bbox: Tuple, is_main_target: bool = False):
                """Process a single person track and return (position, velocity, radius, depth_val)"""
                x1, y1, x2, y2 = bbox

                # Initialize EKF if needed
                if tid not in self.ekfs:
                    self.ekfs[tid] = CTRV_EKF(dt=1/30.0)
                    self.velocity_filters[tid] = VelocityFilter(alpha=0.3)
                    self.position_history[tid] = deque(maxlen=self.max_position_history)

                ekf = self.ekfs[tid]

                # Calculate real dt from timestamps
                real_dt = self._get_real_dt(tid, ts_now)
                ekf.dt = real_dt

                # Use weighted depth across entire bbox
                depth_val = self._get_weighted_depth_in_bbox(bbox, self._current_depth_map, h_img, w_img)

                if depth_val is None:
                    # No depth measurement, just predict
                    ekf.predict()
                else:
                    # Compute 3D position from depth measurement
                    cx = (int(x1) + int(x2)) // 2
                    cy = int(y1 + (y2 - y1) * 0.3)  # Upper chest region

                    px_3d = float(depth_val)
                    py_3d = float(-(cx - cx_cam) * depth_val / fx)

                    # Track position history for velocity estimation
                    pos_3d = np.array([px_3d, py_3d])
                    self.position_history[tid].append(pos_3d)

                    # Calculate velocity from position history using real dt
                    if len(self.position_history[tid]) >= 2:
                        prev_pos = self.position_history[tid][-2]
                        meas_vel = (pos_3d - prev_pos) / real_dt

                        if np.linalg.norm(meas_vel) > 10.0:  # Sanity check (10 m/s max)
                            meas_vel = np.array([0.0, 0.0])
                    else:
                        meas_vel = np.array([0.0, 0.0])

                    # EKF predict then update
                    ekf.predict()
                    ekf.update(pos_3d, meas_vel)

                # Extract filtered state
                ekf_x = float(ekf.state.x)
                ekf_y = float(ekf.state.y)
                ekf_vx = float(ekf.state.v * np.cos(ekf.state.psi))
                ekf_vy = float(ekf.state.v * np.sin(ekf.state.psi))

                # Apply velocity filter
                vel_filter = self.velocity_filters[tid]
                filtered_vel = vel_filter.update(np.array([ekf_vx, ekf_vy, 0.0]))
                final_vx, final_vy = float(filtered_vel[0]), float(filtered_vel[1])

                # Calculate obstacle radius from bbox width
                if depth_val:
                    radius = float(abs(x2 - x1) * depth_val / fx / 2.0 * 0.85)
                else:
                    radius = 0.5

                return ekf_x, ekf_y, final_vx, final_vy, radius, depth_val

            def create_predicted_trajectory(ekf_x: float, ekf_y: float, 
                                          final_vx: float, final_vy: float) -> Path:
                """Create predicted trajectory for next 3 seconds (6 steps of 0.5s each)"""
                predicted_path = Path()
                predicted_path.header = obstacle_array_msg.header
                
                for i in range(6):
                    dt_pred = i * 0.5
                    pose = PoseStamped()
                    pose.header = obstacle_array_msg.header
                    pose.pose.position.x = ekf_x + final_vx * dt_pred
                    pose.pose.position.y = ekf_y + final_vy * dt_pred
                    pose.pose.position.z = 0.0
                    predicted_path.poses.append(pose)
                
                return predicted_path

            # ===== CASE 1: IDLE STATE - Publish all people as obstacles =====
            if self.current_robot_state == 'IDLE':
                for person_track in person_tracks:
                    tid = person_track['track_id']
                    ekf_x, ekf_y, final_vx, final_vy, radius, depth_val = \
                        process_person_track(tid, person_track['bbox'], is_main_target=False)
                    
                    # Create trajectory prediction
                    predicted_path = create_predicted_trajectory(ekf_x, ekf_y, final_vx, final_vy)
                    
                    # Add to ObstacleArray
                    obs = DynaObstacle()
                    obs.id = float(tid)
                    obs.distance = float(math.sqrt(ekf_x**2 + ekf_y**2))
                    obs.radius = float(radius)
                    obs.trajectory = predicted_path
                    obstacle_array_msg.dyna_obstacles.append(obs)
                    
                    self.get_logger().debug(f'📌 IDLE obstacle (ID={tid}): px={ekf_x:.2f}, py={ekf_y:.2f}, r={radius:.2f}m')

                # Publish obstacle array (even if empty)
                self._pub_obstacles.publish(obstacle_array_msg)
                return

            # ===== CASE 2: TRACKING STATE - Publish main_person + other obstacles =====
            elif self.current_robot_state in ['TRACKING', 'RE_TRACKING']:
                if self.main_target_id is None:
                    # No main target yet, publish all as obstacles
                    for person_track in person_tracks:
                        tid = person_track['track_id']
                        ekf_x, ekf_y, final_vx, final_vy, radius, depth_val = \
                            process_person_track(tid, person_track['bbox'], is_main_target=False)
                        
                        # Create trajectory prediction
                        predicted_path = create_predicted_trajectory(ekf_x, ekf_y, final_vx, final_vy)
                        
                        # Add to ObstacleArray
                        obs = DynaObstacle()
                        obs.id = float(tid)
                        obs.distance = float(math.sqrt(ekf_x**2 + ekf_y**2))
                        obs.radius = float(radius)
                        obs.trajectory = predicted_path
                        obstacle_array_msg.dyna_obstacles.append(obs)
                    
                    self._pub_obstacles.publish(obstacle_array_msg)
                    return

                # Process ALL people in frame
                for person_track in person_tracks:
                    tid = person_track['track_id']
                    ekf_x, ekf_y, final_vx, final_vy, radius, depth_val = \
                        process_person_track(tid, person_track['bbox'], is_main_target=(tid == self.main_target_id))

                    # Create trajectory prediction
                    predicted_path = create_predicted_trajectory(ekf_x, ekf_y, final_vx, final_vy)

                    if tid == self.main_target_id:
                        # Publish main target as HumanState
                        msg = HumanState()
                        msg.px = ekf_x
                        msg.py = ekf_y
                        msg.vx = final_vx
                        msg.vy = final_vy
                        msg.radius = 1.5  # Safety distance for main person
                        msg.trajectory = [predicted_path]
                        
                        self._pub_human_state.publish(msg)
                        self.get_logger().debug(f'🎯 Main target (ID={tid}): px={msg.px:.2f}, py={msg.py:.2f}, vx={msg.vx:.2f}, vy={msg.vy:.2f}')
                    else:
                        # Publish other people as obstacles
                        obs = DynaObstacle()
                        obs.id = float(tid)
                        obs.distance = float(math.sqrt(ekf_x**2 + ekf_y**2))
                        obs.radius = float(radius)
                        obs.trajectory = predicted_path
                        obstacle_array_msg.dyna_obstacles.append(obs)
                        
                        self.get_logger().debug(f'⚠️  Obstacle (ID={tid}): px={ekf_x:.2f}, py={ekf_y:.2f}, r={radius:.2f}m')

                # Publish all obstacles at once (ObstacleArray)
                self._pub_obstacles.publish(obstacle_array_msg)
                return

            # No-op for other states

        except Exception as e:
            self.get_logger().error(f'Error in _publish_human_state: {e}', throttle_duration_sec=2.0)

    def _log_performance_metrics(self):
        """Log performance metrics every 5 seconds - Simplified"""
        if not self._inference_times or not self._frame_times:
            return
        
        avg_latency = np.mean(self._frame_times) * 1000
        fps = 1.0 / np.mean(self._frame_times)
        
        self.get_logger().info(
            f'STATE: {self.current_state:12s} | FPS: {fps:.1f} | Avg Latency: {avg_latency:.1f}ms'
        )

    def _publish_annotated_image(self, frame: np.ndarray, person_tracks: List[Dict],
                                 hand_tracks: List[Dict], h_img: int, w_img: int):
        """Draw bounding boxes and publish annotated image for debugging
        
        Colors:
        - GREEN: Person detections (tracked)
        - YELLOW: Hand gestures (START/STOP)
        - RED: Main target person (locked for tracking)
        """
        try:
            # Make a copy to avoid modifying original
            annotated = frame.copy()
            
            # Draw person bounding boxes (GREEN for normal, RED for main target)
            for person_track in person_tracks:
                x1, y1, x2, y2 = person_track['bbox']
                track_id = person_track['track_id']
                
                # Choose color based on whether this is the main target
                if track_id == self.main_target_id:
                    color = (0, 0, 255)  # RED for main target
                    thickness = 3  # Thicker for visibility
                else:
                    color = (0, 255, 0)  # GREEN for other persons
                    thickness = 2
                
                # Draw rectangle
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                
                # Draw label with track ID
                label = f'Person #{track_id}'
                cv2.putText(annotated, label, (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw hand gesture bounding boxes (YELLOW)
            for hand_track in hand_tracks:
                x1, y1, x2, y2 = hand_track['bbox']
                track_id = hand_track['track_id']
                label = hand_track['label']
                conf = hand_track['confidence']
                
                # YELLOW for hand gestures
                color = (0, 255, 255)
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Draw label with gesture type and confidence
                text = f'{label} #{track_id} ({conf:.2f})'
                cv2.putText(annotated, text, (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw state machine status in top-left corner
            state_color_map = {
                'IDLE': (200, 200, 200),       # Gray
                'TRACKING': (0, 255, 0),      # Green
                'Re-TRACKING': (0, 255, 255) # Yellow
            }
            state_color = state_color_map.get(self.current_state, (255, 255, 255))
            state_text = f'State: {self.current_state}'
            cv2.putText(annotated, state_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
            
            # Draw FPS in top-right corner
            if self._frame_times:
                fps = 1.0 / np.mean(self._frame_times)
                fps_text = f'FPS: {fps:.1f}'
                text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.putText(annotated, fps_text, (w_img - text_size[0] - 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Convert to ROS Image message and publish
            img_msg = self._bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            self._pub_annotated_image.publish(img_msg)
        
        except Exception as e:
            self.get_logger().debug(f'Annotation error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = TrackingNode()
    # ✅ OPTIMIZATION: Use MultiThreadedExecutor to prevent callback blocking
    # This allows image_callback to run in parallel without blocking depth/camera_info callbacks
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
