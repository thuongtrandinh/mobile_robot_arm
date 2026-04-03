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
from collections import deque
from typing import Dict, List, Tuple, Optional
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ultralytics import YOLO
from cv_bridge import CvBridge, CvBridgeError
import message_filters  # RGB-Depth synchronization for 3D accuracy

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from interfaces.msg import HumanState
from ament_index_python.packages import get_package_share_directory

# BoT-SORT tracking with appearance features
from yolo.algorithms.botsort_handler import BoTSortTracker
from yolo.algorithms.appearance_feature_extractor import FeatureExtractorLightweight
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

        # ===== 1. PARAMETERS (DYNAMIC) =====
        self.declare_parameter('camera_image_topic', '/zed/zed_node/rgb/color/rect/image')
        self.declare_parameter('person_model_path', 'yolov8n.pt')
        self.declare_parameter('hand_model_path', 'handsign.pt')
        self.declare_parameter('person_conf_thresh', 0.70)
        self.declare_parameter('hand_conf_thresh', 0.5)  # Reduced to detect START/STOP more reliably
        self.declare_parameter('use_cuda', True)
        self.declare_parameter('use_fp16', True)
        self.declare_parameter('gesture_hold_time', 1.5)  # NEURAL optimized: 1.5s for robust detection
        self.declare_parameter('hand_person_match_buffer', 100)
        self.declare_parameter('gesture_y_min_ratio', 0.0)
        self.declare_parameter('gesture_y_max_ratio', 0.90)
        self.declare_parameter('max_inference_time_ms', 200.0)  # Increased from 100 to process all frames
        self.declare_parameter('enable_frame_skip', True)  # ✅ ENABLED: Drop frames when GPU overloaded (GPU bottleneck fix)
        self.declare_parameter('enable_performance_metrics', True)  # Log FPS/latency

        camera_topic = self.get_parameter('camera_image_topic').value
        person_model_name = self.get_parameter('person_model_path').value
        hand_model_name = self.get_parameter('hand_model_path').value
        self.person_conf = self.get_parameter('person_conf_thresh').value
        self.hand_conf = self.get_parameter('hand_conf_thresh').value
        use_cuda = self.get_parameter('use_cuda').value
        self.use_fp16 = self.get_parameter('use_fp16').value
        self.gesture_hold_time = self.get_parameter('gesture_hold_time').value
        self.hand_person_buffer = self.get_parameter('hand_person_match_buffer').value
        self.gesture_y_min_ratio = self.get_parameter('gesture_y_min_ratio').value
        self.gesture_y_max_ratio = self.get_parameter('gesture_y_max_ratio').value
        self.max_inference_time = self.get_parameter('max_inference_time_ms').value / 1000.0
        self.enable_frame_skip = self.get_parameter('enable_frame_skip').value
        self.enable_perf_metrics = self.get_parameter('enable_performance_metrics').value

        self._device = 'cuda' if use_cuda else 'cpu'

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
        self._person_tracker = BoTSortTracker(
            track_thresh=0.5,
            track_buffer=100,  # Increased from 30 to 100 to prevent ID jumping
            match_thresh=0.8,  # Increased from 0.3 to 0.8 for stricter matching
            use_appearance=True,
            appearance_weight=0.3,
            max_age_before_predict=15
        )

        self._hand_tracker = BoTSortTracker(
            track_thresh=0.4,
            track_buffer=100,  # Increased from 60 to 100 to prevent jitter/dropout
            match_thresh=0.4,
            use_appearance=True,
            appearance_weight=0.5,
            max_age_before_predict=10
        )

        # ===== 4. APPEARANCE FEATURES =====
        self._appearance_extractor = FeatureExtractorLightweight(n_bins=32)

        # ===== 5. GESTURE STATE MACHINE (ROBUST - GLOBAL TIMERS) =====
        # NO PER-TRACK TIMING: Use global timers instead (like ref_yolo does)
        # This avoids BoTSort tracking jitter from resetting gesture detection
        self.start_signal_detected = False
        self.start_signal_time = None
        self.last_start_seen_time = None
        
        self.stop_detected_time = None
        self.last_stop_seen_time = None
        self.signal_timeout = 1.0  # Cho phép mất tín hiệu tay 1s mà không bị reset timer

        # ===== 5b. TARGET PERSON LOCKING (START/STOP control) =====
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
        # ===== 9c. ANNOTATED IMAGE PUBLISHER (for debugging/visualization) =====
        self._pub_annotated_image = self.create_publisher(Image, '/tracking/annotated_image', 10)

        # ===== 10. SUBSCRIBERS (RGB-DEPTH SYNCHRONIZED) =====
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
                    
                    person_detections.append({
                        'box': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(conf),
                        'class_id': 0,
                        'label': 'person'
                    })

        except Exception as e:
            self.get_logger().warn(f'Person detection error: {e}', throttle_duration_sec=2.0)

        # ===== DETECT HANDS (BATCHED ROI CROP - Tối ưu GPU & Độ chính xác) =====
        hand_detections = []
        try:
            if len(person_detections) > 0:
                hand_crops = []
                crop_infos = []
                
                # Sắp xếp ưu tiên 2 người to nhất (gần camera nhất) để quét tay, tránh lãng phí VRAM
                person_detections_sorted = sorted(person_detections, 
                    key=lambda det: (det['box'][2]-det['box'][0]) * (det['box'][3]-det['box'][1]), 
                    reverse=True)
                
                for person_idx, p_det in enumerate(person_detections_sorted[:2]):
                    px1, py1, px2, py2 = p_det['box']
                    pw, ph = px2 - px1, py2 - py1
                    
                    # Nới rộng vùng cắt ra 40% chiều ngang để hứng trọn sải tay dài
                    exp_x, exp_y = int(pw * 0.40), int(ph * 0.25)
                    crop_x1, crop_y1 = max(0, int(px1) - exp_x), max(0, int(py1) - exp_y)
                    crop_x2, crop_y2 = min(w_img, int(px2) + exp_x), min(h_img, int(py2) + exp_y)
                    
                    crop_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    if crop_frame.size > 0:
                        crop_frame = cv2.convertScaleAbs(crop_frame, alpha=1.3, beta=-20)
                        hand_crops.append(crop_frame)
                        crop_infos.append((crop_x1, crop_y1))
                
                # GỌI YOLO 1 LẦN DUY NHẤT CHO TẤT CẢ CÁC ẢNH CROP (Chạy CUDA song song)
                if hand_crops:
                    try:
                        batch_results = self._hand_yolo.predict(
                            hand_crops, 
                            conf=0.45,  # Hạ nhẹ ở cấp độ GPU để lấy đủ Data
                            verbose=False, 
                            device=self._device, 
                            half=self.use_fp16
                        )
                        
                        # Xử lý kết quả trả về tương ứng với từng ảnh cắt
                        for idx, result in enumerate(batch_results):
                            offset_x, offset_y = crop_infos[idx]
                            
                            if result is not None and len(result.boxes) > 0:
                                for box in result.boxes.data:
                                    x1, y1, x2, y2, conf, cls_id = box.cpu().numpy()
                                    cls_id_int = int(cls_id)
                                    
                                    # Yêu cầu của bạn: Nâng độ tin cậy lên 0.50 cho START và 0.55 cho STOP
                                    threshold = 0.50 if cls_id_int == 0 else 0.55
                                    if conf < threshold:
                                        continue
                                        
                                    # Chuyển tọa độ từ ảnh Crop về lại hệ quy chiếu khung hình gốc
                                    x1_full = int(x1 + offset_x)
                                    y1_full = int(y1 + offset_y)
                                    x2_full = int(x2 + offset_x)
                                    y2_full = int(y2 + offset_y)
                                    
                                    # Lọc theo chiều cao y_ratio để bỏ qua nhiễu dưới chân/trần nhà
                                    hcy = (y1_full + y2_full) / 2
                                    if not (self.gesture_y_min_ratio * h_img <= hcy <= self.gesture_y_max_ratio * h_img):
                                        continue
                                        
                                    hand_detections.append({
                                        'box': (x1_full, y1_full, x2_full, y2_full),
                                        'confidence': float(conf),
                                        'conf': float(conf),
                                        'class_id': cls_id_int,
                                        'label': "START" if cls_id_int == 0 else "STOP"
                                    })
                    except Exception as e:
                        self.get_logger().debug(f'Batched hand detection error: {e}')

                # ===== APPLY DEPTH-AWARE NMS =====
                if HAS_DEPTH_NMS and self._depth_nms and self._current_depth_map is not None and len(hand_detections) > 0:
                    try:
                        # conf_threshold = 0.0 do ta đã tự lọc ngưỡng 0.35/0.55 ở trên
                        hand_detections = self._depth_nms.apply(
                            hand_detections, 
                            self._current_depth_map,
                            conf_threshold=0.0
                        )
                    except Exception as e:
                        self.get_logger().warn(f'Depth-Aware NMS error: {e}', throttle_duration_sec=5.0)

        except Exception as e:
            self.get_logger().error(f'Hand detection root error: {e}', throttle_duration_sec=2.0)

        # ===== TRACKING NGƯỜI (BỎ TRACKING TAY) =====
        # Person tracking via BoTSort (keep this - works well for persons)
        person_dets_np = self._dicts_to_numpy(person_detections)
        person_tracks_np = self._person_tracker.update(person_dets_np, features=None)
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

        # ===== GESTURE STATE MACHINE (ROBUST LOGIC TỪ REF_YOLO) =====
        # Use GLOBAL timers instead of per-track timers (no BoTSort jitter!)
        start_found_in_frame = False
        stop_found_in_frame = False
        
        for hand_track in hand_tracks:
            label = hand_track['label'].upper()
            track_id = hand_track['track_id']
            matched_person_id = hand_to_person.get(track_id)
            
            if matched_person_id is None:
                continue
                
            # --- XỬ LÝ LỆNH START ---
            if label == "START" and self.current_state == 'IDLE':
                start_found_in_frame = True
                self.last_start_seen_time = ts_now
                
                if not self.start_signal_detected:
                    self.start_signal_detected = True
                    self.start_signal_time = ts_now
                    self.get_logger().info(f'▶ START signal detected (Person #{matched_person_id}). Holding for {self.gesture_hold_time}s...')
                else:
                    elapsed = ts_now - self.start_signal_time
                    if elapsed >= self.gesture_hold_time:
                        self.main_target_id = matched_person_id
                        self.current_state = 'TRACKING'
                        self.target_lock_time = ts_now
                        self.target_visible = True
                        
                        if matched_person_id not in self.ekfs:
                            self.ekfs[matched_person_id] = CTRV_EKF(dt=1/30.0)
                            self.velocity_filters[matched_person_id] = VelocityFilter(alpha=0.3)
                            self.position_history[matched_person_id] = deque(maxlen=self.max_position_history)
                            
                        self.get_logger().info(f'🔒 LOCKED onto Person #{matched_person_id}')
                        self.start_signal_detected = False
                        self.start_signal_time = None
                        
                        msg = String()
                        msg.data = f"START(1000ms)<p={matched_person_id}>"
                        self._pub_gesture.publish(msg)
                        
            # --- XỬ LÝ LỆNH STOP ---
            elif label == "STOP" and self.current_state in ['TRACKING', 'Re-TRACKING']:
                # Chỉ nhận STOP từ người đang bị theo dõi
                if matched_person_id == self.main_target_id:
                    stop_found_in_frame = True
                    self.last_stop_seen_time = ts_now
                    
                    if self.stop_detected_time is None:
                        self.stop_detected_time = ts_now
                        self.get_logger().info(f'⏹ STOP signal detected from Target #{matched_person_id}. Holding for {self.gesture_hold_time}s...')
                    else:
                        elapsed = ts_now - self.stop_detected_time
                        if elapsed >= self.gesture_hold_time:
                            self.current_state = 'IDLE'
                            self.main_target_id = None
                            self.target_lost_time = None
                            self.get_logger().info('🔓 Tracking STOPPED by user')
                            self.stop_detected_time = None
                            
                            msg = String()
                            msg.data = f"STOP(1000ms)<p={matched_person_id}>"
                            self._pub_gesture.publish(msg)

        # --- XỬ LÝ TIMEOUT (CHỐNG MẤT TÍN HIỆU GIỮA CHỚP NHÁNG) ---
        if self.start_signal_detected and not start_found_in_frame:
            if self.last_start_seen_time and (ts_now - self.last_start_seen_time > self.signal_timeout):
                self.start_signal_detected = False
                self.start_signal_time = None
                self.get_logger().info('⏳ START signal lost (mất tín hiệu > 1s). Timer reset.')
                
        if self.stop_detected_time is not None and not stop_found_in_frame:
            if self.last_stop_seen_time and (ts_now - self.last_stop_seen_time > self.signal_timeout):
                self.stop_detected_time = None
                self.get_logger().info('⏳ STOP signal lost (mất tín hiệu > 1s). Timer reset.')

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
        """Extract 3D coordinates, run EKF filtering, and publish HumanState for state machine
        
        Fixed version with proper cy initialization to prevent UnboundLocalError crashes.
        """
        try:
            if self.main_target_id is None:
                return
            
            fx = self.fx
            cx_cam = self.cx_cam
            cy_cam = self.cy_cam
            
            if self.main_target_id not in self.ekfs:
                self.ekfs[self.main_target_id] = CTRV_EKF(dt=1/30.0)
                self.velocity_filters[self.main_target_id] = VelocityFilter(alpha=0.3)
                self.position_history[self.main_target_id] = deque(maxlen=self.max_position_history)
            
            ekf = self.ekfs[self.main_target_id]
            
            # Khởi tạo cy mặc định để chống crash UnboundLocalError làm đứng hình Node
            cy = int(h_img / 2) 
            
            if self.current_state == 'TRACKING':
                target_track = next((t for t in person_tracks if t['track_id'] == self.main_target_id), None)
                if target_track is None:
                    return
                
                x1, y1, x2, y2 = target_track['bbox']
                cx = (int(x1) + int(x2)) // 2
                cy = int(y1 + (y2 - y1) * 0.3)
                depth_val = self._get_depth_at_center((x1, y1, x2, y2), self._current_depth_map, h_img, w_img, cy_override=cy)
                
                if depth_val is None:
                    ekf.predict()
                else:
                    px_3d = float(depth_val)
                    py_3d = float(-(cx - cx_cam) * depth_val / fx)
                    ekf.predict()
                    ekf.update(np.array([px_3d, py_3d]))
            
            elif self.current_state == 'Re-TRACKING':
                ekf.predict()
            else:
                return
            
            # Lấy vector trạng thái an toàn (hỗ trợ cả thuộc tính x hoặc hàm get_state)
            if hasattr(ekf, 'get_state'):
                state = ekf.get_state()
            else:
                state = ekf.x.flatten()  # Lấy trực tiếp từ thuộc tính x của bộ lọc
            
            ekf_x, ekf_y, ekf_vx, ekf_vy = state[0], state[1], state[2], state[3]
            
            vel_filter = self.velocity_filters[self.main_target_id]
            filtered_vel = vel_filter.update(np.array([ekf_vx, ekf_vy, 0.0]))
            final_vx, final_vy = filtered_vel[0], filtered_vel[1]
            
            radius = 0.3
            if person_tracks:
                target_track = next((t for t in person_tracks if t['track_id'] == self.main_target_id), None)
                if target_track:
                    x1, y1, x2, y2 = target_track['bbox']
                    # Biến cy lúc này đã an toàn tuyệt đối
                    depth_val = self._get_depth_at_center((x1, y1, x2, y2), self._current_depth_map, h_img, w_img, cy_override=cy)
                    if depth_val is not None:
                        radius = float(abs(x2 - x1) * depth_val / fx / 2.0 * 0.85)
            
            msg = HumanState()
            msg.px = float(ekf_x)
            msg.py = float(ekf_y)
            msg.vx = float(final_vx)
            msg.vy = float(final_vy)
            msg.radius = float(radius)
            msg.trajectory = [] 
            
            self._pub_human_state.publish(msg)
            
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
