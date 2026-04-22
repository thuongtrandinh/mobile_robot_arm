#!/usr/bin/env python3
"""
ROS2 Dynamic Object Tracking Node (ULTRA-MINIMALIST / 30Hz LOCKED)
- Đã khắc phục lỗi đồng bộ TimeSynchronizer.
- Sửa lại bộ đếm FPS để phản ánh đúng tốc độ nhận ảnh thực tế từ Camera.
"""

import rclpy
import os
import cv2
import numpy as np
import time
import math
from collections import deque

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ultralytics import YOLO
from cv_bridge import CvBridge
import message_filters

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from interfaces.msg import HumanState, HumanArray
from ament_index_python.packages import get_package_share_directory

from yolo.algorithms.botsort_handler import BoTSortTracker
from yolo.algorithms.ctrv_ekf import CTRV_EKF
from yolo.algorithms.velocity_filter import VelocityFilter

try:
    from yolo.algorithms.depth_aware_nms import DepthAwareNMS
    HAS_DEPTH_NMS = True
except ImportError:
    HAS_DEPTH_NMS = False

class MultiObjectTrackingNode(Node):
    def __init__(self):
        # Enable automatic parameter declaration from YAML config
        super().__init__('multi_object_tracking_node',
                         allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)

        # ===== LẤY GIÁ TRỊ PARAMETERS TỨ FILE YAML =====
        # Helper function to get parameter with default value
        def get_param(name, default):
            param = self.get_parameter(name)
            return param.value if param else default
        
        self.obj_conf = get_param('yolo.object_conf_thresh', 0.65)
        self.iou_thresh = get_param('yolo.iou_thresh', 0.50)
        self.depth_nms_threshold = get_param('yolo.depth_nms_threshold', 0.5)
        self.dynamic_classes = set(get_param('yolo.dynamic_classes', [0]))
        
        self.track_cleanup_timeout = get_param('tracking.tracking_timeout', 2.0)
        track_buffer = get_param('tracking.track_buffer', 60)
        track_thresh = get_param('tracking.track_thresh', 0.40)
        self.track_low_thresh = get_param('tracking.track_low_thresh', 0.10)
        match_thresh = get_param('tracking.match_thresh', 0.85)
        use_appearance = get_param('tracking.use_appearance', False)
        appearance_weight = get_param('tracking.appearance_weight', 0.5)
        max_age_before_predict = get_param('tracking.max_age_before_predict', 15)
        
        self.min_depth = get_param('depth_filter.min_depth_m', 0.3)
        self.max_depth = get_param('depth_filter.max_depth_m', 6.0)
        
        self.velocity_filter_alpha = get_param('velocity_filter.alpha', 0.25)
        
        pub_depth = get_param('ros2_qos.publisher_depth', 1)
        sub_depth = get_param('ros2_qos.subscriber_depth', 30)
        sync_slop = get_param('ros2_qos.sync_slop', 0.1)
        sync_queue_size = get_param('ros2_qos.sync_queue_size', 30)
        
        self.fx_param = get_param('camera.fx', 609.268)
        self.fy_param = get_param('camera.fy', 609.338)
        self.cx_param = get_param('camera.cx', 323.599)
        self.cy_param = get_param('camera.cy', 246.130)
        
        self.get_logger().info(
            f'⚙️ Parameters loaded: conf={self.obj_conf}, classes={self.dynamic_classes}, '
            f'timeout={self.track_cleanup_timeout}s, depth_range=[{self.min_depth:.1f}, {self.max_depth:.1f}]m'
        )

        pkg_share = get_package_share_directory('yolo')
        model_path = os.path.join(pkg_share, 'models', self.get_parameter('model_path').value)
        
        self._yolo = YOLO(model_path)
        self._yolo.to('cuda')
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        self._yolo.predict(dummy_img, device='cuda', verbose=False, half=True)
        self.get_logger().info('✅ YOLOv8n & GPU đã khởi tạo.')

        self._tracker = BoTSortTracker(
            track_thresh=track_thresh, 
            track_buffer=track_buffer, 
            match_thresh=match_thresh,
            use_appearance=use_appearance,
            track_low_thresh=self.track_low_thresh,
            max_age_before_predict=max_age_before_predict
        )
        
        if HAS_DEPTH_NMS:
            self._depth_nms = DepthAwareNMS(nms_threshold=self.depth_nms_threshold, depth_threshold=self.min_depth)

        self.ekfs, self.velocity_filters = {}, {}
        self.track_last_seen_time, self.last_frame_ts, self.position_history = {}, {}, {}
        
        self._bridge = CvBridge()
        self._current_depth_map = None
        
        # Inicializar intrínsecos da câmera com valores padrão (serão sobrescrevidos por camera_info)
        self.fx = self.fx_param
        self.fy = self.fy_param
        self.cx = self.cx_param
        self.cy = self.cy_param
        self.camera_info_received = False
        
        # QoS profiles from config
        qos_pub = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=pub_depth)
        qos_sub = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=sub_depth)
        
        self._pub_humans = self.create_publisher(HumanArray, '/tracking/humans', qos_profile=qos_pub)
        
        # Publisher cho debug - Sử dụng CompressedImage để nhẹ nhất có thể
        self._pub_debug_img = self.create_publisher(
            CompressedImage, 
            '/tracking/debug_image/compressed', 
            qos_profile=qos_pub
        )
        self.get_logger().info('🛠️ Debug mode sẵn sàng (chỉ hoạt động khi có người xem).')

        self.img_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw', qos_profile=qos_sub)
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw', qos_profile=qos_sub)
        
        # Synchronizer với parameters từ config
        self.ts = message_filters.ApproximateTimeSynchronizer([self.img_sub, self.depth_sub], queue_size=sync_queue_size, slop=sync_slop)
        self.ts.registerCallback(self._synced_callback)
        
        self.create_subscription(CameraInfo, '/camera/color/camera_info', self._cam_cb, 10)

        self._frame_intervals = deque(maxlen=30)
        self._last_cb_time = None
        self.create_timer(3.0, self._log_performance)

    def _cam_cb(self, msg):
        if not self.camera_info_received:
            self.fx, self.fy, self.cx, self.cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
            self.camera_info_received = True

    def _synced_callback(self, img_msg, depth_msg):
        if not self.camera_info_received: return
        ts_now = time.time()
        
        # FIX: Đo khoảng thời gian THỰC TẾ giữa các lần Camera gửi ảnh đến
        if self._last_cb_time is not None:
            self._frame_intervals.append(ts_now - self._last_cb_time)
        self._last_cb_time = ts_now

        try:
            frame_raw = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
            if frame_raw is None or frame_raw.size == 0: return
            frame = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR) if img_msg.encoding == 'rgb8' else frame_raw

            depth_raw = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            if depth_raw is not None and depth_raw.size > 0:
                self._current_depth_map = depth_raw.astype(np.float32) / 1000.0 if depth_msg.encoding == '16UC1' else depth_raw.astype(np.float32)
            
            self._process_frame(frame, ts_now)
        except Exception as e:
            self.get_logger().error(f'Sync error: {e}', throttle_duration_sec=2.0)

    def _process_frame(self, frame, ts_recv):
        ts_now = self.get_clock().now().nanoseconds * 1e-9

        # FIX: Use track_low_thresh for BYTE Association to catch weak arm detections
        # Ensures that outstretched limbs don't create separate ghost bboxes
        # NMS (iou_thresh=0.35) will merge them into single unified box
        results = self._yolo.predict(frame, conf=self.track_low_thresh, verbose=False, device='cuda', half=True)[0]
            
        raw_dets = []
        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box.cpu().numpy()
            if int(cls) in self.dynamic_classes:
                raw_dets.append({'box': (int(x1), int(y1), int(x2), int(y2)), 'conf': float(conf), 'cls': int(cls)})

        detections = self._depth_nms.apply(raw_dets, self._current_depth_map, self.obj_conf) if HAS_DEPTH_NMS and self._current_depth_map is not None and raw_dets else raw_dets

        if detections:
            dets_np = np.array([[*d['box'], d['conf']] for d in detections], dtype=np.float32)
        else:
            dets_np = np.empty((0, 5), dtype=np.float32)

        tracks_np = self._tracker.update(dets_np, features=None)

        h, w = frame.shape[:2]
        self._publish_debug_visual(frame, tracks_np)
        self._publish_data(tracks_np, h, w, ts_now)
        self._cleanup_memory(ts_now)

    def _publish_data(self, tracks_np, h, w, ts_now):
        # Tạo message mảng người động
        human_msg = HumanArray()
        human_msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id="camera_link")

        for i, track in enumerate(tracks_np):
            tid = int(track[4])
            bbox = track[:4]
            self.track_last_seen_time[tid] = ts_now

            depth = self._get_depth(bbox, h, w)
            if depth is None: continue

            # Tọa độ tương đối so với camera (X hướng tới, Y hướng trái)
            px = float(depth)
            py = float(-(( (bbox[0]+bbox[2])/2 - self.cx ) * depth / self.fx))
            radius = float(abs(bbox[2] - bbox[0]) * depth / self.fx / 2.0 * 0.85)

            # Cập nhật EKF và lấy thẳng vx, vy
            vx, vy = self._update_ekf_velocity(tid, px, py, ts_now)
            
            # Gói vào HumanState
            human = HumanState()
            human.px = px
            human.py = py
            human.vx = float(vx)
            human.vy = float(vy)
            human.radius = radius
            
            human_msg.humans.append(human)

        # Bắn dữ liệu sang cho Controller
        self._pub_humans.publish(human_msg)

    def _get_depth(self, bbox, h, w):
        x1, y1, x2, y2 = map(int, bbox)
        if not (0 <= x1 < w and x2 <= w and 0 <= y1 < h and y2 <= h): return None
        
        # Primary ROI: focus on torso center only
        margin_w = int((x2 - x1) * 0.35)
        margin_h = int((y2 - y1) * 0.15)
        roi = self._current_depth_map[y1+margin_h:y2-margin_h, x1+margin_w:x2-margin_w]
        
        valid = roi[(np.isfinite(roi)) & (roi > self.min_depth) & (roi < self.max_depth)]
        if valid.size > 0:
            return float(np.median(valid))
        
        # FIX: FALLBACK MECHANISM for close-range tracking (< 50cm)
        # If center region is empty (standing too close), use entire BBox
        # This prevents humans: [] flicker when person is at minimum distance
        roi_fallback = self._current_depth_map[y1:y2, x1:x2]
        valid_fallback = roi_fallback[(np.isfinite(roi_fallback)) & (roi_fallback > self.min_depth) & (roi_fallback < self.max_depth)]
        
        return float(np.median(valid_fallback)) if valid_fallback.size > 0 else None

    def _update_ekf_velocity(self, tid, px, py, ts_now):
        if tid not in self.ekfs:
            self.ekfs[tid] = CTRV_EKF(dt=0.033)
            self.velocity_filters[tid] = VelocityFilter(alpha=self.velocity_filter_alpha)
            # FIX: Increase buffer from 5 to 15 frames (~0.5s)
            # Longer derivative window = cleaner velocity estimates, noise cancellation
            self.position_history[tid] = deque(maxlen=15)

        dt = max(ts_now - self.last_frame_ts.get(tid, ts_now - 0.033), 0.01)
        self.last_frame_ts[tid] = ts_now
        
        ekf = self.ekfs[tid]
        ekf.dt = dt
        ekf.predict()
        
        # Store position with timestamp for real-time velocity calculation
        self.position_history[tid].append((np.array([px, py]), ts_now))
        
        # FIX: Calculate velocity using ACTUAL elapsed time, not frame count
        # This handles variable camera framerate and eliminates timestamp jitter
        if len(self.position_history[tid]) >= 2:
            pos_old, ts_old = self.position_history[tid][0]
            pos_new, ts_new = self.position_history[tid][-1]
            # FIX: Increase min delta_t from 0.01 to 0.033 to prevent velocity explosion
            # Dividing by very small time values causes velocity to spike unrealistically
            # 0.033s = 1 frame at 30Hz, preventing division by tiny numbers
            delta_t = max(ts_new - ts_old, 0.033)  # Actual time elapsed
            meas_vel = (pos_new - pos_old) / delta_t
        else:
            meas_vel = np.zeros(2)
            
        ekf.update(np.array([px, py]), meas_vel)

        filtered_vel = self.velocity_filters[tid].update(np.array([
            float(ekf.state.v * np.cos(ekf.state.psi)), 
            float(ekf.state.v * np.sin(ekf.state.psi)), 
            0.0
        ]))
        
        vx, vy = filtered_vel[0], filtered_vel[1]
        
        # FIX: Eliminate subnormal float garbage (2.8e-45 is subnormal zero)
        # Check if values are near zero and clamp to exactly 0.0
        if abs(vx) < 1e-4: vx = 0.0
        if abs(vy) < 1e-4: vy = 0.0
        
        return float(vx), float(vy)

    def _cleanup_memory(self, ts_now):
        stale_ids = [tid for tid, last in self.track_last_seen_time.items() if ts_now - last > self.track_cleanup_timeout]
        for tid in stale_ids:
            for d in [self.ekfs, self.velocity_filters, self.position_history, self.last_frame_ts, self.track_last_seen_time]:
                d.pop(tid, None)

    def _publish_debug_visual(self, frame, tracks_np):
        # KIỂM TRA: Nếu không có ai đang xem topic này thì THOÁT NGAY để tiết kiệm CPU
        if self._pub_debug_img.get_subscription_count() == 0:
            return

        try:
            # Resize nhỏ lại để nén và gửi nhanh hơn (giảm 4 lần tải)
            debug_frame = cv2.resize(frame, (480, 360)) 
            scale_x = 480 / frame.shape[1]
            scale_y = 360 / frame.shape[0]

            for track in tracks_np:
                x1, y1, x2, y2, tid = track[:5]
                # Vẽ bounding box (đã scale theo ảnh nhỏ)
                cv2.rectangle(debug_frame, 
                              (int(x1*scale_x), int(y1*scale_y)), 
                              (int(x2*scale_x), int(y2*scale_y)), (0, 255, 0), 2)
                cv2.putText(debug_frame, f"ID:{int(tid)}", 
                            (int(x1*scale_x), int(y1*scale_y)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Nén trực tiếp thành JPEG
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = "jpeg"
            # Quality=50 là đủ nhìn debug mà cực nhẹ
            msg.data = np.array(cv2.imencode('.jpg', debug_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])[1]).tobytes()
            
            self._pub_debug_img.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Debug image error: {e}")

    def _log_performance(self):
        if self._frame_intervals:
            real_fps = 1.0 / np.mean(self._frame_intervals)
            self.get_logger().info(f'⚡ Tần số nhận ảnh thực tế: {real_fps:.1f} Hz | Đang theo dõi: {len(self.ekfs)} vật cản')

def main(args=None):
    rclpy.init(args=args)
    node = MultiObjectTrackingNode()
    try:
        # Bật lại MultiThreadedExecutor để tăng tốc độ nhận callback
        from rclpy.executors import MultiThreadedExecutor
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()

