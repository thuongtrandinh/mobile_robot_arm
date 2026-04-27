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
import torch
from ultralytics import YOLO
import message_filters
from builtin_interfaces.msg import Time as RosTime

from visualization_msgs.msg import Marker, MarkerArray

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import PoseStamped, Point
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

        self.human_pub = self.create_publisher(HumanArray, '/tracking/humans', 10)
        # Publisher cho RViz Debug
        self.marker_pub = self.create_publisher(MarkerArray, 'human_markers', 10)

        # ===== LẤY GIÁ TRỊ PARAMETERS TỨ FILE YAML =====
        # Helper function to get parameter with default value
        def get_param(name, default):
            param = self.get_parameter(name)
            return param.value if param else default

        def as_bool(value):
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on')
            return bool(value)
        
        self.obj_conf = float(get_param('yolo.object_conf_thresh', 0.65))
        self.iou_thresh = float(get_param('yolo.iou_thresh', 0.50))
        self.depth_nms_threshold = float(get_param('yolo.depth_nms_threshold', 0.5))
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

        self.rgb_topic = get_param('topics.rgb', '/camera/color/image_raw')
        self.depth_topic = get_param('topics.depth', '/camera/aligned_depth_to_color/image_raw')
        self.camera_info_topic = get_param('topics.camera_info', '/camera/color/camera_info')
        self.camera_frame = get_param('camera.frame_id', 'camera_link')
        self.marker_lifetime_sec = float(get_param('visualization.marker_lifetime_sec', 0.5))
        self.use_cuda = as_bool(get_param('use_cuda', True))
        self.use_fp16 = as_bool(get_param('use_fp16', True))
        
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
        cuda_available = torch.cuda.is_available()
        if self.use_cuda and not cuda_available:
            self.get_logger().warn('CUDA requested but unavailable. Falling back to CPU inference.')
        self.inference_device = 'cuda' if self.use_cuda and cuda_available else 'cpu'
        if self.inference_device == 'cuda':
            self._yolo.to('cuda')
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        self._yolo.predict(dummy_img, device=self.inference_device, verbose=False, half=self.use_fp16 and self.use_cuda)
        self.get_logger().info(f'✅ YOLO đã khởi tạo trên {self.inference_device}.')

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
        
        # Publisher cho debug - Sử dụng CompressedImage để nhẹ nhất có thể
        self._pub_debug_img = self.create_publisher(
            CompressedImage, 
            '/tracking/debug_image/compressed', 
            qos_profile=qos_pub
        )
        self.get_logger().info('🛠️ Debug mode sẵn sàng (chỉ hoạt động khi có người xem).')

        self.img_sub = message_filters.Subscriber(self, Image, self.rgb_topic, qos_profile=qos_sub)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=qos_sub)
        
        # Synchronizer với parameters từ config
        self.ts = message_filters.ApproximateTimeSynchronizer([self.img_sub, self.depth_sub], queue_size=sync_queue_size, slop=sync_slop)
        self.ts.registerCallback(self._synced_callback)
        
        self.create_subscription(CameraInfo, self.camera_info_topic, self._cam_cb, 10)
        self.get_logger().info(
            f'📷 Camera topics: rgb={self.rgb_topic}, depth={self.depth_topic}, info={self.camera_info_topic}'
        )

        self._frame_intervals = deque(maxlen=30)
        self._last_cb_time = None
        self.create_timer(3.0, self._log_performance)


    def _cam_cb(self, msg):
        if not self.camera_info_received:
            self.fx, self.fy, self.cx, self.cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
            self.camera_info_received = True

    def _image_msg_to_numpy(self, msg):
        encoding = msg.encoding.lower()
        encoding_info = {
            'rgb8': (np.uint8, 3),
            'bgr8': (np.uint8, 3),
            'rgba8': (np.uint8, 4),
            'bgra8': (np.uint8, 4),
            'mono8': (np.uint8, 1),
            '8uc1': (np.uint8, 1),
            '16uc1': (np.uint16, 1),
            'mono16': (np.uint16, 1),
            '32fc1': (np.float32, 1),
        }
        if encoding not in encoding_info:
            raise ValueError(f'Unsupported image encoding: {msg.encoding}')

        dtype, channels = encoding_info[encoding]
        data = np.frombuffer(msg.data, dtype=dtype)
        row_items = msg.step // np.dtype(dtype).itemsize
        if channels == 1:
            image = data.reshape((msg.height, row_items))[:, :msg.width]
        else:
            image = data.reshape((msg.height, row_items // channels, channels))[:, :msg.width, :]
        return np.ascontiguousarray(image)

    def _color_msg_to_bgr(self, msg):
        image = self._image_msg_to_numpy(msg)
        encoding = msg.encoding.lower()
        if encoding == 'rgb8':
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding == 'rgba8':
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if encoding == 'bgra8':
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if encoding in ('mono8', '8uc1'):
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    def _depth_msg_to_meters(self, msg):
        depth = self._image_msg_to_numpy(msg)
        encoding = msg.encoding.lower()
        if encoding in ('16uc1', 'mono16'):
            return depth.astype(np.float32) / 1000.0
        if encoding == '32fc1':
            return depth.astype(np.float32)
        raise ValueError(f'Unsupported depth encoding: {msg.encoding}')

    def _synced_callback(self, img_msg, depth_msg):
        if not self.camera_info_received: return
        ts_now = time.time()
        
        # FIX: Đo khoảng thời gian THỰC TẾ giữa các lần Camera gửi ảnh đến
        if self._last_cb_time is not None:
            self._frame_intervals.append(ts_now - self._last_cb_time)
        self._last_cb_time = ts_now

        try:
            frame = self._color_msg_to_bgr(img_msg)
            if frame is None or frame.size == 0: return

            depth_map = self._depth_msg_to_meters(depth_msg)
            if depth_map is not None and depth_map.size > 0:
                self._current_depth_map = depth_map
            
            self._process_frame(frame, ts_now)
        except Exception as e:
            self.get_logger().error(f'Sync error: {e}', throttle_duration_sec=2.0)

    def _process_frame(self, frame, ts_recv):
        ts_now = self.get_clock().now().nanoseconds * 1e-9

        # 1. Dự đoán YOLO với ngưỡng thấp để BYTE Association bắt được các phần cơ thể mờ
        results = self._yolo.predict(
            frame,
            conf=self.track_low_thresh,
            verbose=False,
            device=self.inference_device,
            half=self.use_fp16 and self.use_cuda,
        )[0]
            
        raw_dets = []
        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = box.cpu().numpy()
            if int(cls) in self.dynamic_classes:
                raw_dets.append({'box': (int(x1), int(y1), int(x2), int(y2)), 'conf': float(conf), 'cls': int(cls)})

        # 2. Lọc NMS kết hợp Depth để tránh gộp nhầm người ở các khoảng cách khác nhau
        detections = self._depth_nms.apply(raw_dets, self._current_depth_map, self.obj_conf) \
                     if (self._current_depth_map is not None and raw_dets) else raw_dets

        if detections:
            dets_np = np.array([[*d['box'], d['conf']] for d in detections], dtype=np.float32)
        else:
            dets_np = np.empty((0, 5), dtype=np.float32)

        # 3. Cập nhật Tracker (BoT-SORT/ByteTrack)
        tracks_np = self._tracker.update(dets_np, features=None)

        # 4. KHỞI TẠO TIN NHẮN VÀ MARKER
        msg = HumanArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.camera_frame
        marker_array = MarkerArray()

        h, w = frame.shape[:2]
        
        for track in tracks_np:
            x1, y1, x2, y2, tid = track
            tid = int(tid)
            
            depth = self._get_depth((x1, y1, x2, y2), h, w)
            if depth is None:
                continue

            u = (x1 + x2) / 2.0
            cx_cam = w / 2.0
            px = float(depth)
            py = float(-(u - cx_cam) * depth / self.fx)

            v_center = (y1 + y2) / 2.0
            cy_cam = h / 2.0
            pz = float(-(v_center - cy_cam) * depth / self.fy)
            human_height = float(abs(y2 - y1) * depth / self.fy)

            vx, vy = self._update_ekf_velocity(tid, px, py, ts_now)
            
            human_msg = HumanState()
            human_msg.id = tid
            human_msg.px = px
            human_msg.py = py
            human_msg.vx = vx
            human_msg.vy = vy
            human_msg.radius = float(abs(x2 - x1) * depth / self.fx / 2.0)
            
            msg.humans.append(human_msg)
            h_markers = self.create_human_marker(human_msg, tid, pz, human_height)
            marker_array.markers.extend(h_markers)

        # 5. PUBLISH DỮ LIỆU ĐIỀU KHIỂN
        self.human_pub.publish(msg)
        
        # 6. HIỂN THỊ DEBUG TRÊN RVIZ2 (MarkerArray)
        if marker_array.markers:
            self.marker_pub.publish(marker_array)

        # 7. DỌN DẸP BỘ NHỚ VÀ DEBUG IMAGE
        self._publish_debug_visual(frame, tracks_np)
        self._cleanup_memory(ts_now)

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
    
    def create_human_marker(self, human, human_id, pz, human_height):
        markers = []
        # Stamp 0 tells RViz to use the latest available TF. This avoids marker
        # flicker when Gazebo camera messages are a few milliseconds ahead of TF.
        marker_stamp = RosTime()
        marker_lifetime = rclpy.duration.Duration(seconds=self.marker_lifetime_sec).to_msg()
        
        # 1. Khối trụ đỏ (Vùng vật cản)
        obs = Marker()
        obs.header.frame_id = self.camera_frame
        obs.header.stamp = marker_stamp
        obs.ns = "danger_zones"
        obs.id = human_id
        obs.type = Marker.CYLINDER
        obs.action = Marker.ADD
        obs.pose.position.x = human.px
        obs.pose.position.y = human.py
        obs.pose.position.z = pz
        
        # Scale dựa trên bán kính né tránh và chiều cao thực tế
        obs.scale.x = human.radius * 2.0 
        obs.scale.y = human.radius * 2.0
        obs.scale.z = max(0.5, human_height)
        
        obs.color.r = 1.0; obs.color.g = 0.1; obs.color.b = 0.1; obs.color.a = 0.5
        obs.lifetime = marker_lifetime
        markers.append(obs)

        # 2. Mũi tên vận tốc (Trend di chuyển)
        vel_mag = math.hypot(human.vx, human.vy)
        if vel_mag > 0.05:
            arrow = Marker()
            arrow.header = obs.header
            arrow.ns = "velocity_vectors"
            arrow.id = human_id + 1000
            arrow.type = Marker.ARROW
            z_arrow = pz - (human_height / 2.0) + 0.1
            arrow.points = [
                Point(x=human.px, y=human.py, z=z_arrow),
                Point(x=human.px + human.vx * 0.5, y=human.py + human.vy * 0.5, z=z_arrow)
            ]
            arrow.scale.x = 0.05; arrow.scale.y = 0.1; arrow.scale.z = 0.1
            arrow.color.r = 1.0; arrow.color.g = 1.0; arrow.color.b = 0.0; arrow.color.a = 1.0
            arrow.lifetime = marker_lifetime
            markers.append(arrow)

        return markers

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
        # 1. Lấy danh sách tất cả các ID ĐANG TỒN TẠI trong thuật toán BoT-SORT
        active_tids = [track.track_id for track in self._tracker.tracks]
        
        # 2. Quét mảng EKF của bạn. Nếu có ID nào nằm trong EKF mà BoT-SORT đã xóa, 
        # thì đưa vào danh sách cần dọn dẹp (expired_tids)
        expired_tids = [tid for tid in self.ekfs.keys() if tid not in active_tids]

        # 3. Dọn dẹp sạch sẽ mọi từ điển (dictionary)
        for tid in expired_tids:
            for d in [self.ekfs, self.velocity_filters, self.position_history, self.last_frame_ts, self.track_last_seen_time]:
                # Sử dụng hàm pop an toàn, nếu không có key thì trả về None thay vì báo lỗi
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
