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

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from interfaces.msg import ObstacleArray, DynaObstacle
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
        super().__init__('multi_object_tracking_node')

        self.declare_parameter('model_path', 'yolov8n.pt')
        self.obj_conf = 0.50
        
        self.dynamic_classes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}

        pkg_share = get_package_share_directory('yolo')
        model_path = os.path.join(pkg_share, 'models', self.get_parameter('model_path').value)
        
        self._yolo = YOLO(model_path)
        self._yolo.to('cuda')
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        self._yolo.predict(dummy_img, device='cuda', verbose=False, half=True)
        self.get_logger().info('✅ YOLOv8n & GPU đã khởi tạo.')

        self._tracker = BoTSortTracker(
            track_thresh=0.40, 
            track_buffer=60, 
            match_thresh=0.8,
            use_appearance=False, 
            max_age_before_predict=15
        )
        
        if HAS_DEPTH_NMS:
            self._depth_nms = DepthAwareNMS(nms_threshold=0.65, depth_threshold=0.50)

        self.ekfs, self.velocity_filters = {}, {}
        self.track_last_seen_time, self.last_frame_ts, self.position_history = {}, {}, {}
        self.track_cleanup_timeout = 2.0

        self._bridge = CvBridge()
        
        qos_pub = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        # Tăng depth=30 để không bỏ lọt gói tin mạng nào từ camera
        qos_sub = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=30)
        
        self._pub_dynamics = self.create_publisher(ObstacleArray, '/tracking/dynamics', qos_profile=qos_pub)

        self.img_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw', qos_profile=qos_sub)
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw', qos_profile=qos_sub)
        
        # FIX CỰC QUAN TRỌNG: Tăng queue_size và slop để bao dung hơn với độ trễ của D435i
        self.ts = message_filters.ApproximateTimeSynchronizer([self.img_sub, self.depth_sub], queue_size=30, slop=0.1)
        self.ts.registerCallback(self._synced_callback)
        
        self.camera_info_received = False
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

        results = self._yolo.predict(frame, conf=self.obj_conf, verbose=False, device='cuda', half=True)[0]
            
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
        self._publish_data(tracks_np, h, w, ts_now)
        self._cleanup_memory(ts_now)

    def _publish_data(self, tracks_np, h, w, ts_now):
        dyna_msg = ObstacleArray(header=Header(stamp=self.get_clock().now().to_msg(), frame_id="camera_link"))

        for i, track in enumerate(tracks_np):
            tid = int(track[4])
            bbox = track[:4]
            self.track_last_seen_time[tid] = ts_now

            depth = self._get_depth(bbox, h, w)
            if depth is None: continue

            px = float(depth)
            py = float(-(( (bbox[0]+bbox[2])/2 - self.cx ) * depth / self.fx))
            pz = float(-(( (bbox[1]+bbox[3])/2 - self.cy ) * depth / self.fy))
            radius = float(abs(bbox[2] - bbox[0]) * depth / self.fx / 2.0 * 0.85)

            trajectory = self._update_ekf_and_traj(tid, px, py, pz, ts_now, dyna_msg.header)
            dyna_msg.dyna_obstacles.append(DynaObstacle(id=float(tid), distance=math.hypot(px, py), radius=radius, trajectory=trajectory))

        self._pub_dynamics.publish(dyna_msg)

    def _get_depth(self, bbox, h, w):
        x1, y1, x2, y2 = map(int, bbox)
        if not (0 <= x1 < w and x2 <= w and 0 <= y1 < h and y2 <= h): return None
        roi = self._current_depth_map[y1:y2, x1:x2]
        valid = roi[(np.isfinite(roi)) & (roi > 0.3) & (roi < 8.0)]
        return float(np.median(valid)) if valid.size > 0 else None

    def _update_ekf_and_traj(self, tid, px, py, pz, ts_now, header):
        if tid not in self.ekfs:
            self.ekfs[tid] = CTRV_EKF(dt=0.033)
            self.velocity_filters[tid] = VelocityFilter(alpha=0.3)
            self.position_history[tid] = deque(maxlen=5)

        dt = max(ts_now - self.last_frame_ts.get(tid, ts_now - 0.033), 0.01)
        self.last_frame_ts[tid] = ts_now
        
        ekf = self.ekfs[tid]
        ekf.dt = dt
        ekf.predict()
        
        self.position_history[tid].append(np.array([px, py]))
        meas_vel = (self.position_history[tid][-1] - self.position_history[tid][-2])/dt if len(self.position_history[tid]) >= 2 else np.zeros(2)
        ekf.update(np.array([px, py]), meas_vel)

        filtered_vel = self.velocity_filters[tid].update(np.array([float(ekf.state.v * np.cos(ekf.state.psi)), float(ekf.state.v * np.sin(ekf.state.psi)), 0.0]))
        vx, vy = filtered_vel[0], filtered_vel[1]
        
        path = Path(header=header)
        for i in range(6): 
            pose = PoseStamped(header=header)
            pose.pose.position.x = float(px + vx*(i*0.25))
            pose.pose.position.y = float(py + vy*(i*0.25))
            pose.pose.position.z = float(pz)
            path.poses.append(pose)
        return path

    def _cleanup_memory(self, ts_now):
        stale_ids = [tid for tid, last in self.track_last_seen_time.items() if ts_now - last > self.track_cleanup_timeout]
        for tid in stale_ids:
            for d in [self.ekfs, self.velocity_filters, self.position_history, self.last_frame_ts, self.track_last_seen_time]:
                d.pop(tid, None)

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

