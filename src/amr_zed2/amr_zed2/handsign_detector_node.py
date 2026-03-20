"""
AMR ZED2 Hand Sign Detection and Person Tracking ROS2 Node
Optimized for real-time planner integration
"""

import cv2
import pyzed.sl as sl
from ultralytics import YOLO
import numpy as np
import time
import os
from typing import Tuple, List, Dict, Optional
from collections import deque
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Bool, Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

from amr_interfaces.msg import HumanState, HumanStateArray, ObstacleState, ObstacleStateArray

os.environ['ZED_LOG_LEVEL'] = 'ERROR'

from .depth_aware_nms import DepthAwareNMS
from .bytetrack_handler import ByteTracker, Track
from .zed2_velocity_optimizer import ZED2VelocityEstimator


def resolve_model_path(model_path: str, package_name: str = 'amr_zed2') -> str:
    """Resolve model path - check absolute, then package share, then relative"""
    if os.path.isabs(model_path) and os.path.exists(model_path):
        return model_path
    try:
        pkg_share = get_package_share_directory(package_name)
        pkg_model_path = os.path.join(pkg_share, 'models', model_path)
        if os.path.exists(pkg_model_path):
            return pkg_model_path
    except Exception:
        pass
    if os.path.exists(model_path):
        return model_path
    raise FileNotFoundError(f"Model not found: {model_path}")


@dataclass(slots=True)
class TrackedPerson:
    """Tracked person state"""
    track_id: int
    bbox: Tuple[int, int, int, int]
    center_3d: Tuple[float, float, float]
    velocity_3d: Tuple[float, float, float]
    speed: float


class HandSignDetectorNode(Node):
    """ROS2 Node for Hand Sign Detection and Person Tracking"""

    __slots__ = [
        '_params', '_pubs', '_zed', '_models', '_trackers', '_state',
        '_signal_state', '_perf', '_cam_fx', '_cam_fy', '_cam_cx', '_cam_cy',
        'timer', 'bridge'
    ]

    def __init__(self):
        super().__init__('handsign_detector')
        self._init_parameters()
        self._init_publishers()
        self._init_camera()
        self._init_models()
        self._init_tracking()
        self._init_state()

        self.timer = self.create_timer(1.0 / 30.0, self._process_frame)
        self.get_logger().info('HandSign Detector initialized')

    def _init_parameters(self):
        """Initialize ROS parameters"""
        defaults = {
            'hand_model_path': 'handsign_exp14_best.pt',
            'person_model_path': 'yolov8n.pt',
            'start_hold_time': 3.0,
            'stop_hold_time': 3.0,
            'signal_timeout': 2.0,
            'show_display': True,
            'publish_image': False,
            'person_radius': 0.3,
            'detection_conf': 0.5,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        self._params = {k: self.get_parameter(k).value for k in defaults}

    def _init_publishers(self):
        """Initialize ROS publishers"""
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self._pubs = {
            'status': self.create_publisher(Bool, '/zed2/tracking/status', 10),
            'signal': self.create_publisher(String, '/zed2/tracking/hand_signal', 10),
            'humans': self.create_publisher(HumanStateArray, '/zed2/humans', qos),
            'obstacles': self.create_publisher(ObstacleStateArray, '/zed2/obstacles', qos),
        }
        if self._params['publish_image']:
            self._pubs['image'] = self.create_publisher(Image, '/zed2/tracking/image', qos)
            self.bridge = CvBridge()
        else:
            self.bridge = None

    def _init_camera(self):
        """Initialize ZED camera"""
        camera = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_units = sl.UNIT.METER

        if camera.open(init_params) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError('ZED camera initialization failed')

        cam_info = camera.get_camera_information().calibration_parameters.left_cam
        self._cam_fx, self._cam_fy = cam_info.fx, cam_info.fy
        self._cam_cx, self._cam_cy = cam_info.cx, cam_info.cy

        self._zed = {
            'camera': camera,
            'image': sl.Mat(),
            'depth': sl.Mat(),
            'runtime': sl.RuntimeParameters(),
        }
        self.get_logger().info('ZED camera initialized')

    def _init_models(self):
        """Initialize YOLO models"""
        self.get_logger().info('Loading YOLO models...')
        hand_path = resolve_model_path(self._params['hand_model_path'])
        person_path = resolve_model_path(self._params['person_model_path'])
        self.get_logger().info(f'Hand model: {hand_path}')
        self.get_logger().info(f'Person model: {person_path}')
        self._models = {
            'hand': YOLO(hand_path),
            'person': YOLO(person_path),
        }
        self.get_logger().info('Models loaded')

    def _init_tracking(self):
        """Initialize tracking components"""
        self._trackers = {
            'nms_hand': DepthAwareNMS(nms_threshold=0.45, depth_threshold=0.5),
            'nms_person': DepthAwareNMS(nms_threshold=0.45, depth_threshold=0.4),
            'person': ByteTracker(max_age=30, min_hits=3, iou_threshold=0.5),
            'hand': ByteTracker(max_age=20, min_hits=2, iou_threshold=0.4,
                                track_high_thresh=0.5, new_track_thresh=0.4),
            'velocity': ZED2VelocityEstimator(fps=30.0),
            'vel_history': {},
            'vel_estimates': {},
        }

    def _init_state(self):
        """Initialize tracking state"""
        self._state = {'active': False, 'target': None, 'loss_count': 0}
        self._signal_state = {
            'start_detected': False, 'start_time': None, 'start_hold': 0.0, 'last_start_seen': None,
            'stop_time': None, 'stop_hold': 0.0, 'last_stop_seen': None,
        }
        self._perf = {'frame_count': 0, 'fps_time': time.time(), 'fps': 30.0}

    def _get_depth_at_box(self, box: Tuple[int, int, int, int], depth_map: np.ndarray) -> float:
        """Get depth at bounding box center with fallback to median"""
        x1, y1, x2, y2 = box
        h, w = depth_map.shape[:2]
        cx = min(max((x1 + x2) >> 1, 0), w - 1)
        cy = min(max((y1 + y2) >> 1, 0), h - 1)

        depth = depth_map[cy, cx]
        if depth > 0:
            return float(depth)

        # Fallback: median of valid depth in ROI
        roi = depth_map[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        valid = roi[roi > 0]
        return float(np.median(valid)) if valid.size > 0 else 0.0

    def _pixel_to_3d(self, cx: int, cy: int, depth: float) -> Tuple[float, float, float]:
        """Convert pixel coordinates to 3D camera frame"""
        if depth <= 0:
            return 0.0, 0.0, 0.0
        X = depth * (cx - self._cam_cx) / self._cam_fx
        Y = depth * (cy - self._cam_cy) / self._cam_fy
        return float(X), float(Y), float(depth)

    def _detect_and_track(self, frame: np.ndarray, depth_map: np.ndarray):
        """Run detection and tracking pipeline"""
        conf = self._params['detection_conf']

        # YOLO inference
        hand_res = self._models['hand'](frame, conf=conf, verbose=False)
        person_res = self._models['person'](frame, conf=conf, classes=0, verbose=False)

        # Process detections
        hand_dets = self._process_detections(hand_res, depth_map, self._trackers['nms_hand'])
        person_dets = self._process_detections(person_res, depth_map, self._trackers['nms_person'])

        # Update trackers
        fps = self._perf['fps']
        self._trackers['person'].set_fps(fps)
        self._trackers['hand'].set_fps(fps)

        person_tracks = self._trackers['person'].update(person_dets)
        hand_tracks = self._trackers['hand'].update(hand_dets)

        # Update velocities and cleanup
        self._update_velocities(person_tracks)

        return hand_tracks, person_tracks

    def _process_detections(self, results, depth_map: np.ndarray, nms: DepthAwareNMS) -> List[Dict]:
        """Process YOLO results to tracking-ready detections"""
        if results[0].boxes is None:
            return []

        detections = []
        for box in results[0].boxes:
            bbox = tuple(int(i) for i in box.xyxy[0])
            depth = self._get_depth_at_box(bbox, depth_map)
            cx, cy = (bbox[0] + bbox[2]) >> 1, (bbox[1] + bbox[3]) >> 1

            detections.append({
                'box': bbox,
                'conf': float(box.conf[0]),
                'cls': int(box.cls[0]),
                'label': results[0].names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'class_id': int(box.cls[0]),
                'center_2d': (cx, cy),
                'center_3d': self._pixel_to_3d(cx, cy, depth),
                'mean_depth': depth,
                'variance': 0.0,
                'median_depth': depth,
            })

        # Apply NMS
        filtered = nms.apply(detections, depth_map, self._params['detection_conf'])
        return filtered

    def _update_velocities(self, tracks: List[Track]):
        """Update velocity estimates and cleanup dead tracks"""
        vel_hist = self._trackers['vel_history']
        vel_est = self._trackers['vel_estimates']
        estimator = self._trackers['velocity']
        active_ids = {t.track_id for t in tracks}

        # Cleanup dead track histories
        dead_ids = [tid for tid in vel_hist if tid not in active_ids]
        for tid in dead_ids:
            del vel_hist[tid]
            vel_est.pop(tid, None)

        # Update active tracks
        for track in tracks:
            tid = track.track_id
            pos_3d = track.center_3d
            cx = (track.bbox[0] + track.bbox[2]) >> 1
            cy = (track.bbox[1] + track.bbox[3]) >> 1
            depth = pos_3d[2] if pos_3d[2] > 0 else 0.0

            if tid not in vel_hist:
                vel_hist[tid] = deque(maxlen=10)
            vel_hist[tid].append((pos_3d, (cx, cy), depth, 0.8))

            if len(vel_hist[tid]) >= 2:
                history = vel_hist[tid]
                pos_h = deque([(h[0], h[1]) for h in history])
                depth_h = deque([h[2] for h in history])
                conf_h = deque([h[3] for h in history])
                dt = (len(history) - 1) / max(self._perf['fps'], 1.0)
                raw_vel = estimator.estimate_velocity_from_history(pos_h, depth_h, conf_h, dt)
                vel_est[tid] = estimator.smooth_velocity(tid, raw_vel)

    def _process_hand_signals(self, hand_tracks: List[Track], person_tracks: List[Track], t: float):
        """Process hand sign detections"""
        ss = self._signal_state
        is_active = self._state['active']
        target = self._state['target']
        start_found = stop_found = False

        for hand in hand_tracks:
            x1, y1, x2, y2 = hand.bbox
            label = hand.label
            hcx, hcy = (x1 + x2) >> 1, (y1 + y2) >> 1

            # START signal (only when not tracking)
            if label == "start" and not is_active:
                start_found = True
                ss['last_start_seen'] = t
                if not ss['start_detected']:
                    ss['start_detected'] = True
                    ss['start_time'] = t
                    ss['start_hold'] = 0.0
                    self._publish_signal('START_DETECTED')
                else:
                    ss['start_hold'] = t - ss['start_time']
                    if ss['start_hold'] >= self._params['start_hold_time'] and person_tracks:
                        if hand.center_3d != (0.0, 0.0, 0.0):
                            if self._start_tracking(person_tracks, hand.center_3d):
                                self._publish_signal('START_CONFIRMED')

            # STOP signal (only when tracking, within target bounds)
            if is_active and target and label == "stop":
                px1, py1, px2, py2 = target.bbox
                if px1 - 30 <= hcx <= px2 + 30 and py1 - 30 <= hcy <= py2 + 30:
                    stop_found = True
                    ss['last_stop_seen'] = t
                    if ss['stop_time'] is None:
                        ss['stop_time'] = t
                        ss['stop_hold'] = 0.0
                        self._publish_signal('STOP_DETECTED')
                    else:
                        ss['stop_hold'] = t - ss['stop_time']
                        if ss['stop_hold'] >= self._params['stop_hold_time']:
                            self._reset_tracking('STOP confirmed')
                            self._publish_signal('STOP_CONFIRMED')

        # Handle timeouts
        timeout = self._params['signal_timeout']
        if ss['start_detected'] and not start_found and ss['last_start_seen']:
            if t - ss['last_start_seen'] > timeout:
                ss['start_detected'] = False
                ss['start_time'] = ss['last_start_seen'] = None
                ss['start_hold'] = 0.0

        if ss['stop_time'] and not stop_found and ss['last_stop_seen']:
            if t - ss['last_stop_seen'] > timeout:
                ss['stop_time'] = ss['last_stop_seen'] = None
                ss['stop_hold'] = 0.0

    def _start_tracking(self, person_tracks: List[Track], hand_3d: Tuple[float, float, float]) -> bool:
        """Start tracking closest person to hand signal"""
        if not person_tracks:
            return False

        hand_pos = np.array(hand_3d)
        closest, min_dist = None, 1.5  # Max 1.5m distance

        for track in person_tracks:
            if track.center_3d == (0.0, 0.0, 0.0):
                continue
            dist = float(np.linalg.norm(hand_pos - np.array(track.center_3d)))
            if dist < min_dist:
                min_dist, closest = dist, track

        if closest:
            self._state['active'] = True
            self._state['target'] = TrackedPerson(
                track_id=closest.track_id,
                bbox=closest.bbox,
                center_3d=closest.center_3d,
                velocity_3d=closest.velocity_3d,
                speed=closest.speed_3d
            )
            self._state['loss_count'] = 0
            self.get_logger().info(f'Tracking started - ID: {closest.track_id}')
            self._publish_status(True)
            return True
        return False

    def _update_tracked_person(self, person_tracks: List[Track]) -> bool:
        """Update tracked person from ByteTrack results"""
        target = self._state['target']
        if not self._state['active'] or not target:
            return False

        target_id = target.track_id
        vel_est = self._trackers['vel_estimates']

        for track in person_tracks:
            if track.track_id == target_id:
                v = vel_est.get(target_id)
                self._state['target'] = TrackedPerson(
                    track_id=target_id,
                    bbox=track.bbox,
                    center_3d=track.center_3d,
                    velocity_3d=v.velocity_3d if v else track.velocity_3d,
                    speed=v.speed_3d if v else track.speed_3d
                )
                self._state['loss_count'] = 0
                return True

        self._state['loss_count'] += 1
        if self._state['loss_count'] > 60:
            self._reset_tracking('Tracking lost')
        return False

    def _reset_tracking(self, reason: str = ""):
        """Reset tracking state"""
        self._state = {'active': False, 'target': None, 'loss_count': 0}
        self._signal_state = {
            'start_detected': False, 'start_time': None, 'start_hold': 0.0, 'last_start_seen': None,
            'stop_time': None, 'stop_hold': 0.0, 'last_stop_seen': None,
        }
        if reason:
            self.get_logger().info(reason)
        self._publish_status(False)

    def _publish_status(self, active: bool):
        msg = Bool(data=active)
        self._pubs['status'].publish(msg)

    def _publish_signal(self, signal: str):
        msg = String(data=signal)
        self._pubs['signal'].publish(msg)

    def _publish_data(self, person_tracks: List[Track]):
        """Publish humans and obstacles in single pass"""
        now = self.get_clock().now().to_msg()
        radius = self._params['person_radius']
        target = self._state['target']
        is_active = self._state['active']
        target_id = target.track_id if target else -1

        # Human message
        human_msg = HumanStateArray()
        human_msg.header.stamp = now
        human_msg.header.frame_id = 'base_link'

        if is_active and target and target.center_3d != (0.0, 0.0, 0.0):
            cam_x, _, cam_z = target.center_3d
            vx_cam, _, vz_cam = target.velocity_3d
            human = HumanState()
            human.px, human.py = float(cam_z), float(-cam_x)
            human.vx, human.vy = float(vz_cam), float(-vx_cam)
            human.radius = radius
            human_msg.humans.append(human)

        # Obstacle message
        obs_msg = ObstacleStateArray()
        obs_msg.header.stamp = now
        obs_msg.header.frame_id = 'base_link'

        for track in person_tracks:
            if is_active and track.track_id == target_id:
                continue
            if track.center_3d == (0.0, 0.0, 0.0):
                continue
            cam_x, _, cam_z = track.center_3d
            obs = ObstacleState()
            obs.px, obs.py = float(cam_z), float(-cam_x)
            obs.radius = radius
            obs_msg.obstacles.append(obs)

        self._pubs['humans'].publish(human_msg)
        self._pubs['obstacles'].publish(obs_msg)

    def _process_frame(self):
        """Main processing loop"""
        zed = self._zed
        if zed['camera'].grab(zed['runtime']) != sl.ERROR_CODE.SUCCESS:
            return

        zed['camera'].retrieve_image(zed['image'], sl.VIEW.LEFT)
        zed['camera'].retrieve_measure(zed['depth'], sl.MEASURE.DEPTH)

        frame = cv2.cvtColor(zed['image'].get_data(), cv2.COLOR_RGBA2BGR)
        depth_map = zed['depth'].get_data().astype(np.float32)
        t = time.time()

        # Detection and tracking
        hand_tracks, person_tracks = self._detect_and_track(frame, depth_map)

        # Update tracked person
        if self._state['active']:
            self._update_tracked_person(person_tracks)

        # Process signals and publish
        self._process_hand_signals(hand_tracks, person_tracks, t)
        self._publish_data(person_tracks)

        # FPS update
        self._perf['frame_count'] += 1
        if t - self._perf['fps_time'] >= 1.0:
            self._perf['fps'] = self._perf['frame_count'] / (t - self._perf['fps_time'])
            self._perf['frame_count'] = 0
            self._perf['fps_time'] = t

        # Visualization
        if self._params['show_display']:
            self._visualize(frame, hand_tracks, person_tracks)

        if self._params['publish_image'] and self.bridge:
            vis = self._draw_frame(frame.copy(), hand_tracks, person_tracks)
            self._pubs['image'].publish(self.bridge.cv2_to_imgmsg(vis, 'bgr8'))

    def _visualize(self, frame: np.ndarray, hand_tracks: List[Track], person_tracks: List[Track]):
        """Draw and show visualization"""
        vis = self._draw_frame(frame, hand_tracks, person_tracks)
        cv2.imshow('AMR ZED2', vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

    def _draw_frame(self, frame: np.ndarray, hand_tracks: List[Track], person_tracks: List[Track]) -> np.ndarray:
        """Draw visualization on frame"""
        ss = self._signal_state
        target = self._state['target']
        target_id = target.track_id if target else -1

        # Draw hands
        hand_color = (0, 255, 255) if ss['start_detected'] else (0, 255, 0)
        for hand in hand_tracks:
            x1, y1, x2, y2 = hand.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), hand_color, 2)
            cv2.putText(frame, f"#{hand.track_id} {hand.label}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)

        # Draw persons
        for track in person_tracks:
            px1, py1, px2, py2 = track.bbox
            if self._state['active'] and track.track_id == target_id:
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 3)
                cv2.putText(frame, f"TARGET #{track.track_id} {target.speed:.2f}m/s",
                            (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (px1, py1), (px2, py2), (200, 200, 200), 1)

        # Status bar
        if self._state['active'] and target:
            status = f"TRACKING #{target.track_id} | {target.speed:.2f}m/s"
            bar_color = (0, 0, 255)
        elif ss['start_detected']:
            status = f"START: {ss['start_hold']:.1f}s/{self._params['start_hold_time']:.1f}s"
            bar_color = (0, 165, 255)
        else:
            status = "WAITING"
            bar_color = (0, 255, 0)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), bar_color, -1)
        cv2.putText(frame, f"{status} | FPS:{self._perf['fps']:.0f}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame

    def destroy_node(self):
        self._zed['camera'].close()
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
