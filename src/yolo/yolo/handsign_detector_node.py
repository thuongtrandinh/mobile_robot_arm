import rclpy
import os
import math
import cv2
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration
from ultralytics import YOLO
from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import PointStamped, Vector3Stamped
from sensor_msgs.msg import CompressedImage, Image
from ament_index_python.packages import get_package_share_directory
import message_filters
import tf2_ros
import tf2_geometry_msgs

# Sử dụng package interface của bạn
from interfaces.msg import HumanState
from zed_msgs.msg import ObjectsStamped

class HandsignDetectorNode(Node):
    def __init__(self):
        super().__init__('handsign_detector_node')

        # 1. ĐỌC THÔNG SỐ TỪ PARAMS.YAML
        self.declare_parameter('hand_model_path', 'handsign.pt')
        self.declare_parameter('detection_conf', 0.5)
        self.declare_parameter('person_radius', 0.45)
        
        model_name = self.get_parameter('hand_model_path').value
        self.conf_thresh = self.get_parameter('detection_conf').value
        self.person_radius = self.get_parameter('person_radius').value

        # 2. CẤU HÌNH YOLO CHO BÀN TAY
        pkg_share_dir = get_package_share_directory('yolo')
        model_path = os.path.join(pkg_share_dir, 'models', model_name)
        
        self.get_logger().info(f'🚀 Đang tải YOLO Model: {model_name} (CUDA)...')
        self._yolo_hand = YOLO(model_path)
        self._yolo_hand.to('cuda')

        self._bridge = CvBridge()
        
        # 3. CẤU HÌNH TF BỘ LỌC TỌA ĐỘ
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._global_frame = 'base_footprint'
        
        # Trạng thái bám đuổi
        self._tracking_state = "IDLE"
        self._gesture_state = "IDLE"
        self._gesture_timer_start = None
        self._target_id = None
        self._current_hand_boxes = []

        # 4. PUBLISHERS
        qos_img = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self._pub_viz = self.create_publisher(CompressedImage, '/tracking/annotated_image/compressed', qos_img)
        self._pub_main = self.create_publisher(HumanState, '/tracking/main_person', 10)

        # 5. SUBSCRIBERS (Khớp chính xác với Topic của ZED)
        self.image_sub = message_filters.Subscriber(self, Image, '/zed/zed_node/rgb/color/rect/image', qos_profile=qos_img)
        self.body_sub = message_filters.Subscriber(self, ObjectsStamped, '/zed/zed_node/body_trk/skeletons', qos_profile=qos_img)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.body_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self._sync_callback)

        self.get_logger().info('✅ Node YOLO đã kết nối và tối ưu hoàn hảo với ZED 2.')

    def _sync_callback(self, img_msg: Image, body_msg: ObjectsStamped):
        # --- BƯỚC 1: FIX LỖI ẢNH RỖNG TỪ OPENCV ---
        if img_msg.width == 0 or img_msg.height == 0 or not img_msg.data:
            self.get_logger().warn('⏳ ZED đang warm-up, nhận được frame trống...', throttle_duration_sec=2.0)
            return

        try:
            frame = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Lỗi CvBridge: {e}", throttle_duration_sec=2.0)
            return

        if frame is None or frame.size == 0:
            return

        ts_now = self.get_clock().now()
        dt_float = ts_now.nanoseconds * 1e-9
        tracked_objects = []
        from_frame = body_msg.header.frame_id
        
        # --- BƯỚC 2: TỐI ƯU HÓA TF TRANSFORM ---
        try:
            # Chỉ lookup transform đúng 1 lần cho mỗi frame ảnh thay vì lookup cho từng người
            transform = self._tf_buffer.lookup_transform(
                self._global_frame, 
                from_frame, 
                rclpy.time.Time(), 
                timeout=Duration(seconds=0.05)
            )
            
            for obj in body_msg.objects:
                if obj.tracking_state != 1: 
                    continue
                
                tid = obj.label_id
                
                p = PointStamped()
                p.header = body_msg.header
                p.point.x, p.point.y, p.point.z = float(obj.position[0]), float(obj.position[1]), float(obj.position[2])
                
                v = Vector3Stamped()
                v.header = body_msg.header
                v.vector.x, v.vector.y, v.vector.z = float(obj.velocity[0]), float(obj.velocity[1]), float(obj.velocity[2])
                
                # Biến đổi mượt mà bằng do_transform
                p_base = tf2_geometry_msgs.do_transform_point(p, transform)
                v_base = tf2_geometry_msgs.do_transform_vector3(v, transform)
                
                xs = [c.kp[0] for c in obj.bounding_box_2d.corners]
                ys = [c.kp[1] for c in obj.bounding_box_2d.corners]
                px1, py1, px2, py2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                
                tracked_objects.append({
                    'id': tid, 'bbox': (px1, py1, px2, py2),
                    'px': p_base.point.x, 'py': p_base.point.y,
                    'vx': v_base.vector.x, 'vy': v_base.vector.y,
                    'dist': math.hypot(p_base.point.x, p_base.point.y),
                    'is_target': (tid == self._target_id)
                })
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f"Chưa có map TF từ {from_frame} sang {self._global_frame}", throttle_duration_sec=2.0)
            # Vẫn chạy hàm vẽ hình để hiển thị camera dù chưa có tọa độ

        # --- BƯỚC 3: XỬ LÝ SONG SONG ---
        self._handle_gestures(frame, tracked_objects, dt_float)
        self._publish_to_controller(tracked_objects, img_msg.header.stamp)
        self._visualize(frame, tracked_objects, img_msg.header.stamp)

    def _publish_to_controller(self, objects, ts):
        for o in objects:
            if o['is_target']:
                msg = HumanState()
                msg.id = o['id']
                msg.px = float(o['px'])
                msg.py = float(o['py'])
                msg.vx = float(o['vx'])
                msg.vy = float(o['vy'])
                self._pub_main.publish(msg)

    def _handle_gestures(self, frame, objects, dt_now):
        # Yolo vẫn xử lý tay để vẽ khung hình ngay cả khi SDK chưa bắt được người
        hand_results = self._yolo_hand.predict(frame, conf=self.conf_thresh, imgsz=416, half=True, verbose=False, device='cuda')[0]
        self._current_hand_boxes = []
        gesture_found = False
        
        if len(hand_results.boxes) > 0:
            for hand in hand_results.boxes.data:
                hx1, hy1, hx2, hy2, hconf, hcls = hand.cpu().numpy()
                hcx, hcy = (hx1 + hx2)/2, (hy1 + hy2)/2
                self._current_hand_boxes.append(((int(hx1), int(hy1), int(hx2), int(hy2)), int(hcls)))
                
                for obj in objects:
                    px1, py1, px2, py2 = obj['bbox']
                    if px1-100 < hcx < px2+100 and py1-100 < hcy < py2+100:
                        gesture_found = True
                        if int(hcls) == 0: # START (Class 0)
                            if self._gesture_state != "START_HELD":
                                self._gesture_state, self._gesture_timer_start = "START_HELD", dt_now
                            elif (dt_now - self._gesture_timer_start) >= 2.0:
                                if self._target_id != obj['id']:
                                    self._target_id, self._tracking_state = obj['id'], "TRACKING"
                                    obj['is_target'] = True
                                    self.get_logger().warn(f'🎯 ĐÃ KHÓA MỤC TIÊU: ID {self._target_id}')
                        elif int(hcls) == 1 and obj['id'] == self._target_id: # STOP (Class 1)
                            if self._gesture_state != "STOP_HELD":
                                self._gesture_state, self._gesture_timer_start = "STOP_HELD", dt_now
                            elif (dt_now - self._gesture_timer_start) >= 2.0:
                                self._target_id, self._tracking_state = None, "IDLE"
                                obj['is_target'] = False
                                self.get_logger().warn('⏹ ĐÃ HỦY THEO DÕI MỤC TIÊU.')
                        break # Tay đã thuộc về người này thì break vòng lặp người

        if not gesture_found:
            self._gesture_state, self._gesture_timer_start = "IDLE", None

    def _visualize(self, frame, objects, ts):
        status_color = (0, 0, 255) if self._tracking_state == "TRACKING" else (0, 255, 255)
        cv2.putText(frame, f"Robot: {self._tracking_state} | Hand: {self._gesture_state}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                    
        for o in objects:
            x1, y1, x2, y2 = o['bbox']
            color = (0, 0, 255) if o['is_target'] else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{o['id']} {o['dist']:.1f}m", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        for box, cls in self._current_hand_boxes:
            color = (255, 255, 0) if cls == 0 else (0, 165, 255)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            label = "START" if cls == 0 else "STOP"
            cv2.putText(frame, label, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        try:
            msg = self._bridge.cv2_to_compressed_imgmsg(frame, dst_format='jpg')
            msg.header.stamp, msg.header.frame_id = ts, self._global_frame
            self._pub_viz.publish(msg)
        except: pass

def main(args=None):
    rclpy.init(args=args)
    node = HandsignDetectorNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()