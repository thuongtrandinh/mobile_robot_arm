import numpy as np
import cv2
from typing import Tuple, List

class DepthAwareNMS:
    """
    Depth-Aware NMS for improved bounding box filtering using depth information
    """
    
    def __init__(self, nms_threshold=0.45, depth_threshold=0.5):
        self.nms_threshold = nms_threshold
        self.depth_threshold = depth_threshold
    
    def get_box_depth(self, depth_map, x1: int, y1: int, x2: int, y2: int) -> Tuple[float, float, float]:
        h, w = depth_map.shape[:2]
        # Ép kiểu int để tránh lỗi slice index
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        
        if x2 <= x1 or y2 <= y1:
            return 0.0, 0.0, 0.0
        
        roi = depth_map[y1:y2, x1:x2]
        
        # [FIX BUG 3]: Xử lý an toàn các giá trị NaN và Inf của camera ZED
        valid_mask = np.isfinite(roi) & (roi > 0.1) & (roi < 10.0)
        valid_depths = roi[valid_mask]
        
        if len(valid_depths) == 0:
            return 0.0, 0.0, 0.0
        
        mean_depth = float(np.mean(valid_depths))
        variance = float(np.var(valid_depths))
        median_depth = float(np.median(valid_depths))
        
        return mean_depth, variance, median_depth
    
    def calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i, y1_i = max(x1_1, x1_2), max(y1_1, y1_2)
        x2_i, y2_i = min(x2_1, x2_2), min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def depth_consistency_score(self, depth1: float, depth2: float, variance1: float, variance2: float) -> float:
        if depth1 == 0 or depth2 == 0:
            return 1.0  
        
        depth_diff = abs(depth1 - depth2) / (max(depth1, depth2) + 1e-6)
        depth_diff *= 0.7  
        variance_penalty = (variance1 + variance2) / 5000.0
        consistency = max(0.0, 1.0 - depth_diff - variance_penalty)
        return consistency
    
    def apply(self, detections: List[dict], depth_map, conf_threshold: float = 0.0) -> List[dict]:
        """
        [FIX BUG 1]: Mặc định conf_threshold = 0.0 để tôn trọng các bộ lọc 
        đã được tính toán kỹ từ file tracking_node.py
        """
        if len(detections) == 0:
            return []
        
        filtered = [d for d in detections if d['conf'] >= conf_threshold]
        if len(filtered) == 0:
            return []
        
        for detection in filtered:
            x1, y1, x2, y2 = detection['box']
            mean_depth, variance, median_depth = self.get_box_depth(depth_map, x1, y1, x2, y2)
            detection['mean_depth'] = mean_depth
            detection['variance'] = variance
            detection['median_depth'] = median_depth
        
        filtered.sort(key=lambda x: x['conf'], reverse=True)
        
        keep = []
        suppressed = set() # [FIX BUG 2]: Thêm tập hợp các hộp bị loại bỏ
        
        for i in range(len(filtered)):
            if i in suppressed:
                continue
            
            keep.append(filtered[i])
            
            for j in range(i + 1, len(filtered)):
                if j in suppressed:
                    continue
                
                iou = self.calculate_iou(filtered[i]['box'], filtered[j]['box'])
                
                if iou > self.nms_threshold:
                    consistency = self.depth_consistency_score(
                        filtered[i]['median_depth'] if filtered[i]['median_depth'] > 0 else filtered[i]['mean_depth'],
                        filtered[j]['median_depth'] if filtered[j]['median_depth'] > 0 else filtered[j]['mean_depth'],
                        filtered[i]['variance'],
                        filtered[j]['variance']
                    )
                    
                    # Nếu trùng vị trí và cùng chiều sâu -> Loại bỏ thằng confidence thấp hơn (j)
                    if consistency >= self.depth_threshold:
                        suppressed.add(j)
        
        # Dọn dẹp key phụ
        for d in keep:
            if '_idx' in d:
                del d['_idx']
                
        return keep
    
    def refine_positions(self, detections: List[dict], depth_map, frame_width: int, frame_height: int) -> List[dict]:
        # (Giữ nguyên hàm này như cũ của bạn)
        for detection in detections:
            x1, y1, x2, y2 = detection['box']
            mean_depth = detection.get('mean_depth', 0)
            median_depth = detection.get('median_depth', 0)
            depth_value = median_depth if median_depth > 0 else mean_depth
            
            detection['center_x'] = (x1 + x2) // 2
            detection['center_y'] = (y1 + y2) // 2
            detection['width'] = x2 - x1
            detection['height'] = y2 - y1
            detection['depth'] = depth_value
            
            if depth_value > 0:
                normalized_depth = min(depth_value / 10.0, 1.0)
                detection['normalized_depth'] = normalized_depth
        
        return detections
