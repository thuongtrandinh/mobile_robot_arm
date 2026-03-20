"""
Depth-Aware NMS for improved bounding box filtering using depth information
Optimized for ZED2 camera
"""

import numpy as np
from typing import Tuple, List


class DepthAwareNMS:
    """Depth-Aware NMS combining confidence with depth consistency"""

    __slots__ = ['nms_threshold', 'depth_threshold']

    def __init__(self, nms_threshold: float = 0.45, depth_threshold: float = 0.5):
        self.nms_threshold = nms_threshold
        self.depth_threshold = depth_threshold

    def get_box_depth(self, depth_map: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Tuple[float, float, float]:
        """Calculate mean, variance, median depth for bounding box"""
        h, w = depth_map.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0, 0.0, 0.0

        roi = depth_map[y1:y2, x1:x2]
        valid = roi[roi > 0]

        if len(valid) == 0:
            return 0.0, 0.0, 0.0

        return float(np.mean(valid)), float(np.var(valid)), float(np.median(valid))

    @staticmethod
    def calculate_iou(box1: Tuple, box2: Tuple) -> float:
        """Calculate IoU between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        x1_i, y1_i = max(x1_1, x1_2), max(y1_1, y1_2)
        x2_i, y2_i = min(x2_1, x2_2), min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        inter = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

    def depth_consistency_score(self, depth1: float, depth2: float, var1: float, var2: float) -> float:
        """Calculate depth consistency score between two detections"""
        if depth1 == 0 or depth2 == 0:
            return 1.0

        depth_diff = abs(depth1 - depth2) / (max(depth1, depth2) + 1e-6)
        var_penalty = (var1 + var2) / 1000.0

        return max(0.0, 1.0 - depth_diff - var_penalty)

    def apply(self, detections: List[dict], depth_map: np.ndarray, conf_threshold: float = 0.5) -> List[dict]:
        """Apply Depth-Aware NMS to detections"""
        if not detections:
            return []

        # Filter by confidence
        filtered = [d for d in detections if d['conf'] >= conf_threshold]
        if not filtered:
            return []

        # Calculate depth for each detection
        for det in filtered:
            x1, y1, x2, y2 = det['box']
            mean_d, var_d, median_d = self.get_box_depth(depth_map, x1, y1, x2, y2)
            det['mean_depth'] = mean_d
            det['variance'] = var_d
            det['median_depth'] = median_d

        # Sort by confidence
        filtered.sort(key=lambda x: x['conf'], reverse=True)

        # Apply NMS with depth awareness
        keep = []
        suppressed = set()

        for i, det in enumerate(filtered):
            if i in suppressed:
                continue

            keep.append(det)

            for j in range(i + 1, len(filtered)):
                if j in suppressed:
                    continue

                iou = self.calculate_iou(det['box'], filtered[j]['box'])
                if iou > self.nms_threshold:
                    consistency = self.depth_consistency_score(
                        det['mean_depth'], filtered[j]['mean_depth'],
                        det['variance'], filtered[j]['variance']
                    )
                    # Suppress only if same depth layer
                    if consistency >= self.depth_threshold:
                        suppressed.add(j)

        return keep

    def refine_positions(self, detections: List[dict], depth_map: np.ndarray,
                        frame_width: int, frame_height: int) -> List[dict]:
        """Refine bounding box positions using depth information"""
        for det in detections:
            x1, y1, x2, y2 = det['box']
            depth_value = det.get('median_depth') or det.get('mean_depth', 0)

            det['center_x'] = (x1 + x2) // 2
            det['center_y'] = (y1 + y2) // 2
            det['width'] = x2 - x1
            det['height'] = y2 - y1
            det['depth'] = depth_value

            if depth_value > 0:
                det['normalized_depth'] = min(depth_value / 10.0, 1.0)

        return detections
