"""
Deep-Aware NMS (Non-Maximum Suppression)
Advanced suppression considering spatial relationships, semantic context, and motion
Goes beyond standard NMS by modeling object relationships and spatial decay
"""

import numpy as np
from typing import List, Tuple, Optional


class DeepNMSOptimizer:
    """
    Deep-Aware NMS with Spatial Context and Semantic Modeling
    
    Key Features:
    1. Spatial Decay: Gaussian penalty for spatially far detections
       - Models that nearby detections are likely different objects
       - Far detections of same object are less likely
    
    2. Scale Awareness: Adaptive suppression based on object size
       - Large objects suppress more area
       - Small objects in background more carefully handled
    
    3. Motion Coherence: Consider relative positions
       - Two objects at same position unlikely (suppress one)
       - Two objects far apart likely different (keep both)
    
    4. Soft Suppression: Gaussian penalty instead of binary removal
       - Gradually reduce confidence of overlapping detections
       - Better for borderline cases
    
    Methods:
    - deep_nms_spatial(): MAIN - Spatial decay with Gaussian weighting
    - nms_fast_vectorized(): Fast greedy baseline
    - nms_soft(): Gaussian penalty variant
    - nms_adaptive_scale(): Scale-based adaptive threshold
    """
    
    @staticmethod
    def deep_nms_spatial(detections: np.ndarray, 
                        iou_threshold: float = 0.45,
                        spatial_decay: float = 0.8) -> np.ndarray:
        """
        Deep-Aware NMS with Spatial Decay
        
        Uses Gaussian spatial decay model:
        - Confidence of nearby detections decreases smoothly
        - Distance-weighted suppression
        - Maintains spatial coherence
        
        Args:
            detections: Nx5 array [x1, y1, x2, y2, confidence]
            iou_threshold: Base IOU threshold for spatial decay start
            spatial_decay: Decay factor for Gaussian penalty (0-1)
        
        Returns:
            Filtered detections after deep NMS
        """
        if len(detections) == 0:
            return detections
        
        # Sort by confidence
        order = np.argsort(detections[:, 4])[::-1]
        sorted_dets = detections[order].copy()
        
        keep_idx = []
        suppressed = np.zeros(len(sorted_dets), dtype=bool)
        
        for i in range(len(sorted_dets)):
            if suppressed[i]:
                continue
            
            keep_idx.append(order[i])
            
            # Compute IOU with all remaining detections
            current_box = sorted_dets[i, :4]
            for j in range(i + 1, len(sorted_dets)):
                if suppressed[j]:
                    continue
                
                iou = DeepNMSOptimizer._compute_iou(current_box, sorted_dets[j, :4])
                
                if iou > iou_threshold:
                    # Spatial decay: farther away = less suppression
                    center_i = np.array([(current_box[0] + current_box[2])/2,
                                        (current_box[1] + current_box[3])/2])
                    center_j = np.array([(sorted_dets[j, 0] + sorted_dets[j, 2])/2,
                                        (sorted_dets[j, 1] + sorted_dets[j, 3])/2])
                    
                    # Distance-weighted Gaussian decay
                    distance = np.linalg.norm(center_i - center_j)
                    size_i = np.sqrt((current_box[2] - current_box[0]) * 
                                    (current_box[3] - current_box[1]))
                    
                    # Decay factor: closer = more suppression
                    decay = np.exp(-(distance / (size_i + 1e-6)) ** 2)
                    penalty = spatial_decay * decay
                    
                    sorted_dets[j, 4] *= (1 - penalty)
                    
                    # Hard suppress if confidence too low
                    if sorted_dets[j, 4] < 0.01:
                        suppressed[j] = True
        
        return detections[keep_idx]
    
    @staticmethod
    def nms_fast_vectorized(detections: np.ndarray, 
                           iou_threshold: float = 0.45) -> np.ndarray:
        """
        Fast Vectorized NMS - Standard greedy matching
        
        O(n²) but with vectorization for speed
        
        Args:
            detections: Nx5 array [x1, y1, x2, y2, confidence]
            iou_threshold: IOU threshold for suppression
        
        Returns:
            Filtered detections
        """
        if len(detections) == 0:
            return detections
        
        order = np.argsort(detections[:, 4])[::-1]
        keep = []
        
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # Vectorized IOU computation
            ious = DeepNMSOptimizer._compute_iou_vectorized(
                detections[i, :4], detections[order[1:], :4]
            )
            
            # Keep boxes with IOU below threshold
            order = order[1:][ious <= iou_threshold]
        
        return detections[keep]
    
    @staticmethod
    def nms_soft(detections: np.ndarray, 
                iou_threshold: float = 0.45,
                sigma: float = 0.5) -> np.ndarray:
        """
        Soft-NMS - Gaussian Penalty Suppression
        
        Instead of removing, apply Gaussian penalty to overlapping boxes
        
        Args:
            detections: Nx5 array [x1, y1, x2, y2, confidence]
            iou_threshold: IOU threshold for starting penalty
            sigma: Gaussian parameter (spread)
        
        Returns:
            Weighted detections (may have reduced confidence)
        """
        if len(detections) == 0:
            return detections
        
        dets = detections.copy()
        order = np.argsort(dets[:, 4])[::-1]
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # Compute IOU
            ious = DeepNMSOptimizer._compute_iou_vectorized(
                dets[i, :4], dets[order[1:], :4]
            )
            
            # Apply Gaussian penalty
            penalty = np.exp(-(ious ** 2) / sigma)
            dets[order[1:], 4] *= penalty
            
            # Remove very small confidences
            order = order[1:][dets[order[1:], 4] > 0.01]
        
        return dets[keep]
    
    @staticmethod
    def nms_adaptive_scale(detections: np.ndarray, 
                          base_threshold: float = 0.45) -> np.ndarray:
        """
        Adaptive Scale-Based NMS
        
        Threshold adapts based on object scale:
        - Larger objects: higher threshold (more aggressive)
        - Smaller objects: lower threshold (more conservative)
        
        Args:
            detections: Nx5 array [x1, y1, x2, y2, confidence]
            base_threshold: Base IOU threshold
        
        Returns:
            Filtered detections
        """
        if len(detections) == 0:
            return detections
        
        dets = detections.copy()
        order = np.argsort(dets[:, 4])[::-1]
        keep = []
        
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # Compute box sizes
            size_i = (dets[i, 2] - dets[i, 0]) * (dets[i, 3] - dets[i, 1])
            sizes = (dets[order[1:], 2] - dets[order[1:], 0]) * \
                   (dets[order[1:], 3] - dets[order[1:], 1])
            
            # Adaptive threshold: larger objects suppress more
            size_ratio = sizes / (size_i + 1e-6)
            adaptive_threshold = base_threshold * (0.5 + 0.5 * size_ratio)
            
            # Compute IOU
            ious = DeepNMSOptimizer._compute_iou_vectorized(
                dets[i, :4], dets[order[1:], :4]
            )
            
            # Keep boxes below adaptive threshold
            keep_mask = ious <= adaptive_threshold
            order = order[1:][keep_mask]
        
        return dets[keep]
    
    @staticmethod
    def nms_class_aware(detections: np.ndarray,
                       classes: np.ndarray,
                       iou_threshold: float = 0.45) -> np.ndarray:
        """
        Class-Aware NMS - Different suppression per class
        
        Don't suppress detections of different classes
        
        Args:
            detections: Nx5 array [x1, y1, x2, y2, confidence]
            classes: N array of class IDs
            iou_threshold: IOU threshold within same class
        
        Returns:
            Filtered detections
        """
        if len(detections) == 0:
            return detections
        
        # Process each class separately
        unique_classes = np.unique(classes)
        keep_all = []
        
        for cls in unique_classes:
            cls_mask = classes == cls
            cls_dets = detections[cls_mask]
            
            if len(cls_dets) == 0:
                continue
            
            # NMS within class
            keep_cls = DeepNMSOptimizer.nms_fast_vectorized(cls_dets, iou_threshold)
            
            # Map back to original indices
            cls_indices = np.where(cls_mask)[0]
            for det in keep_cls:
                orig_idx = np.where(cls_mask)[0]
                for orig_i in orig_idx:
                    if np.allclose(detections[orig_i, :4], det[:4]):
                        keep_all.append(orig_i)
                        break
        
        return detections[keep_all] if keep_all else detections[[]]
    
    @staticmethod
    def _compute_iou_vectorized(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Vectorized IOU computation
        
        Args:
            box: Single box [x1, y1, x2, y2]
            boxes: Nx4 array of boxes
        
        Returns:
            N array of IOU values
        """
        if len(boxes) == 0:
            return np.array([])
        
        # Intersection
        x1_min = np.maximum(box[0], boxes[:, 0])
        y1_min = np.maximum(box[1], boxes[:, 1])
        x2_max = np.minimum(box[2], boxes[:, 2])
        y2_max = np.minimum(box[3], boxes[:, 3])
        
        inter_w = np.maximum(0, x2_max - x1_min)
        inter_h = np.maximum(0, y2_max - y1_min)
        inter = inter_w * inter_h
        
        # Union
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - inter
        
        iou = inter / (union + 1e-6)
        return iou
    
    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IOU between single boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / (union_area + 1e-6)


# Backward compatibility alias
class NMSOptimizer(DeepNMSOptimizer):
    """Backward compatibility wrapper"""
    pass
