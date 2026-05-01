import numpy as np
import cv2
from typing import Tuple, List

class DepthAwareNMS:
    """
    Depth-Aware NMS OPTIMIZED for Multi-Object Avoidance:
    
    CRITICAL FIXES:
    1. Class-Aware NMS: Never suppress objects of different classes (Person vs Chair)
    2. Top-Center Sampling: Sample depth from head/shoulders (top 30%) not belly (center)
       This prevents misreading depth when person stands behind furniture
    3. Relaxed IoU Threshold: 0.65 instead of 0.45 to prevent flickering when objects touch
    
    Real-world scenario handled:
    - Person behind chair: Head/shoulders sampled correctly (not occluded by chair)
    - Chair preserved: Different class, never deleted by NMS
    - No flickering: Higher IoU threshold (0.65) prevents jitter during contact
    """
    
    def __init__(self, nms_threshold=0.65, depth_threshold=0.5):
        """
        Args:
            nms_threshold: IoU threshold (OPTIMIZED: 0.65 instead of 0.45)
                          Prevents flickering when objects touch
            depth_threshold: Z-distance threshold (meters) to consider objects different
        """
        self.nms_threshold = nms_threshold
        self.depth_threshold = depth_threshold
    
    def get_robust_depth(self, depth_map, box: Tuple[int, int, int, int]) -> float:
        """
        TOP-CENTER SAMPLING for occluded objects:
        
        Instead of sampling from bbox center (which gets occluded by foreground objects),
        sample from top 30% (head/shoulders region for persons, top of furniture for objects).
        
        This ensures we get the ACTUAL depth of the object, not the depth of what's in front.
        
        Args:
            depth_map: Depth frame from camera (float32, units: meters)
            box: Bounding box (x1, y1, x2, y2)
        
        Returns:
            Median depth value (float), or 0.0 if invalid
        """
        h_map, w_map = depth_map.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        
        bw, bh = x2 - x1, y2 - y1
        
        # Horizontal: Sample from 35%-65% (avoid left/right edges)
        cx1 = max(0, int(x1 + bw * 0.35))
        cx2 = min(w_map, int(x2 - bw * 0.35))
        
        # ===== KEY FIX: TOP-CENTER SAMPLING =====
        # Vertical: Sample from top 10%-40% (head/shoulders, not belly/chest)
        # Skip top 10% to avoid background noise behind head
        # Use top 40% where most person/object identity is visible
        cy1 = max(0, int(y1 + bh * 0.10))
        cy2 = min(h_map, int(y1 + bh * 0.40))
        
        if cx2 <= cx1 or cy2 <= cy1:
            return 0.0
        
        # Extract ROI from top-center region
        roi = depth_map[cy1:cy2, cx1:cx2]
        
        # Filter valid depths (0.3m - 8m for D435i)
        valid_mask = np.isfinite(roi) & (roi > 0.3) & (roi < 8.0)
        valid_depths = roi[valid_mask]
        
        if len(valid_depths) == 0:
            return 0.0
        
        # Use MEDIAN to resist noise (chair gaps, background)
        return float(np.median(valid_depths))
    
    def calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate Intersection over Union (IoU) of 2 bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection region
        x1_i, y1_i = max(x1_1, x1_2), max(y1_1, y1_2)
        x2_i, y2_i = min(x2_1, x2_2), min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def apply(self, detections: List[dict], depth_map, conf_threshold: float = 0.5) -> List[dict]:
        """
        Apply Depth-Aware NMS with Class-Aware suppression
        
        FIX 1: Class-Aware NMS - Never suppress objects of different classes
        FIX 2: Top-Center Sampling - Get correct depth for occluded objects
        FIX 3: Higher IoU Threshold - Prevent flickering when objects touch
        
        Args:
            detections: List of detection dicts with keys: box, conf, cls
            depth_map: Depth frame
            conf_threshold: Confidence threshold for filtering detections
        
        Returns:
            Filtered detections list (after class-aware NMS)
        """
        if not detections:
            return []
        
        # Filter by confidence
        filtered = [d for d in detections if d.get('conf', d.get('confidence', 0)) >= conf_threshold]
        if not filtered:
            return []
        
        # Compute robust depth for each detection
        for det in filtered:
            det['z_depth'] = self.get_robust_depth(depth_map, det['box'])
        
        # Sort by confidence (high confidence first)
        filtered.sort(key=lambda x: x.get('conf', x.get('confidence', 0)), reverse=True)
        
        keep = []
        suppressed = [False] * len(filtered)
        
        for i in range(len(filtered)):
            if suppressed[i]:
                continue
            
            keep.append(filtered[i])
            z1 = filtered[i]['z_depth']
            cls_i = filtered[i].get('cls', -1)
            
            for j in range(i + 1, len(filtered)):
                if suppressed[j]:
                    continue
                
                cls_j = filtered[j].get('cls', -1)
                
                # ===== FIX 1: CLASS-AWARE NMS =====
                # If objects belong to different classes (Person vs Chair),
                # NEVER suppress them, even if they overlap 100%
                if cls_i != cls_j and cls_i >= 0 and cls_j >= 0:
                    # Different classes: protect both
                    continue
                
                # Calculate 2D overlap
                iou = self.calculate_iou(filtered[i]['box'], filtered[j]['box'])
                
                # ===== FIX 3: RELAXED IoU THRESHOLD =====
                # Only start considering suppression if IoU > 0.65 (instead of 0.45)
                # This prevents flickering when two people are merely touching shoulders
                if iou > self.nms_threshold:
                    z2 = filtered[j]['z_depth']
                    
                    # Check depth separation
                    if z1 > 0 and z2 > 0:
                        depth_diff = abs(z1 - z2)
                        
                        # If same class but different Z-depth: keep both
                        if depth_diff > self.depth_threshold:
                            continue
                        else:
                            # Same class, same location, same depth: suppress lower confidence
                            suppressed[j] = True
                    else:
                        # Invalid depth: fall back to traditional NMS
                        suppressed[j] = True
        
        return keep
    
    def refine_positions(self, detections: List[dict], depth_map, frame_width: int, frame_height: int) -> List[dict]:
        """Add position refinement metadata (backward compatibility)"""
        for detection in detections:
            x1, y1, x2, y2 = detection['box']
            z_depth = detection.get('z_depth', 0)
            
            detection['center_x'] = (x1 + x2) // 2
            detection['center_y'] = (y1 + y2) // 2
            detection['width'] = x2 - x1
            detection['height'] = y2 - y1
            detection['depth'] = z_depth
            
            if z_depth > 0:
                normalized_depth = min(z_depth / 10.0, 1.0)
                detection['normalized_depth'] = normalized_depth
        
        return detections
