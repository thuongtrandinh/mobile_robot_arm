"""
BoT-SORT Tracker Handler - Bag of Tricks SORT
Advanced multi-object tracker with motion prediction and appearance memory
Maintains features and motion history even during occlusion for robust re-identification
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import deque


@dataclass
class MotionHistory:
    """Stores motion trajectory for prediction during occlusion"""
    positions: deque = field(default_factory=lambda: deque(maxlen=10))  # Last 10 positions
    velocities: deque = field(default_factory=lambda: deque(maxlen=10))  # Last 10 velocities
    
    def add(self, position: np.ndarray, velocity: np.ndarray):
        """Add position and velocity to history"""
        self.positions.append(position.copy())
        self.velocities.append(velocity.copy())
    
    def predict_next(self, steps: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next position using average velocity"""
        if len(self.velocities) == 0:
            return np.zeros(2), np.zeros(2)
        
        avg_velocity = np.mean(list(self.velocities), axis=0)
        if len(self.positions) == 0:
            return np.zeros(2), avg_velocity
        
        last_pos = np.array(self.positions[-1])
        predicted_pos = last_pos + avg_velocity * steps
        
        return predicted_pos, avg_velocity


@dataclass
class Track:
    """Single track state in BoT-SORT"""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    age: int = 1
    hit_streak: int = 0  # Consecutive successful matches
    time_since_update: int = 0  # Frames since last update
    features: np.ndarray = field(default_factory=lambda: np.array([]))  # Only for ACTIVE tracks
    motion_history: MotionHistory = field(default_factory=MotionHistory)  # Persistent motion memory
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))  # Current velocity
    prev_position: np.ndarray = field(default_factory=lambda: np.zeros(2))


class BoTSortTracker:
    """
    Bag of Tricks SORT Tracker with Motion Memory
    
    Key Features:
    - IOU-based association (geometric)
    - Appearance features for active tracks only (memory-efficient)
    - Motion history persistence (predicts during occlusion)
    - Multi-cue fusion: IOU + appearance + velocity prediction
    - Re-identification when object reappears
    
    Motion Prediction:
    - Stores motion history even when track is lost
    - Predicts trajectory during occlusion (kalman-like prediction)
    - Re-identifies object when it reappears with matching features
    
    Improvements:
    - Handles temporary occlusion gracefully
    - Predicts direction of movement during gap
    - Robust re-identification after reappearance
    """
    
    def __init__(self, 
                 track_thresh: float = 0.5,
                 track_low_thresh: float = 0.1,  # NEW: Low threshold for keeping weak detections
                 track_buffer: int = 30,
                 match_thresh: float = 0.8,
                 use_appearance: bool = True,
                 appearance_weight: float = 0.5,
                 max_age_before_predict: int = 5):
        """
        Args:
            track_thresh: Detection confidence threshold for starting new tracks
            track_low_thresh: Low threshold for BYTE Association Stage 2 (keep weak dets)
            track_buffer: Frames to keep lost tracks for re-identification
            match_thresh: IOU threshold for matching
            use_appearance: Use appearance features for active tracks
            appearance_weight: Weight of appearance in matching (0-1)
            max_age_before_predict: Use motion prediction after this many frames
        """
        self.track_thresh = track_thresh
        self.track_low_thresh = track_low_thresh  # BYTE Association low threshold
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.use_appearance = use_appearance
        self.appearance_weight = appearance_weight
        self.max_age_before_predict = max_age_before_predict
        
        self.tracks: List[Track] = []
        self.frame_count = 0
        self.next_track_id = 1
    
    def update(self, detections: np.ndarray, 
               features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Update tracker with Two-Stage Matching (BYTE Association)
        
        STAGE 1: Match HIGH confidence detections (conf > track_thresh)
        STAGE 2: Match LOW confidence detections (track_low_thresh < conf <= track_thresh)
                 to unmatched tracks to prevent flickering
        
        Args:
            detections: Nx5 array [x1, y1, x2, y2, confidence]
            features: Nx256 array of appearance features (optional)
        
        Returns:
            Mx5 array of tracks [x1, y1, x2, y2, track_id]
        """
        self.frame_count += 1
        
        # STEP 1: Separate detections into HIGH and LOW confidence
        if len(detections) > 0:
            scores = detections[:, 4]
            det_high_idx = scores > self.track_thresh
            det_low_idx = (scores > self.track_low_thresh) & (scores <= self.track_thresh)
            
            det_high = detections[det_high_idx]
            feat_high = features[det_high_idx] if features is not None else None
            
            det_low = detections[det_low_idx]
            feat_low = features[det_low_idx] if features is not None else None
        else:
            det_high = np.empty((0, 5))
            det_low = np.empty((0, 5))
            feat_high = None
            feat_low = None
        
        # STEP 2: Predict track positions
        predictions = self._predict_tracks()
        
        # STAGE 1: Match HIGH confidence detections with active tracks
        matched_idx_high, unmatched_dets_high, unmatched_tracks = self._associate_detections(
            det_high, predictions, feat_high
        )
        
        # Update tracks matched in STAGE 1
        for det_idx, track_idx in matched_idx_high:
            feat = feat_high[det_idx] if feat_high is not None else None
            self.tracks[track_idx].update(det_high[det_idx], feat)
        
        # STAGE 2: Match LOW confidence detections with unmatched tracks (BYTE Association)
        # This is the KEY to preventing flickering: use weak detections to keep tracks alive
        if len(det_low) > 0 and len(unmatched_tracks) > 0:
            unmatched_predictions = predictions[unmatched_tracks]
            
            # Associate det_low with unmatched tracks (IOU only, no appearance)
            matched_idx_low, _, unmatched_tracks_final = self._associate_detections(
                det_low, unmatched_predictions, features=None  # No appearance for low conf
            )
            
            # Update unmatched tracks with LOW detections (3-point rule: use weak detections)
            for det_idx, local_track_idx in matched_idx_low:
                real_track_idx = unmatched_tracks[local_track_idx]
                self.tracks[real_track_idx].update(det_low[det_idx], None)  # None for feature = no appearance update
            
            # Update unmatched_tracks to reflect STAGE 2 matches
            unmatched_tracks = [unmatched_tracks[i] for i in unmatched_tracks_final]
        
        # STEP 3: Create new tracks from unmatched HIGH detections
        for det_idx in unmatched_dets_high:
            feat = feat_high[det_idx] if feat_high is not None else None
            self._create_track(det_high[det_idx], feat)
        
        # STEP 4: Mark completely unmatched tracks as lost
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].time_since_update += 1
        
        # STEP 5: Remove dead tracks
        self.tracks = [t for t in self.tracks 
                      if t.time_since_update <= self.track_buffer + 10]
        
        # Return active tracks (recently matched)
        active_tracks = []
        for track in self.tracks:
            if track.time_since_update == 0:
                bbox = track.bbox
                active_tracks.append([
                    bbox[0], bbox[1], bbox[2], bbox[3], track.track_id
                ])
        
        return np.array(active_tracks) if active_tracks else np.empty((0, 5))
    
    def _predict_tracks(self) -> np.ndarray:
        """
        Predict track positions using:
        1. Current velocity (short-term)
        2. Motion history average (long-term during occlusion)
        """
        predictions = []
        
        for track in self.tracks:
            if track.time_since_update == 0:
                # Recently matched - use current velocity
                if len(track.velocity) > 0 and np.any(track.velocity):
                    predicted_bbox = track.bbox.copy()
                    predicted_bbox[0:2] += track.velocity
                    predicted_bbox[2:4] += track.velocity
                else:
                    predicted_bbox = track.bbox.copy()
            elif track.time_since_update <= self.max_age_before_predict:
                # Slightly occluded - use motion history prediction
                predicted_pos, avg_vel = track.motion_history.predict_next(steps=track.time_since_update)
                predicted_bbox = track.bbox.copy()
                # Update center
                curr_center = np.array([(track.bbox[0] + track.bbox[2])/2,
                                       (track.bbox[1] + track.bbox[3])/2])
                offset = predicted_pos - curr_center
                predicted_bbox[0:2] += offset
                predicted_bbox[2:4] += offset
            else:
                # Lost for too long - keep last position
                predicted_bbox = track.bbox.copy()
            
            predictions.append(predicted_bbox)
        
        return np.array(predictions) if predictions else np.empty((0, 4))
    
    def _associate_detections(self, 
                             detections: np.ndarray,
                             predictions: np.ndarray,
                             features: Optional[np.ndarray] = None):
        """
        Motion-aware association: IOU + Appearance + Motion Prediction
        
        Returns:
            matched_idx: List of (detection_idx, track_idx) tuples
            unmatched_detections: List of detection indices
            unmatched_tracks: List of track indices
        """
        if len(detections) == 0 or len(self.tracks) == 0:
            return [], list(range(len(detections))), list(range(len(self.tracks)))
        
        # Compute IOU matrix
        iou_matrix = self._compute_iou_matrix(detections, predictions)
        
        # Compute appearance distance only for ACTIVE tracks (with features)
        if self.use_appearance and features is not None and len(features) > 0:
            app_distance_matrix = self._compute_appearance_distance(features)
            # Fuse IOU and appearance
            cost_matrix = (1 - iou_matrix) * (1 - self.appearance_weight) + \
                         app_distance_matrix * self.appearance_weight
        else:
            cost_matrix = 1 - iou_matrix
        
        # Hungarian algorithm for optimal matching
        matched_idx = self._hungarian_algorithm(cost_matrix)
        
        # Separate matches/unmatches
        unmatched_detections = [d for d in range(len(detections)) 
                               if d not in matched_idx[:, 0]]
        unmatched_tracks = [t for t in range(len(self.tracks)) 
                           if t not in matched_idx[:, 1]]
        
        return matched_idx, unmatched_detections, unmatched_tracks
    
    def _compute_iou_matrix(self, detections: np.ndarray, 
                           predictions: np.ndarray) -> np.ndarray:
        """Compute IOU between detections and predicted tracks"""
        if len(predictions) == 0:
            return np.zeros((len(detections), 0))
        
        det_boxes = detections[:, :4]  # [x1, y1, x2, y2]
        pred_boxes = predictions[:, :4]
        
        iou_matrix = np.zeros((len(det_boxes), len(pred_boxes)))
        
        for i, det in enumerate(det_boxes):
            for j, pred in enumerate(pred_boxes):
                iou_matrix[i, j] = self._iou(det, pred)
        
        return iou_matrix
    
    def _iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute Intersection over Union"""
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
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _compute_appearance_distance(self, features: np.ndarray) -> np.ndarray:
        """
        Compute appearance distance (cosine distance) between detections and ACTIVE tracks only
        
        Returns:
            Distance matrix [len(features), len(tracks)]
        """
        distance_matrix = np.zeros((len(features), len(self.tracks)))
        
        for i, det_feature in enumerate(features):
            for j, track in enumerate(self.tracks):
                if len(track.features) > 0:  # Only for tracks with features
                    # Cosine distance
                    dist = 1 - self._cosine_similarity(det_feature, track.features)
                    distance_matrix[i, j] = dist
                else:
                    distance_matrix[i, j] = 1.0  # Max distance if no feature
        
        return distance_matrix
    
    def _cosine_similarity(self, A: np.ndarray, B: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B)
        
        if norm_A == 0 or norm_B == 0:
            return 0.0
        
        return np.dot(A, B) / (norm_A * norm_B)
    
    def _hungarian_algorithm(self, cost_matrix: np.ndarray) -> np.ndarray:
        """
        Simple greedy matching (approximation of Hungarian algorithm)
        In production, use scipy.optimize.linear_sum_assignment
        
        Returns:
            Nx2 array of matched indices [(det_idx, track_idx), ...]
        """
        matched = []
        used_dets = set()
        used_tracks = set()
        
        # Sort by cost and match greedily
        matches = []
        for i in range(len(cost_matrix)):
            for j in range(len(cost_matrix[i])):
                matches.append((cost_matrix[i, j], i, j))
        
        matches.sort()
        
        for cost, i, j in matches:
            if i not in used_dets and j not in used_tracks:
                if cost < self.match_thresh:
                    matched.append([i, j])
                    used_dets.add(i)
                    used_tracks.add(j)
        
        return np.array(matched) if matched else np.empty((0, 2), dtype=int)
    
    def _create_track(self, detection: np.ndarray, 
                     feature: Optional[np.ndarray] = None):
        """Create new track from detection"""
        track = Track(
            track_id=self.next_track_id,
            bbox=detection[:4].copy(),
            confidence=detection[4],
            features=feature if feature is not None else np.array([])
        )
        self.tracks.append(track)
        self.next_track_id += 1


# Extend Track class with update method
def _track_update(self, detection: np.ndarray, feature: Optional[np.ndarray] = None):
    """
    Update track with new detection + EMA Smoothing (Anti-Jitter)
    
    Key optimization: Apply Exponential Moving Average (EMA) to bounding box
    to prevent jerky movements while keeping responsiveness to actual motion.
    """
    # Update velocity and store in motion history
    center_new = np.array([(detection[0] + detection[2])/2, 
                          (detection[1] + detection[3])/2])
    if np.any(self.prev_position):
        self.velocity = center_new - self.prev_position
    self.prev_position = center_new.copy()
    
    # Store motion in history (persistent even when occluded)
    self.motion_history.add(center_new, self.velocity)
    
    # ===== EMA SMOOTHING (Anti-Jitter) =====
    # Only smooth if this is an active track update (time_since_update == 0 and has history)
    if self.time_since_update == 0 and self.hit_streak > 1:
        # Exponential Moving Average: trust new detection 70%, keep momentum 30%
        # This prevents jittering while maintaining responsiveness
        alpha = 0.7  # Higher alpha = more responsive to new detections
        self.bbox = alpha * detection[:4] + (1 - alpha) * self.bbox
    else:
        # First detection or after gap: use detection directly
        self.bbox = detection[:4].copy()
    
    self.confidence = detection[4]
    self.time_since_update = 0
    self.hit_streak += 1
    self.age += 1
    
    # Update feature ONLY if provided (only for active tracks)
    if feature is not None and len(feature) > 0:
        if len(self.features) > 0:
            # Exponential moving average for appearance features
            self.features = 0.9 * self.features + 0.1 * feature
        else:
            self.features = feature.copy()

Track.update = _track_update
