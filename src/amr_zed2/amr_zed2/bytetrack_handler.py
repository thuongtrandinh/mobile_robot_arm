"""
ByteTrack Implementation for Multi-Object Tracking with Velocity Estimation
Optimized for hand sign detection and person tracking
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class Track:
    """Tracked object with ID and state"""
    track_id: int
    bbox: Tuple[int, int, int, int]
    center_3d: Tuple[float, float, float]
    confidence: float
    class_id: int
    label: str
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    position_history: deque = field(default_factory=lambda: deque(maxlen=10))
    velocity_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity_2d: Tuple[float, float] = (0.0, 0.0)
    speed_3d: float = 0.0


class ByteTracker:
    """ByteTrack Algorithm for Multi-Object Tracking"""

    __slots__ = ['max_age', 'min_hits', 'iou_threshold', 'track_high_thresh',
                 'track_low_thresh', 'new_track_thresh', 'tracks', 'frame_id',
                 'next_id', 'fps_estimate']

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.5,
                 track_high_thresh: float = 0.6, track_low_thresh: float = 0.1,
                 new_track_thresh: float = 0.5):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.tracks: Dict[int, Track] = {}
        self.frame_id = 0
        self.next_id = 1
        self.fps_estimate = 30.0

    def set_fps(self, fps: float):
        self.fps_estimate = max(fps, 1.0)

    @staticmethod
    def calculate_iou(box1: Tuple, box2: Tuple) -> float:
        """Calculate IoU between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        inter_x1, inter_y1 = max(x1_1, x1_2), max(y1_1, y1_2)
        inter_x2, inter_y2 = min(x2_1, x2_2), min(y2_1, y2_2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        union_area = ((x2_1 - x1_1) * (y2_1 - y1_1) +
                      (x2_2 - x1_2) * (y2_2 - y1_2) - inter_area)

        return inter_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def calculate_3d_distance(pos1: Tuple[float, float, float],
                             pos2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two 3D points"""
        if pos1[0] == 0 or pos2[0] == 0:
            return float('inf')
        return float(np.linalg.norm(np.array(pos2) - np.array(pos1)))

    def match_detections(self, detections: List[Dict], tracks: List[Track]):
        """Match detections to existing tracks"""
        if not tracks:
            return [], list(range(len(detections))), []
        if not detections:
            return [], [], list(range(len(tracks)))

        # Build cost matrix
        n_det, n_track = len(detections), len(tracks)
        cost_matrix = np.full((n_det, n_track), float('inf'))

        for i, det in enumerate(detections):
            for j, track in enumerate(tracks):
                iou = self.calculate_iou(det['box'], track.bbox)
                if det.get('center_3d') and track.center_3d != (0.0, 0.0, 0.0):
                    dist_3d = self.calculate_3d_distance(det['center_3d'], track.center_3d)
                    dist_cost = min(dist_3d / 5.0, 1.0)
                    cost_matrix[i, j] = (1.0 - iou) * 0.7 + dist_cost * 0.3
                else:
                    cost_matrix[i, j] = 1.0 - iou

        # Greedy matching
        matched_pairs = []
        matched_det = set()
        matched_track = set()

        # Sort by confidence
        det_order = sorted(range(n_det), key=lambda x: detections[x]['confidence'], reverse=True)

        for det_idx in det_order:
            if det_idx in matched_det:
                continue

            best_track, best_cost = -1, 1.0 - self.iou_threshold
            for track_idx in range(n_track):
                if track_idx in matched_track:
                    continue
                if cost_matrix[det_idx, track_idx] < best_cost:
                    best_cost = cost_matrix[det_idx, track_idx]
                    best_track = track_idx

            if best_track >= 0:
                matched_pairs.append((det_idx, best_track))
                matched_det.add(det_idx)
                matched_track.add(best_track)

        unmatched_det = [i for i in range(n_det) if i not in matched_det]
        unmatched_track = [i for i in range(n_track) if i not in matched_track]

        return matched_pairs, unmatched_det, unmatched_track

    def update_track_velocity(self, track: Track, new_center_3d: Tuple, new_center_2d: Tuple):
        """Update track velocity from position history"""
        track.position_history.append((new_center_3d, new_center_2d))

        if len(track.position_history) < 2:
            return

        history_len = min(5, len(track.position_history))
        old_pos_3d, old_pos_2d = track.position_history[0]
        new_pos_3d, new_pos_2d = track.position_history[-1]

        dt = (history_len - 1) / self.fps_estimate

        if new_pos_3d != (0.0, 0.0, 0.0) and old_pos_3d != (0.0, 0.0, 0.0) and dt > 0:
            vx = (new_pos_3d[0] - old_pos_3d[0]) / dt
            vy = (new_pos_3d[1] - old_pos_3d[1]) / dt
            vz = (new_pos_3d[2] - old_pos_3d[2]) / dt
            track.velocity_3d = (float(vx), float(vy), float(vz))
            track.speed_3d = float(np.sqrt(vx**2 + vy**2 + vz**2))

        if history_len > 1:
            vx_2d = (new_pos_2d[0] - old_pos_2d[0]) / (history_len - 1)
            vy_2d = (new_pos_2d[1] - old_pos_2d[1]) / (history_len - 1)
            track.velocity_2d = (float(vx_2d), float(vy_2d))

    def update(self, detections: List[Dict]) -> List[Track]:
        """Update tracker with new detections"""
        self.frame_id += 1
        active_tracks = list(self.tracks.values())

        matched_pairs, unmatched_dets, _ = self.match_detections(detections, active_tracks)

        updated_ids = set()

        # Update matched tracks
        for det_idx, track_idx in matched_pairs:
            det = detections[det_idx]
            track = active_tracks[track_idx]

            track.bbox = det['box']
            track.confidence = det['confidence']
            track.center_3d = det.get('center_3d', (0.0, 0.0, 0.0))

            center_2d = det.get('center_2d', ((det['box'][0] + det['box'][2]) // 2,
                                              (det['box'][1] + det['box'][3]) // 2))
            self.update_track_velocity(track, track.center_3d, center_2d)

            track.hits += 1
            track.time_since_update = 0
            updated_ids.add(track.track_id)

        # Create new tracks
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            if det['confidence'] >= self.new_track_thresh:
                new_track = Track(
                    track_id=self.next_id,
                    bbox=det['box'],
                    center_3d=det.get('center_3d', (0.0, 0.0, 0.0)),
                    confidence=det['confidence'],
                    class_id=det.get('class_id', 0),
                    label=det.get('label', 'unknown')
                )
                self.tracks[self.next_id] = new_track
                self.next_id += 1
                updated_ids.add(new_track.track_id)

        # Age unmatched tracks
        for track in active_tracks:
            if track.track_id not in updated_ids:
                track.time_since_update += 1
                track.age += 1

        # Remove dead tracks
        dead = [tid for tid, t in self.tracks.items() if t.time_since_update > self.max_age]
        for tid in dead:
            del self.tracks[tid]

        # Return confirmed tracks
        return [t for t in self.tracks.values() if t.hits >= self.min_hits or t.time_since_update == 0]

    def reset(self):
        self.tracks.clear()
        self.frame_id = 0
        self.next_id = 1
