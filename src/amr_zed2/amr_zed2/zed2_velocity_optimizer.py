"""
ZED2 Optimized Velocity Estimation using Neural Depth
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class VelocityEstimate:
    """Velocity estimation result"""
    velocity_3d: Tuple[float, float, float]
    velocity_2d: Tuple[float, float]
    speed_3d: float
    confidence: float
    is_valid: bool


class ZED2VelocityEstimator:
    """Optimized velocity estimation for ZED2 camera"""

    __slots__ = ['fps', 'dt', 'velocity_history', 'max_history',
                 'min_depth_confidence', 'depth_quality_threshold']

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.dt = 1.0 / fps
        self.velocity_history: Dict[int, deque] = {}
        self.max_history = 5
        self.min_depth_confidence = 0.3
        self.depth_quality_threshold = 0.5

    def estimate_velocity_from_history(self, position_history: deque, depth_history: deque,
                                       depth_confidence_history: deque, dt: float) -> VelocityEstimate:
        """Estimate velocity from position history"""
        if len(position_history) < 2:
            return VelocityEstimate((0.0, 0.0, 0.0), (0.0, 0.0), 0.0, 0.0, False)

        oldest_pos_3d, oldest_pos_2d = position_history[0]
        newest_pos_3d, newest_pos_2d = position_history[-1]

        oldest_depth = depth_history[0] if depth_history else 0
        newest_depth = depth_history[-1] if depth_history else 0

        depth_confs = [c for c in depth_confidence_history if c > 0]
        avg_conf = float(np.mean(depth_confs)) if depth_confs else 0.0

        if oldest_depth <= 0 or newest_depth <= 0 or avg_conf < self.min_depth_confidence:
            return self._estimate_2d_velocity(oldest_pos_2d, newest_pos_2d, dt, avg_conf)

        if dt > 0:
            vx = (newest_pos_3d[0] - oldest_pos_3d[0]) / dt
            vy = (newest_pos_3d[1] - oldest_pos_3d[1]) / dt
            vz = (newest_pos_3d[2] - oldest_pos_3d[2]) / dt
            velocity_3d = (float(vx), float(vy), float(vz))
            speed = float(np.sqrt(vx**2 + vy**2 + vz**2))
        else:
            velocity_3d = (0.0, 0.0, 0.0)
            speed = 0.0

        n = len(position_history)
        vx_2d = (newest_pos_2d[0] - oldest_pos_2d[0]) / (n - 1) if n > 1 else 0.0
        vy_2d = (newest_pos_2d[1] - oldest_pos_2d[1]) / (n - 1) if n > 1 else 0.0

        depth_factor = min(1.0, avg_conf / self.depth_quality_threshold)
        consistency = self._motion_consistency(position_history)
        overall_conf = depth_factor * 0.6 + consistency * 0.4

        return VelocityEstimate(
            velocity_3d=velocity_3d,
            velocity_2d=(float(vx_2d), float(vy_2d)),
            speed_3d=speed,
            confidence=float(overall_conf),
            is_valid=overall_conf >= self.min_depth_confidence
        )

    def _estimate_2d_velocity(self, pos1_2d, pos2_2d, dt: float, depth_conf: float) -> VelocityEstimate:
        """Fallback 2D velocity estimation"""
        if dt > 0:
            vx_2d = (pos2_2d[0] - pos1_2d[0]) / dt * self.fps
            vy_2d = (pos2_2d[1] - pos1_2d[1]) / dt * self.fps
        else:
            vx_2d = vy_2d = 0.0

        return VelocityEstimate(
            velocity_3d=(0.0, 0.0, 0.0),
            velocity_2d=(float(vx_2d), float(vy_2d)),
            speed_3d=0.0,
            confidence=depth_conf * 0.5,
            is_valid=depth_conf > 0.2
        )

    def _motion_consistency(self, position_history: deque) -> float:
        """Calculate motion consistency score"""
        if len(position_history) < 3:
            return 0.5

        distances = []
        for i in range(len(position_history) - 1):
            pos1, pos2 = position_history[i][0], position_history[i+1][0]
            if pos1 != (0.0, 0.0, 0.0) and pos2 != (0.0, 0.0, 0.0):
                distances.append(np.linalg.norm(np.array(pos2) - np.array(pos1)))

        if len(distances) < 2:
            return 0.5

        avg_dist = np.mean(distances)
        if avg_dist == 0:
            return 0.0

        std_dist = np.std(distances)
        return float(min(1.0, max(0.0, 1.0 / (1.0 + std_dist / avg_dist))))

    def smooth_velocity(self, track_id: int, velocity_estimate: VelocityEstimate) -> VelocityEstimate:
        """Smooth velocity estimates over multiple frames"""
        if track_id not in self.velocity_history:
            self.velocity_history[track_id] = deque(maxlen=self.max_history)

        self.velocity_history[track_id].append(velocity_estimate)

        valid_vels = [(v.velocity_3d, v.confidence) for v in self.velocity_history[track_id] if v.is_valid]

        if not valid_vels:
            return velocity_estimate

        total_conf = sum(c for _, c in valid_vels)
        if total_conf == 0:
            return velocity_estimate

        weights = [c / total_conf for _, c in valid_vels]
        avg_vx = sum(v[0] * w for (v, _), w in zip(valid_vels, weights))
        avg_vy = sum(v[1] * w for (v, _), w in zip(valid_vels, weights))
        avg_vz = sum(v[2] * w for (v, _), w in zip(valid_vels, weights))

        velocities_2d = [v.velocity_2d for v in self.velocity_history[track_id]]
        avg_vx_2d = np.mean([v[0] for v in velocities_2d])
        avg_vy_2d = np.mean([v[1] for v in velocities_2d])

        speed = float(np.sqrt(avg_vx**2 + avg_vy**2 + avg_vz**2))
        avg_conf = total_conf / len(valid_vels)

        return VelocityEstimate(
            velocity_3d=(float(avg_vx), float(avg_vy), float(avg_vz)),
            velocity_2d=(float(avg_vx_2d), float(avg_vy_2d)),
            speed_3d=speed,
            confidence=float(avg_conf),
            is_valid=avg_conf >= self.min_depth_confidence
        )

    def reset(self, track_id: int = None):
        if track_id is None:
            self.velocity_history.clear()
        elif track_id in self.velocity_history:
            del self.velocity_history[track_id]


class OptimizedNMSVelocityIntegration:
    """Integration between Depth-Aware NMS and velocity estimation"""

    def __init__(self, fps: float = 30.0):
        self.velocity_estimator = ZED2VelocityEstimator(fps)
        self.fps = fps
