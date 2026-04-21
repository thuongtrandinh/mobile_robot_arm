"""
Optimization algorithms for multi-object tracking (GPU-optimized)
- botsort_handler: Multi-object tracking (no appearance features for speed)
- ctrv_ekf: Constant-turn-rate EKF for motion estimation (dynamic objects only)
- velocity_filter: Exponential moving average velocity smoothing
- depth_aware_nms: Depth-aware non-maximum suppression for 3D filtering
"""

from .botsort_handler import BoTSortTracker
from .ctrv_ekf import CTRV_EKF
from .velocity_filter import VelocityFilter
from .depth_aware_nms import DepthAwareNMS

__all__ = [
    'BoTSortTracker',
    'CTRV_EKF',
    'VelocityFilter',
    'DepthAwareNMS',
]
