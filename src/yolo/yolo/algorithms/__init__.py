"""
Optimization algorithms for hand tracking and gesture recognition
- botsort_handler: Multi-object tracking with appearance features
- appearance_feature_extractor: Lightweight color histogram features
- ctrv_ekf: Constant-turn-rate EKF for motion estimation
- velocity_filter: Exponential moving average velocity smoothing
- depth_aware_nms: Depth-aware non-maximum suppression for ZED2
"""

from .botsort_handler import BoTSortTracker
from .appearance_feature_extractor import FeatureExtractorLightweight
from .ctrv_ekf import CTRV_EKF
from .velocity_filter import VelocityFilter
from .depth_aware_nms import DepthAwareNMS

__all__ = [
    'BoTSortTracker',
    'FeatureExtractorLightweight',
    'CTRV_EKF',
    'VelocityFilter',
    'DepthAwareNMS',
]
