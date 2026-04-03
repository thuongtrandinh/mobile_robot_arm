"""
AMR ZED2 Package
Hand sign detection and person tracking for mobile robot
Supports both real ZED2 camera and Gazebo simulation
"""

__version__ = '2.0.0'

# Lazy imports to avoid circular dependency and reduce load time
__all__ = [
    'HandSignDetectorSimNode',
    'TrackingNode',
]
