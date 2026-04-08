"""
State Machine Package for Mobile Robot Arm
Manages robot operational states with ZED2 camera optimization

Note: VelocityFilter and KalmanPredictor have been moved to yolo/algorithms/velocity_filter.py
for better separation of concerns (perception layer vs decision layer).
"""

from .state_manager import (
    RobotState,
    StateConfig,
    TrackingTarget,
    RobotStateMachine
)

__all__ = [
    'RobotState',
    'StateConfig',
    'TrackingTarget',
    'RobotStateMachine',
]

__version__ = '2.0.0'
__author__ = 'Mobile Robot Arm Team'
