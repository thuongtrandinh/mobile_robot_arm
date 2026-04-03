"""
State Machine Package for Mobile Robot Arm
Manages robot operational states with ZED2 camera optimization
"""

from .state_manager import (
    RobotState, 
    StateConfig, 
    TrackingTarget,
    RobotStateMachine
)

from .zed2_optimizer import (
    ZED2Config,
    ZED2TrackingOptimizer,
    VelocityFilter,
    KalmanPredictor
)

__all__ = [
    'RobotState',
    'StateConfig', 
    'TrackingTarget',
    'RobotStateMachine',
    'ZED2Config',
    'ZED2TrackingOptimizer',
    'VelocityFilter',
    'KalmanPredictor',
]

__version__ = '2.0.0'
__author__ = 'Mobile Robot Arm Team'
