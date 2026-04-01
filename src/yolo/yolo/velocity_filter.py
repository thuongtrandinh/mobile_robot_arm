"""
Simple velocity filter for smooth and stable velocity estimation
"""

import numpy as np


class VelocityFilter:
    """
    Exponential moving average filter for velocity estimation
    Reduces noise and provides smooth velocity signals for controller
    """
    
    def __init__(self, alpha: float = 0.3):
        """
        Args:
            alpha: smoothing factor (0 < alpha <= 1)
                - Higher alpha: faster response, more noise
                - Lower alpha: smoother, more lag
        """
        self.alpha = alpha
        self.filtered_velocity = None
    
    def update(self, velocity: np.ndarray) -> np.ndarray:
        """
        Update filter with new velocity measurement
        
        Args:
            velocity: [vx, vy, vz] velocity vector
            
        Returns:
            filtered_velocity: smoothed velocity
        """
        if self.filtered_velocity is None:
            self.filtered_velocity = velocity.copy()
        else:
            self.filtered_velocity = (self.alpha * velocity + 
                                     (1 - self.alpha) * self.filtered_velocity)
        
        return self.filtered_velocity.copy()
    
    def reset(self):
        """Reset filter state"""
        self.filtered_velocity = None
