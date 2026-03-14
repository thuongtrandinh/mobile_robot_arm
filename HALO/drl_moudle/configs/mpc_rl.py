from configs.config import BaseEnvConfig
import numpy as np


class EnvConfig(BaseEnvConfig):
    def __init__(self, debug=False):
        super(EnvConfig, self).__init__(debug)
        # FOV: 110 degrees total => half-angle = 55 degrees
        # Set to np.pi to restore full 360-degree visibility
        self.robot.fov_half_angle = np.radians(55)
