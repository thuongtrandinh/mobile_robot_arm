#!/usr/bin/env python3
"""
Human Walking Animation Controller
This script controls the walking animation of the human model in Gazebo.
It publishes joint commands to animate the legs and arms in a walking motion.
"""

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math
import time


class HumanWalkingAnimator(Node):
    """Controller for human walking animation in Gazebo"""

    def __init__(self):
        super().__init__('human_animator')
        
        # Create publisher for joint trajectory commands
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/human_actor/joint_trajectory_controller/joint_trajectory',
            10
        )
        
        # Walking parameters
        self.walk_cycle_duration = 1.0  # seconds for complete walking cycle
        self.walk_speed = 0.5  # m/s
        self.start_time = time.time()
        
        # Define joints that will be animated
        self.animated_joints = [
            'left_hip',
            'right_hip',
            'left_knee',
            'right_knee',
            'left_shoulder',
            'right_shoulder',
        ]
        
        # Create a timer to update animation at 50 Hz
        self.timer = self.create_timer(0.02, self.animate_walking)
        self.get_logger().info('Human Walking Animator initialized')
    
    def calculate_joint_angles(self, time_in_cycle):
        """
        Calculate joint angles for walking animation
        
        Args:
            time_in_cycle: Current time within the walking cycle (0 to 1)
            
        Returns:
            Dictionary of joint_name -> angle (in radians)
        """
        # Use sinusoidal motion for smooth animation
        joint_angles = {}
        
        # Hip motion: alternate between forward and backward
        # Left hip leads at start of cycle
        hip_amplitude = 0.6
        joint_angles['left_hip'] = hip_amplitude * math.sin(2 * math.pi * time_in_cycle)
        joint_angles['right_hip'] = hip_amplitude * math.sin(2 * math.pi * (time_in_cycle + 0.5))
        
        # Knee motion: bend during swing phase
        knee_amplitude = 1.0
        # Knee bends when hip swings forward
        knee_phase_left = math.sin(2 * math.pi * time_in_cycle)
        knee_phase_right = math.sin(2 * math.pi * (time_in_cycle + 0.5))
        
        joint_angles['left_knee'] = max(0.0, knee_amplitude * knee_phase_left)
        joint_angles['right_knee'] = max(0.0, knee_amplitude * knee_phase_right)
        
        # Shoulder motion: opposite to leg motion for balance
        shoulder_amplitude = 0.4
        joint_angles['left_shoulder'] = -shoulder_amplitude * math.sin(2 * math.pi * time_in_cycle)
        joint_angles['right_shoulder'] = -shoulder_amplitude * math.sin(2 * math.pi * (time_in_cycle + 0.5))
        
        return joint_angles
    
    def animate_walking(self):
        """Publish joint trajectory command for walking animation"""
        # Calculate current time in walking cycle
        elapsed_time = time.time() - self.start_time
        time_in_cycle = (elapsed_time % self.walk_cycle_duration) / self.walk_cycle_duration
        
        # Get joint angles for current phase
        joint_angles = self.calculate_joint_angles(time_in_cycle)
        
        # Create trajectory message
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.animated_joints
        
        # Create a point for the current joint positions
        point = JointTrajectoryPoint()
        point.positions = [float(joint_angles[joint]) for joint in self.animated_joints]
        
        # Set velocity to smoothly transition to next position
        point.velocities = [0.5] * len(self.animated_joints)
        
        # Set time from start (incremental time)
        point.time_from_start = Duration(sec=0, nanosec=50_000_000)  # 50ms
        
        trajectory_msg.points = [point]
        
        # Publish the trajectory
        self.trajectory_pub.publish(trajectory_msg)


def main(args=None):
    rclpy.init(args=args)
    animator = HumanWalkingAnimator()
    
    try:
        rclpy.spin(animator)
    except KeyboardInterrupt:
        animator.get_logger().info('Walking animation stopped')
    finally:
        animator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
