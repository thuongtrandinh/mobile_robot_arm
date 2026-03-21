#!/usr/bin/env python3
"""
Launch file for AMR simulation with human animation
This launches the main Gazebo simulation and the human walking animator
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get the package directory
    pkg_dir = get_package_share_directory('amr_descriptions')
    
    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )
    
    declare_x_pos = DeclareLaunchArgument(
        'x_pos',
        default_value='0.0',
        description='X position of robot'
    )
    
    declare_y_pos = DeclareLaunchArgument(
        'y_pos',
        default_value='0.0',
        description='Y position of robot'
    )
    
    declare_height = DeclareLaunchArgument(
        'height',
        default_value='0.0',
        description='Height of robot'
    )
    
    declare_world = DeclareLaunchArgument(
        'world',
        default_value='amr_simulation.world',
        description='World file to load'
    )
    
    declare_launch_rviz = DeclareLaunchArgument(
        'launch_rviz',
        default_value='true',
        description='Launch RViz'
    )
    
    # Include the main gazebo launch file
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dir, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'x_pos': LaunchConfiguration('x_pos'),
            'y_pos': LaunchConfiguration('y_pos'),
            'height': LaunchConfiguration('height'),
            'world': LaunchConfiguration('world'),
            'launch_rviz': LaunchConfiguration('launch_rviz'),
        }.items()
    )
    
    # Human animator node
    human_animator = Node(
        package='amr_descriptions',
        executable='human_animator.py',
        name='human_animator',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ]
    )
    
    return LaunchDescription([
        declare_use_sim_time,
        declare_x_pos,
        declare_y_pos,
        declare_height,
        declare_world,
        declare_launch_rviz,
        gazebo_launch,
        human_animator,
    ])
