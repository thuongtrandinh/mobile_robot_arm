# Copyright 2026 H-AMPCC Authors
# SPDX-License-Identifier: Apache-2.0

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('amr_controller')

    params_file = os.path.join(pkg_dir, 'config', 'ampcc_params.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use simulation clock'),

        Node(
            package='amr_controller',
            executable='ampcc_node',
            name='ampcc_controller',
            output='screen',
            parameters=[
                params_file,
                {'use_sim_time': use_sim_time},
            ],
            remappings=[
                ('odom',        '/odometry/filtered'),
                ('global_path', '/global_path'),
                ('goal_pose',   '/goal_pose'),
                ('cmd_vel',     '/diff_cont/cmd_vel'),
            ],
        ),
    ])
