"""
Launch file for AMR ZED2 Hand Sign Detector Node
Publishes tracking data in controller-compatible format
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('amr_zed2')
    config_file = os.path.join(pkg_dir, 'config', 'params.yaml')

    # Declare launch arguments
    args = [
        DeclareLaunchArgument(
            'detection_conf',
            default_value='0.5',
            description='Detection confidence threshold (0-1)'
        ),
        DeclareLaunchArgument(
            'enable_visualization',
            default_value='false',
            description='Enable OpenCV visualization window'
        ),
        DeclareLaunchArgument(
            'hand_model',
            default_value='handsign.pt',
            description='Path to hand sign detection model'
        ),
        DeclareLaunchArgument(
            'person_model',
            default_value='yolov8n.pt',
            description='Path to person detection model'
        ),
    ]

    # Hand sign detector node
    handsign_node = Node(
        package='amr_zed2',
        executable='handsign_detector',
        name='handsign_detector',
        output='screen',
        parameters=[
            {
                'detection_conf': LaunchConfiguration('detection_conf'),
                'enable_visualization': LaunchConfiguration('enable_visualization'),
                'hand_model_path': LaunchConfiguration('hand_model'),
                'person_model_path': LaunchConfiguration('person_model'),
                'person_radius': 0.3,
            }
        ],
    )

    ld = LaunchDescription(args)
    ld.add_action(handsign_node)
    return ld
