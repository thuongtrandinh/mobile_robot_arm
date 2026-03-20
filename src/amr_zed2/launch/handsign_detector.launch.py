"""
Launch file for AMR ZED2 Hand Sign Detector Node
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

    # Declare arguments
    show_display_arg = DeclareLaunchArgument(
        'show_display',
        default_value='true',
        description='Show OpenCV visualization window'
    )

    publish_image_arg = DeclareLaunchArgument(
        'publish_image',
        default_value='true',
        description='Publish visualization image to ROS topic'
    )

    # Hand sign detector node
    handsign_detector_node = Node(
        package='amr_zed2',
        executable='handsign_detector',
        name='handsign_detector',
        output='screen',
        parameters=[
            config_file,
            {
                'show_display': LaunchConfiguration('show_display'),
                'publish_image': LaunchConfiguration('publish_image'),
            }
        ],
        remappings=[
            ('tracking/status', '/amr/tracking/status'),
            ('tracking/hand_signal', '/amr/tracking/hand_signal'),
            ('tracking/person_pose', '/amr/tracking/person_pose'),
            ('tracking/person_velocity', '/amr/tracking/person_velocity'),
            ('tracking/collision_risk', '/amr/tracking/collision_risk'),
            ('tracking/image', '/amr/tracking/image'),
        ]
    )

    return LaunchDescription([
        show_display_arg,
        publish_image_arg,
        handsign_detector_node,
    ])
