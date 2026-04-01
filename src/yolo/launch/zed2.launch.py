import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():

    # Chạy node YOLO handsign_detector của package yolo
    yolo_node = Node(
        package='yolo',
        executable='handsign_detector',
        name='handsign_detector',
        output='screen'
    )

    return LaunchDescription([
        yolo_node
    ])