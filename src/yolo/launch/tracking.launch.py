#!/usr/bin/env python3
"""
Launch file for Multi-Object Obstacle Detection (Refactored)

This launches ONE node:
  1. tracking_node: Multi-class COCO detection + Dynamic/Static classification + EKF tracking

Message Flow:
  tracking_node publishes:
    - /tracking/dynamics (ObstacleArray): Dynamic objects with EKF trajectory prediction
    - /tracking/statics (ObstacleArray): Static obstacles with zero velocity
    - /tracking/annotated_image (Image): Visualization (Best Effort QoS)

Dynamic Classes (with velocity):
  person, car, motorcycle, bus, truck, cat, dog, horse

Static Classes (zero velocity):
  furniture, containers, kitchen items, etc.

Usage:
  # Standard launch
  ros2 launch yolo tracking.launch.py

  # With CUDA disabled (CPU-only)
  ros2 launch yolo tracking.launch.py use_cuda:=false

  # With FP16 disabled (full precision)
  ros2 launch yolo tracking.launch.py use_fp16:=false

  # With lower confidence threshold (catches more objects)
  ros2 launch yolo tracking.launch.py object_conf:=0.30
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate launch description for multi-object obstacle detection"""
    
    # ===== LAUNCH ARGUMENTS =====
    declare_use_cuda = DeclareLaunchArgument(
        'use_cuda',
        default_value='true',
        description='Use CUDA for YOLO inference'
    )

    declare_use_fp16 = DeclareLaunchArgument(
        'use_fp16',
        default_value='true',
        description='Use FP16 half-precision for YOLO inference'
    )

    declare_object_conf = DeclareLaunchArgument(
        'object_conf',
        default_value='0.50',
        description='Object detection confidence threshold'
    )

    declare_enable_metrics = DeclareLaunchArgument(
        'enable_metrics',
        default_value='true',
        description='Enable performance metrics logging'
    )

    # ===== TRACKING NODE =====
    # Multi-class COCO detection + Dynamic/Static classification
    # Publishes: /tracking/dynamics, /tracking/statics, /tracking/annotated_image
    tracking_node = Node(
        package='yolo',
        executable='tracking',
        name='tracking_node',
        output='screen',
        parameters=[{
            # Camera topics
            'camera_image_topic': '/camera/color/image_raw',
            'depth_topic': '/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/color/camera_info',
            
            # Model
            'model_path': 'yolov8n.pt',
            
            # Hardware
            'use_cuda': LaunchConfiguration('use_cuda'),
            'use_fp16': LaunchConfiguration('use_fp16'),
            
            # Performance
            'max_inference_time_ms': 200.0,
            'enable_frame_skip': True,
            'enable_performance_metrics': LaunchConfiguration('enable_metrics'),
        }],
        remappings=[],
    )

    # ===== RETURN LAUNCH DESCRIPTION =====
    return LaunchDescription([
        declare_use_cuda,
        declare_use_fp16,
        declare_object_conf,
        declare_enable_metrics,
        tracking_node,
    ])
