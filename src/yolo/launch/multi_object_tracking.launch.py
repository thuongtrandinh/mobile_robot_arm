#!/usr/bin/env python3
"""
Launch file for Multi-Object Obstacle Detection Node (GPU Optimized)

This launches the optimized multi-object tracking node:
  - Single YOLO model (no dual models)
  - No Re-ID features (GPU optimized)
  - Dynamic/Static classification
  - EKF tracking only for dynamic objects
  - Short-term memory (2s buffer)

Message Output:
  /tracking/dynamics (ObstacleArray):
    - Dynamic objects with EKF trajectory prediction
    - Includes: id, distance, radius, trajectory (1.5s future)
    - For motion planning algorithms (MPC, trajectory planning)

  /tracking/statics (StaticObstacleArray):
    - Static objects without motion prediction
    - Includes: px, py, pz (3D position)
    - For simple collision checking

  /tracking/annotated_image (Image):
    - Visualization with bounding boxes
    - Best Effort QoS (dropped if lag detected)

Dynamic Classes (with trajectory):
  person(0)

Static Classes (position only):
  chair(56), couch(57), bed(60), toilet(62), sink(67),
  potted_plant(64), stop_sign(73), etc.

Usage:
  # Standard launch (CUDA + FP16 enabled)
  ros2 launch yolo multi_object_tracking.launch.py

  # CPU-only mode
  ros2 launch yolo multi_object_tracking.launch.py use_cuda:=false

  # Full precision (slower but more accurate)
  ros2 launch yolo multi_object_tracking.launch.py use_fp16:=false

  # Lower confidence (catches smaller objects)
  ros2 launch yolo multi_object_tracking.launch.py object_conf:=0.35

  # Larger model (higher accuracy)
  ros2 launch yolo multi_object_tracking.launch.py model_path:=yolov8s.pt
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Generate launch description for GPU-optimized multi-object tracking"""
    
    # ===== LOAD CONFIG FILE =====
    yolo_pkg_share = get_package_share_directory('yolo')
    config_path = os.path.join(yolo_pkg_share, 'config', 'params.yaml')
    yolo_python = os.path.expanduser('~/miniconda3/envs/yolo_training/bin/python')
    default_python_prefix = yolo_python if os.path.exists(yolo_python) else ''
    
    # ===== LAUNCH ARGUMENTS =====
    declare_use_cuda = DeclareLaunchArgument(
        'use_cuda',
        default_value='true',
        description='Use CUDA GPU for YOLO inference (much faster)'
    )

    declare_use_fp16 = DeclareLaunchArgument(
        'use_fp16',
        default_value='true',
        description='Use FP16 half-precision (Tensor Core acceleration on RTX A4000)'
    )

    declare_model_path = DeclareLaunchArgument(
        'model_path',
        default_value='yolov8n.pt',
        description='YOLO model to use: yolov8n.pt (fast), yolov8s.pt (medium), yolov8m.pt (accurate)'
    )

    declare_object_conf = DeclareLaunchArgument(
        'object_conf',
        default_value='0.50',
        description='Object confidence threshold (0.30-0.70). Lower = more detections but more false positives'
    )

    declare_enable_metrics = DeclareLaunchArgument(
        'enable_metrics',
        default_value='true',
        description='Enable FPS/latency logging every 5 seconds'
    )

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock when running with Gazebo'
    )

    declare_config_file = DeclareLaunchArgument(
        'config_file',
        default_value=config_path,
        description='Path to multi-object tracking YAML config'
    )

    declare_python_prefix = DeclareLaunchArgument(
        'python_prefix',
        default_value=default_python_prefix,
        description='Optional Python executable prefix for running YOLO dependencies from a virtualenv/conda env'
    )

    # ===== MULTI-OBJECT TRACKING NODE =====
    # Load params.yaml for centralized configuration
    tracking_node = Node(
        package='yolo',
        executable='multi_object_tracking',
        name='multi_object_tracking_node',
        output='screen',
        prefix=LaunchConfiguration('python_prefix'),
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'use_cuda': LaunchConfiguration('use_cuda'),
                'use_fp16': LaunchConfiguration('use_fp16'),
                'model_path': LaunchConfiguration('model_path'),
                'yolo.object_conf_thresh': LaunchConfiguration('object_conf'),
            },
        ],
    )

    # ===== RETURN LAUNCH DESCRIPTION =====
    return LaunchDescription([
        declare_use_cuda,
        declare_use_fp16,
        declare_model_path,
        declare_object_conf,
        declare_enable_metrics,
        declare_use_sim_time,
        declare_config_file,
        declare_python_prefix,
        tracking_node,
    ])
