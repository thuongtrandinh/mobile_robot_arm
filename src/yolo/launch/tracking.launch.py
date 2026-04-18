#!/usr/bin/env python3
"""
Launch file for integrated hand sign tracking + state machine

This launches TWO nodes:
  1. tracking_node: Hand gesture detection + 3D position estimation (YOLO + CTRV-EKF)
  2. state_machine_node: Robot state machine (IDLE -> START -> TRACKING -> RE_TRACKING -> STOP)

Message Flow:
  tracking_node publishes:
    - /yolo/gesture_command (String): "START" or "STOP"
    - /tracking/main_person (HumanState): px, py, vx, vy, radius
    - /tracking/annotated_image (Image): Visualization

  state_machine_node subscribes to:
    - /yolo/gesture_command (String)
    - /tracking/main_person (HumanState)
    And publishes control commands to robot

Usage:
  # Standard launch (both nodes)
  ros2 launch yolo tracking_with_state_machine.launch.py

  # With custom gesture hold time (0.5s - 2.0s)
  ros2 launch yolo tracking_with_state_machine.launch.py gesture_hold_time:=0.8

  # Tracking node only (for debugging)
  ros2 launch yolo tracking_with_state_machine.launch.py enable_state_machine:=false

  # State machine only (with external gesture source)
  ros2 launch yolo tracking_with_state_machine.launch.py enable_tracking:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # ===== LAUNCH ARGUMENTS =====
    
    declare_enable_tracking = DeclareLaunchArgument(
        'enable_tracking',
        default_value='true',
        description='Enable hand tracking node'
    )

    declare_enable_state_machine = DeclareLaunchArgument(
        'enable_state_machine',
        default_value='true',
        description='Enable robot state machine node'
    )

    declare_tracking_timeout = DeclareLaunchArgument(
        'tracking_timeout',
        default_value='0.3',
        description='Max time (seconds) allowed when person tracking is lost before re-tracking'
    )

    declare_retracking_timeout = DeclareLaunchArgument(
        'retracking_timeout',
        default_value='3.0',
        description='Max time (seconds) to attempt re-tracking before returning to IDLE'
    )

    declare_velocity_filter_alpha = DeclareLaunchArgument(
        'velocity_filter_alpha',
        default_value='0.3',
        description='Velocity filter smoothing factor (0.1-0.5, lower=more smooth)'
    )

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

    declare_person_conf = DeclareLaunchArgument(
        'person_conf_thresh',
        default_value='0.50',
        description='Person detection confidence threshold (lowered for grayscale robustness)'
    )

    declare_hand_conf = DeclareLaunchArgument(
        'hand_conf_thresh',
        default_value='0.35',
        description='Hand gesture confidence threshold (from params.yaml, allows weak detections)'
    )

    # ===== NODE 1: TRACKING NODE =====
    # Detects persons and hand gestures using YOLO + BoT-SORT tracking
    # Estimates 3D position using CTRV-EKF filtering
    tracking_node = Node(
        package='yolo',
        executable='tracking',
        name='tracking_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_tracking')),
        parameters=[{
            # --- ĐỒNG BỘ TOPIC D435i ---
            'camera_image_topic': '/camera/color/image_raw',
            'depth_topic': '/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/color/camera_info',
            
            # Model paths
            'person_model_path': 'yolov8n.pt',
            'hand_model_path': 'handsign.pt',
            
            # Hardware acceleration
            'use_cuda': LaunchConfiguration('use_cuda'),
            'use_fp16': LaunchConfiguration('use_fp16'),
            
            # Performance
            'max_inference_time': 200,
            'enable_frame_skip': True,
            'enable_performance_metrics': True,
            
            # NOTE: All detection thresholds, tracking parameters, and ROI settings
            # are loaded from config/params.yaml at runtime (NO HARDCODING)
        }],
        remappings=[
            ('/yolo/gesture_command', '/yolo/gesture_command'),
            ('/tracking/main_person', '/tracking/main_person'),
            ('/tracking/annotated_image', '/tracking/annotated_image'),
        ],
    )

    # ===== NODE 2: STATE MACHINE NODE =====
    # Manages robot state transitions based on hand gestures
    # Subscribes to gesture commands and 3D position estimates
    # Publishes control commands to robot controllers
    state_machine_node = Node(
        package='state_machine',
        executable='state_machine',
        name='state_machine_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_state_machine')),
        parameters=[{
            # State timeouts (detection lost, re-tracking)
            'tracking_timeout': LaunchConfiguration('tracking_timeout'),
            'retracking_timeout': LaunchConfiguration('retracking_timeout'),
            
            # Motion filtering
            'velocity_filter_alpha': LaunchConfiguration('velocity_filter_alpha'),
        }],
        remappings=[
            ('/yolo/gesture_command', '/yolo/gesture_command'),
            ('/tracking/main_person', '/tracking/main_person'),
            ('/robot/control_command', '/robot/control_command'),
            ('/robot/state', '/robot/state'),
        ],
    )

    # ===== RETURN LAUNCH DESCRIPTION =====
    return LaunchDescription([
        declare_enable_tracking,
        declare_enable_state_machine,
        declare_tracking_timeout,
        declare_retracking_timeout,
        declare_velocity_filter_alpha,
        declare_use_cuda,
        declare_use_fp16,
        declare_person_conf,
        declare_hand_conf,
        
        tracking_node,
        state_machine_node,
    ])
