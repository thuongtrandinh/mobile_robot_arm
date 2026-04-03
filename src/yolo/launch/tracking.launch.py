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

    declare_gesture_hold_time = DeclareLaunchArgument(
        'gesture_hold_time',
        default_value='0.8',
        description='Time (seconds) to hold gesture for confirmation (0.5-2.0)'
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
        default_value='0.85',
        description='Person detection confidence threshold'
    )

    declare_hand_conf_start = DeclareLaunchArgument(
        'hand_conf_start',
        default_value='0.35',
        description='START gesture confidence threshold (sensitive for small thumbs up)'
    )

    declare_hand_conf_stop = DeclareLaunchArgument(
        'hand_conf_stop',
        default_value='0.55',
        description='STOP gesture confidence threshold (conservative for palm)'
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
            # Image source
            'camera_image_topic': '/zed/zed_node/rgb/color/rect/image',
            'depth_topic': '/zed/zed_node/depth/depth_registered',
            
            # Model paths (relative to package share directory, auto-loaded)
            'person_model_path': 'yolov8n.pt',
            'hand_model_path': 'handsign.pt',
            
            # Confidence thresholds (OPTIMIZED)
            'person_conf_thresh': LaunchConfiguration('person_conf_thresh'),
            'hand_conf_start': LaunchConfiguration('hand_conf_start'),   # START gesture (0.35)
            'hand_conf_stop': LaunchConfiguration('hand_conf_stop'),     # STOP gesture (0.55)
            
            # YOLO settings
            'person_imgsz': 416,              # Person model input size
            'hand_imgsz': 640,                # Hand model input size
            
            # Hardware acceleration
            'use_cuda': LaunchConfiguration('use_cuda'),
            'use_fp16': LaunchConfiguration('use_fp16'),
            
            # Gesture recognition
            'gesture_hold_time': LaunchConfiguration('gesture_hold_time'),
            'gesture_temporal_buffer': 20,    # 20 frames for temporal voting (12/20 confirmation)
            
            # Tracking optimization
            'hand_person_match_buffer': 100,  # BoT-SORT track buffer
            'gesture_y_min_ratio': 0.10,      # Extended Y range: 10%-90%
            'gesture_y_max_ratio': 0.90,
            
            # Depth-Aware NMS (noise reduction for NEURAL depth mode)
            'depth_aware_nms_enabled': True,
            'nms_threshold': 0.60,            # Aggressive box merging
            'depth_threshold': 0.80,          # Relaxed for NEURAL depth noise
            
            # 3D estimation (CTRV-EKF + velocity filter)
            'ekf_enabled': True,
            'velocity_filter_alpha': LaunchConfiguration('velocity_filter_alpha'),
            'ekf_process_noise': 0.1,
            'ekf_measurement_noise': 0.5,
            
            # Performance
            'max_inference_time': 200,        # Max 200ms per frame
            'enable_frame_skip': True,        # ✅ Drop frames when GPU overloaded (GPU bottleneck fix)
            'enable_performance_metrics': True,
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
            # Gesture recognition
            'gesture_hold_time': LaunchConfiguration('gesture_hold_time'),
            
            # State timeouts
            'tracking_timeout': LaunchConfiguration('tracking_timeout'),
            'retracking_timeout': LaunchConfiguration('retracking_timeout'),
            
            # Motion filtering
            'velocity_filter_alpha': LaunchConfiguration('velocity_filter_alpha'),
            
            # Subscribers (match tracking_node publishers)
            'gesture_topic': '/yolo/gesture_command',
            'human_state_topic': '/tracking/main_person',
            
            # Publishers (for robot control)
            'control_command_topic': '/robot/control_command',
            'state_topic': '/robot/state',
            'debug_enabled': True,
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
        declare_gesture_hold_time,
        declare_tracking_timeout,
        declare_retracking_timeout,
        declare_velocity_filter_alpha,
        declare_use_cuda,
        declare_use_fp16,
        declare_person_conf,
        declare_hand_conf_start,
        declare_hand_conf_stop,
        
        tracking_node,
        state_machine_node,
    ])
