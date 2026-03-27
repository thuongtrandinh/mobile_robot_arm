"""
Launch file for AMR ZED2 Hand Sign Detector Node - SIMULATION VERSION
Works with Gazebo simulation using ROS topics instead of ZED SDK
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def launch_setup(context, *args, **kwargs):
    """Setup launch with resolved configurations"""
    # Get configuration values
    use_sim_time_str = context.launch_configurations.get('use_sim_time', 'true')
    use_sim_time = use_sim_time_str.lower() == 'true'
    detection_conf = float(context.launch_configurations.get('detection_conf', '0.5'))
    enable_viz_str = context.launch_configurations.get('enable_visualization', 'false')
    enable_viz = enable_viz_str.lower() == 'true'
    hand_model = context.launch_configurations.get('hand_model', 'handsign.pt')
    person_model = context.launch_configurations.get('person_model', 'yolov8n.pt')
    camera_frame = context.launch_configurations.get('camera_frame', 'zed2_left_camera_optical_frame')
    global_frame = context.launch_configurations.get('global_frame', 'odom')

    # Package directory
    pkg_dir = get_package_share_directory('yolo')
    config_file = os.path.join(pkg_dir, 'config', 'params_sim.yaml')

    print(f"✅ AMR ZED2 Sim - use_sim_time: {use_sim_time}")
    print(f"✅ AMR ZED2 Sim - detection_conf: {detection_conf}")
    print(f"✅ AMR ZED2 Sim - enable_visualization: {enable_viz}")

    # Hand sign detector simulation node
    handsign_sim_node = Node(
        package='yolo',
        executable='handsign_detector_sim',
        name='handsign_detector_sim',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'detection_conf': detection_conf},
            {'enable_visualization': enable_viz},
            {'hand_model_path': hand_model},
            {'person_model_path': person_model},
            {'camera_frame': camera_frame},
            {'global_frame': global_frame},
            {'person_radius': 0.3},
            {'nms_threshold': 0.45},
        ],
        remappings=[
            ('/zed2/left/image_raw', '/zed2/left/image_raw'),
            ('/zed2/left/camera_info', '/zed2/left/camera_info'),
            ('/camera/points', '/camera/points'),
        ],
    )

    return [handsign_sim_node]


def generate_launch_description():
    """Generate launch description with arguments"""

    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    detection_conf_arg = DeclareLaunchArgument(
        'detection_conf',
        default_value='0.5',
        description='Detection confidence threshold (0-1)'
    )

    enable_viz_arg = DeclareLaunchArgument(
        'enable_visualization',
        default_value='false',
        description='Enable OpenCV visualization window'
    )

    hand_model_arg = DeclareLaunchArgument(
        'hand_model',
        default_value='handsign.pt',
        description='Path to hand sign detection model'
    )

    person_model_arg = DeclareLaunchArgument(
        'person_model',
        default_value='yolov8n.pt',
        description='Path to person detection model'
    )

    camera_frame_arg = DeclareLaunchArgument(
        'camera_frame',
        default_value='zed2_left_camera_optical_frame',
        description='Camera optical frame name'
    )

    global_frame_arg = DeclareLaunchArgument(
        'global_frame',
        default_value='odom',
        description='Global reference frame'
    )

    # Launch setup function
    launch_setup_func = OpaqueFunction(function=launch_setup)

    return LaunchDescription([
        use_sim_time_arg,
        detection_conf_arg,
        enable_viz_arg,
        hand_model_arg,
        person_model_arg,
        camera_frame_arg,
        global_frame_arg,
        launch_setup_func,
    ])
