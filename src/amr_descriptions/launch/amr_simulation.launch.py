#!/usr/bin/env python3

import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def launch_setup(context, *args, **kwargs):
    # ========== Configuration ==========
    package_name = 'amr_descriptions'
    use_sim_time_str = context.launch_configurations.get('use_sim_time', 'true')
    use_sim_time = use_sim_time_str.lower() == 'true'
    init_x = context.launch_configurations.get('x_pos', '0.0')
    init_y = context.launch_configurations.get('y_pos', '0.0')
    init_height = context.launch_configurations.get('height', '0.1')
    headless_str = context.launch_configurations.get('headless', 'false')
    headless = headless_str.lower() == 'true'
    spawn_controllers_str = context.launch_configurations.get('spawn_controllers', 'false')
    spawn_controllers = spawn_controllers_str.lower() == 'true'
    
    pkg_path = get_package_share_directory(package_name)
    xacro_file = os.path.join(pkg_path, 'model', 'wheeled', 'urdf', 'mobile_robot.urdf.xacro')
    world_file = os.path.join(pkg_path, 'worlds', 'amr_simulation.world')
    
    # Set Gazebo resource paths
    models_path = os.path.join(pkg_path, 'model')
    os.environ['GZ_SIM_RESOURCE_PATH'] = models_path + ':' + os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    
    print(f"✅ Xacro file: {xacro_file} (exists: {os.path.exists(xacro_file)})")
    print(f"✅ World file: {world_file} (exists: {os.path.exists(world_file)})")
    
    # Process xacro to URDF
    robot_description = xacro.process_file(xacro_file, mappings={'sim_mode': 'true'}).toxml()
    print(f"✅ URDF rendered: {len(robot_description)} characters")
    
    # ========== NODES ==========
    
    gz_args = ('-r -s -v 2 ' if headless else '-r -v 2 ') + world_file

    # 1. Gazebo (server + gui)
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ]),
        launch_arguments=[
            ('gz_args', gz_args),
            ('use_sim_time', use_sim_time_str),
        ]
    )
    
    # 3. Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time,
            'publish_frequency': 50.0,
        }],
        output='screen'
    )
    
    # 4. Spawn Robot in Gazebo
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-topic', 'robot_description', '-entity', 'mobile_robot',
                   '-x', init_x, '-y', init_y, '-z', init_height],
        parameters=[{'use_sim_time': use_sim_time}],
    )
    
    # 5. Controllers (optional)
    joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_broad', '--controller-manager', '/controller_manager'],
        parameters=[{'use_sim_time': use_sim_time}],
    )
    
    diff_drive_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['diff_cont', '--controller-manager', '/controller_manager'],
        parameters=[{'use_sim_time': use_sim_time}],
    )
    
    # 6. ROS-Gazebo Bridge for Sensors
    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
            '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/zed2/left/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/zed2/left/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/zed2/right/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/zed2/right/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    # 7. ZED2 Hand Sign Detector (Simulation)
    zed2_handsign_detector = Node(
        package='amr_zed2',
        executable='handsign_detector_sim',
        name='handsign_detector_sim',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'detection_conf': 0.5},
            {'enable_visualization': True},
            {'camera_frame': 'zed2_left_camera_frame'},
            {'global_frame': 'odom'},
            {'gesture_velocity_threshold': 0.1},
        ],
        output='screen'
    )

    # 8. Obstacle Detector Nodes
    obstacle_extractor = Node(
        package='obstacle_detector',
        executable='obstacle_extractor_node',
        remappings=[
            ('pcl2', '/camera/points'),
            ('raw_obstacles', '/raw_obstacles'),
            ('raw_obstacles_visualization', '/raw_obstacles_visualization'),
        ],
        parameters=[
            {'active': True},
            {'use_scan': False},
            {'use_pcl': False},
            {'use_pcl2': True},
            {'use_sim_time': use_sim_time},
            {'frame_id': 'map'},
            {'max_circle_radius': 0.6},
            {'use_split_and_merge': True},
        ],
        output='screen'
    )
    
    obstacle_tracker = Node(
        package='obstacle_detector',
        executable='obstacle_tracker_node',
        remappings=[
            ('raw_obstacles', '/raw_obstacles'),
            ('tracked_obstacles', '/obstacles'),
            ('tracked_obstacles_visualization', '/obstacles_visualization'),
        ],
        parameters=[
            {'active': True},
            {'use_sim_time': use_sim_time},
            {'frame_id': 'map'},
            {'loop_rate': 100.0},
            {'tracking_duration': 2.0},
        ],
        output='screen'
    )
    
    # Return all nodes
    nodes = [
        gazebo,
        robot_state_publisher,
        spawn_entity,
        bridge_node,
        zed2_handsign_detector,
        obstacle_extractor,
        obstacle_tracker,
    ]

    if spawn_controllers:
        nodes.append(
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=spawn_entity,
                    on_exit=[joint_state_broadcaster],
                )
            )
        )
        nodes.append(
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=joint_state_broadcaster,
                    on_exit=[diff_drive_controller],
                )
            )
        )

    return nodes


def generate_launch_description():
    # Declare arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation time'
    )
    
    x_pos_arg = DeclareLaunchArgument(
        'x_pos', default_value='0.0',
        description='Initial X position of robot'
    )
    
    y_pos_arg = DeclareLaunchArgument(
        'y_pos', default_value='0.0',
        description='Initial Y position of robot'
    )
    
    height_arg = DeclareLaunchArgument(
        'height', default_value='0.1',
        description='Initial height of robot'
    )

    headless_arg = DeclareLaunchArgument(
        'headless', default_value='false',
        description='Run Gazebo server only (no GUI) to avoid OpenGL GUI warnings'
    )

    spawn_controllers_arg = DeclareLaunchArgument(
        'spawn_controllers', default_value='false',
        description='Spawn ros2_control controllers after robot spawn'
    )
    
    # OpaqueFunction to setup launch
    launch_setup_func = OpaqueFunction(function=launch_setup)
    
    return LaunchDescription([
        use_sim_time_arg,
        x_pos_arg,
        y_pos_arg,
        height_arg,
        headless_arg,
        spawn_controllers_arg,
        launch_setup_func,
    ])
