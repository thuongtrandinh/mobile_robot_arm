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


def launch_setup(context, *args, **kwargs):
    package_name = 'amr_descriptions'
    use_sim_time_str = context.launch_configurations['use_sim_time']
    use_sim_time = use_sim_time_str.lower() == 'true'
    init_x = context.launch_configurations['x_pos']
    init_y = context.launch_configurations['y_pos']
    init_height = context.launch_configurations['height']
    world_name = context.launch_configurations['world']
    launch_rviz_str = context.launch_configurations['launch_rviz']
    launch_rviz = launch_rviz_str.lower() == 'true'

    pkg_path = get_package_share_directory(package_name)
    xacro_file = os.path.join(pkg_path, 'model', 'wheeled', 'urdf', 'mobile_robot.urdf.xacro')

    # Set Gazebo resource paths
    models_path = os.path.join(pkg_path, 'model')
    os.environ['GZ_SIM_RESOURCE_PATH'] = models_path + ':' + os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    os.environ['IGN_GAZEBO_RESOURCE_PATH'] = models_path + ':' + os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')

    # World file
    world_file = os.path.join(pkg_path, 'worlds', world_name)

    # Debug
    print(f"✅ Package path: {pkg_path}")
    print(f"✅ Xacro file: {xacro_file} (exists: {os.path.exists(xacro_file)})")
    print(f"✅ World file: {world_file} (exists: {os.path.exists(world_file)})")

    # Process xacro to URDF
    robot_description = xacro.process_file(xacro_file, mappings={'sim_mode': 'true'}).toxml()
    print(f"✅ URDF rendered: {len(robot_description)} characters")

    # RViz config
    rviz_config_file = os.path.join(pkg_path, 'config', 'rviz2.rviz')

    # RViz2 node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=["-d", rviz_config_file],
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Robot State Publisher
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

    # Gazebo spawn entity
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-topic', 'robot_description', '-entity', 'mobile_robot',
                   '-x', init_x, '-y', init_y, '-z', init_height],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # Controllers
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

    # Single bridge node for all topics
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
            '/zed2/left/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/zed2/left/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/zed2/right/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/zed2/right/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    obstacle_extractor_node = Node(
        package='obstacle_detector',
        executable='obstacle_extractor_node',
        name='obstacle_extractor',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'active': True,
            'use_scan': True,
            'use_pcl': False,
            'use_pcl2': False,
            'transform_coordinates': False,
            'frame_id': 'laser',
        }]
    )

    # Gazebo launch
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ]),
        launch_arguments=[
            ('gz_args', '-r -v 4 ' + world_file),
            ('use_sim_time', 'true'),
        ]
    )

    # Build launch list
    nodes_to_launch = [
        gazebo_launch,
        bridge,
        obstacle_extractor_node,
        robot_state_publisher,
        spawn_entity,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_entity,
                on_exit=[joint_state_broadcaster],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=joint_state_broadcaster,
                on_exit=[diff_drive_controller],
            )
        ),
    ]

    if launch_rviz:
        nodes_to_launch.append(rviz_node)

    return nodes_to_launch


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation time'),
        DeclareLaunchArgument('x_pos', default_value='0.0', description='Initial X position'),
        DeclareLaunchArgument('y_pos', default_value='0.0', description='Initial Y position'),
        DeclareLaunchArgument('height', default_value='0.2', description='Initial spawn height'),
        DeclareLaunchArgument('launch_rviz', default_value='true', description='Launch RViz2'),
        DeclareLaunchArgument('world', default_value='room_20x20.world',
                              description='World file to load',
                              choices=['empty.world', 'room_20x20.world', 'small_house.world', 'small_warehouse.world']),
        OpaqueFunction(function=launch_setup),
    ])
