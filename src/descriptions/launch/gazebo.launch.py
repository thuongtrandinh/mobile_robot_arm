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
    package_name = 'descriptions'
    use_sim_time_str = context.launch_configurations.get('use_sim_time', 'true')
    use_sim_time = use_sim_time_str.lower() == 'true'
    init_x = context.launch_configurations.get('x_pos', '0.0')
    init_y = context.launch_configurations.get('y_pos', '0.0')
    init_height = context.launch_configurations.get('height', '0.1')
    world_name = context.launch_configurations.get('world', 'amr_simulation.world')
    launch_rviz_str = context.launch_configurations.get('launch_rviz', 'true')
    launch_rviz = launch_rviz_str.lower() == 'true'
    headless = context.launch_configurations.get('headless', 'false').lower() == 'true'
    spawn_controllers = context.launch_configurations.get('spawn_controllers', 'true').lower() == 'true'

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
    # Tên topic Depth trong Gazebo của bạn có thể khác một chút (tùy vào URDF).
    # Thường nó sẽ là '/zed2/depth_image' hoặc '/zed2/depth/image_raw'.
    # Giả sử ở đây Gazebo phát ra là '/zed2/depth_image'
    
   # Single bridge node for all topics
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
            
            # 1. Bắt đúng tên topic RGB từ Gazebo
            '/zed2/left/image@sensor_msgs/msg/Image[gz.msgs.Image',
            
            # 2. Bắt đúng tên topic Camera Info từ Gazebo
            '/zed2/left/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            
            # 3. Bắt đúng tên topic Depth từ Gazebo
            '/zed2/left/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
        ],
        remappings=[
            # Đổi nhãn (Remap) sang đúng tên mà RTAB-Map đang há miệng chờ
            ('/zed2/left/image', '/zed2/zed_node/rgb/color/rect/image'),
            ('/zed2/left/camera_info', '/zed2/zed_node/rgb/color/rect/camera_info'),
            ('/zed2/left/depth_image', '/zed2/zed_node/depth/depth_registered'),
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Gazebo launch
    gz_args = ('-r -s -v 4 ' if headless else '-r -v 4 ') + world_file
    gazebo_launch = IncludeLaunchDescription(
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

    # Build launch list
    nodes_to_launch = [
        gazebo_launch,
        bridge,
        robot_state_publisher,
        spawn_entity,
    ]

    if spawn_controllers:
        nodes_to_launch.append(
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=spawn_entity,
                    on_exit=[joint_state_broadcaster],
                )
            )
        )
        nodes_to_launch.append(
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=joint_state_broadcaster,
                    on_exit=[diff_drive_controller],
                )
            )
        )


    if launch_rviz:
        nodes_to_launch.append(rviz_node)

    return nodes_to_launch


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation time'),
        DeclareLaunchArgument('x_pos', default_value='0.0', description='Initial X position'),
        DeclareLaunchArgument('y_pos', default_value='0.0', description='Initial Y position'),
        DeclareLaunchArgument('height', default_value='0.1', description='Initial spawn height'),
        DeclareLaunchArgument('launch_rviz', default_value='true', description='Launch RViz2'),
        DeclareLaunchArgument('headless', default_value='false', description='Run Gazebo in server-only mode'),
        DeclareLaunchArgument('spawn_controllers', default_value='true', description='Spawn ros2_control controllers'),
        DeclareLaunchArgument('world', default_value='room_20x20.world',
                              description='World file to load',
                              choices=['amr_simulation.world', 'empty.world', 'room_20x20.world', 'small_house.world', 'small_warehouse.world']),
        OpaqueFunction(function=launch_setup),
    ])
