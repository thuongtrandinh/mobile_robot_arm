#!/usr/bin/env python3

import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def launch_setup(context, *args, **kwargs):
    # Get launch configurations
    use_sim_time_str = context.launch_configurations['use_sim_time']
    use_sim_time = use_sim_time_str.lower() == 'true'
    robot_type = context.launch_configurations['robot_type']
    launch_rviz_str = context.launch_configurations['launch_rviz']
    launch_rviz = launch_rviz_str.lower() == 'true'
    
    # Determine package and xacro file based on robot type
    package_name = 'amr_descriptions'
    pkg_path = os.path.join(get_package_share_directory(package_name))
    
    # Common RViz config for all robots
    rviz_config_file = os.path.join(pkg_path, 'config', 'rviz2.rviz')
    
    xacro_file = os.path.join(pkg_path, 'model', 'wheeled', 'urdf', 'mobile_robot.urdf.xacro')
    controller_config = None  # Will use default ros2_control config
    
    # Debug: Print paths for verification
    print(f"✅ Package: {package_name}")
    print(f"✅ Robot type: {robot_type}")
    print(f"✅ Xacro file: {xacro_file}")
    print(f"✅ Xacro exists: {os.path.exists(xacro_file)}")
    
    # Robot description - use Command for proper xacro processing
    robot_description = ParameterValue(
        Command(['xacro ', xacro_file]),
        value_type=str
    )
    
    print(f"✅ Launch RViz: {launch_rviz}")
    print(f"✅ RViz config file: {rviz_config_file}")
    print(f"✅ RViz config exists: {os.path.exists(rviz_config_file)}")

    # RViz2 node (optional)
    rviz_node = None
    if launch_rviz and os.path.exists(rviz_config_file):
        print(f"✅ Creating RViz node...")
        rviz_node = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=["-d", rviz_config_file],
            parameters=[{'use_sim_time': use_sim_time}]
        )
    elif launch_rviz:
        print(f"❌ RViz config file not found: {rviz_config_file}")
    else:
        print(f"❌ RViz launch disabled")

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time,
        }],
        output='screen'
    )

    # Gazebo spawn entity
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-entity', robot_type,
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Load and start joint_state_broadcaster
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_broadcaster'],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Load and start diff_drive_controller
    diff_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['diff_controller'],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Bridge for camera images from Gazebo to ROS2
    camera_left_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/zed2/left/image_raw@sensor_msgs/msg/Image@gz.msgs.Image',
            '/zed2/left/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    camera_right_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/zed2/right/image_raw@sensor_msgs/msg/Image@gz.msgs.Image',
            '/zed2/right/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Event handler to start controllers after robot spawns
    load_joint_state_broadcaster = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[joint_state_broadcaster_spawner],
        )
    )

    load_diff_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[diff_controller_spawner],
        )
    )

    # Gazebo launch
    world_file = os.path.join(pkg_path, 'worlds', 'empty.world')
    
    # Set Gazebo model path to include the package models
    gazebo_model_path = os.path.join(pkg_path, 'model')
    if 'GAZEBO_MODEL_PATH' in os.environ:
        os.environ['GAZEBO_MODEL_PATH'] = gazebo_model_path + ':' + os.environ['GAZEBO_MODEL_PATH']
    else:
        os.environ['GAZEBO_MODEL_PATH'] = gazebo_model_path
    
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ]),
        launch_arguments={
            'gz_args': f'-r -v 4 {world_file}' if os.path.exists(world_file) else '-r -v 4',
        }.items()
    )

    # Controllers - only for roarm
    nodes_to_launch = [
        gazebo_launch,
        robot_state_publisher,
        spawn_entity,
        load_joint_state_broadcaster,
        load_diff_controller,
        camera_left_bridge,
        camera_right_bridge,
    ]
    
    # Add RViz if requested
    if rviz_node is not None:
        nodes_to_launch.append(rviz_node)
    
    return nodes_to_launch


def generate_launch_description():
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    robot_type_arg = DeclareLaunchArgument(
        'robot_type',
        default_value='mobile_robot',
        description='Type of robot to launch (mobile_robot)',
        choices=['mobile_robot']
    )

    launch_rviz_arg = DeclareLaunchArgument(
        'launch_rviz',
        default_value='true',
        description='Launch RViz2 for visualization'
    )
    
    # Set GZ_SIM_RESOURCE_PATH to include package models
    pkg_share = get_package_share_directory('amr_descriptions')
    model_path = os.path.join(pkg_share, 'model')
    
    gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=model_path
    )

    return LaunchDescription([
        use_sim_time_arg,
        robot_type_arg,
        launch_rviz_arg,
        gz_resource_path,
        OpaqueFunction(function=launch_setup),
    ])
