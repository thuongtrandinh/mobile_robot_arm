#!/usr/bin/env python3

import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
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
    if robot_type == 'roarm':
        package_name = 'amr_descriptions'
        pkg_path = os.path.join(get_package_share_directory(package_name))
        xacro_file = os.path.join(pkg_path, 'model', 'roarm_description', 'urdf', 'roarm.xacro')
        controller_config = os.path.join(pkg_path, 'model', 'roarm_description', 'config', 'roarm_controllers.yaml')
        rviz_config_file = os.path.join(pkg_path, 'model', 'roarm_description', 'config', 'roarm_description.rviz')
    else:  # mobile_robot
        package_name = 'amr_descriptions'
        pkg_path = os.path.join(get_package_share_directory(package_name))
        xacro_file = os.path.join(pkg_path, 'urdf', 'mobile_robot.urdf.xacro')
        controller_config = None  # Will use default ros2_control config
        rviz_config_file = os.path.join(pkg_path, 'rviz', 'rviz2.rviz')
    
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

    # RViz2 node (optional)
    rviz_node = None
    if launch_rviz and os.path.exists(rviz_config_file):
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

    # Gazebo launch
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            ])
        ])
    )

    # Controllers - only for roarm
    nodes_to_launch = [
        gazebo_launch,
        robot_state_publisher,
        spawn_entity,
    ]
    
    # Add RViz if requested
    if rviz_node is not None:
        nodes_to_launch.append(rviz_node)
    
    # Add controllers for roarm
    if robot_type == 'roarm':
        joint_state_broadcaster = Node(
            package="controller_manager",
            executable="spawner",
            arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
            parameters=[{'use_sim_time': use_sim_time}],
        )

        arm_controller = Node(
            package="controller_manager",
            executable="spawner",
            arguments=["arm_controller", "--controller-manager", "/controller_manager"],
            parameters=[{'use_sim_time': use_sim_time}],
        )
        
        nodes_to_launch.extend([
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=spawn_entity,
                    on_exit=[joint_state_broadcaster],
                )
            ),
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=joint_state_broadcaster,
                    on_exit=[arm_controller],
                )
            ),
        ])
    
    return nodes_to_launch


def generate_launch_description():
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    robot_type_arg = DeclareLaunchArgument(
        'robot_type',
        default_value='roarm',
        description='Type of robot to launch (roarm, mobile_robot)',
        choices=['roarm', 'mobile_robot']
    )

    launch_rviz_arg = DeclareLaunchArgument(
        'launch_rviz',
        default_value='false',
        description='Launch RViz2 for visualization'
    )

    return LaunchDescription([
        use_sim_time_arg,
        robot_type_arg,
        launch_rviz_arg,
        OpaqueFunction(function=launch_setup),
    ])
