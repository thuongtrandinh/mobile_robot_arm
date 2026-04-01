#!/usr/bin/env python3

import os

import xacro
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    use_sim_time_str = context.launch_configurations.get("use_sim_time", "true")
    use_sim_time = use_sim_time_str.lower() == "true"
    launch_rviz = context.launch_configurations.get("launch_rviz", "true").lower() == "true"
    start_hardware = context.launch_configurations.get("start_hardware", "true").lower() == "true"

    desc_share = get_package_share_directory("descriptions")
    zed2_share = get_package_share_directory("zed2")
    lidar_share = get_package_share_directory("lidar")
    xacro_file = os.path.join(desc_share, "model", "wheeled", "urdf", "mobile_robot.urdf.xacro")
    rviz_config_file = os.path.join(desc_share, "config", "rviz2.rviz")

    robot_description = xacro.process_file(
        xacro_file,
        mappings={"sim_mode": "true" if use_sim_time else "false"},
    ).toxml()

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": use_sim_time,
                "publish_frequency": 50.0,
            }
        ],
    )

    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": use_sim_time,
            }
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    nodes_to_launch = [robot_state_publisher, joint_state_publisher]

    if launch_rviz:
        nodes_to_launch.append(rviz_node)

    if not use_sim_time and start_hardware:
        zed2_hardware = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(zed2_share, "launch", "zed2.launch.py")),
            launch_arguments={
                "namespace": "",
                "camera_name": "zed2",
                "node_name": "zed_node",
                "use_sim_time": "false",
                "sim_mode": "false",
                "publish_svo_clock": "false",
            }.items(),
        )

        lidar_hardware = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(lidar_share, "launch", "a2m8.launch.py")),
            launch_arguments={
                "use_sim_time": "false",
                "frame_id": "laser",
            }.items(),
        )

        nodes_to_launch.extend([zed2_hardware, lidar_hardware])

    return nodes_to_launch


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use simulation clock",
                choices=["true", "false"],
            ),
            DeclareLaunchArgument(
                "launch_rviz",
                default_value="true",
                description="Launch RViz2",
                choices=["true", "false"],
            ),
            DeclareLaunchArgument(
                "start_hardware",
                default_value="true",
                description="If use_sim_time=false, auto-launch ZED2 and LiDAR drivers",
                choices=["true", "false"],
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
