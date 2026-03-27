#!/usr/bin/env python3

import os

import xacro
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    use_sim_time_str = context.launch_configurations.get("use_sim_time", "true")
    use_sim_time = use_sim_time_str.lower() == "true"

    desc_share = get_package_share_directory("descriptions")
    xacro_file = os.path.join(desc_share, "model", "wheeled", "urdf", "mobile_robot.urdf.xacro")

    robot_description = xacro.process_file(xacro_file, mappings={"sim_mode": "true"}).toxml()

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

    return [robot_state_publisher, joint_state_publisher]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use simulation clock",
                choices=["true", "false"],
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
