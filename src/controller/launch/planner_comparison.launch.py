#!/usr/bin/python3
"""
Local Planner Comparison Framework - Launch File (Revised)

This launch file correctly uses Nav2's Action Server architecture instead of custom topic-based planners.
It switches between DWA, TEB, and AMPCC by loading different YAML configuration files.

Architecture:
- Nav2's controller_server runs as an Action Server (/follow_path)
- bt_navigator sends requests to this server
- Planner implementation is selected via params_X.yaml configuration
- All planners use synchronized kinematic constraints for fair comparison
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('controller')
    
    # ==================== Launch Arguments ====================
    planner_arg = DeclareLaunchArgument(
        'planner',
        default_value='rl_mppi',
        description='Local planner to use: dwa, teb, or rl_mppi',
        choices=['dwa', 'teb', 'rl_mppi']
    )

    enable_comparison_arg = DeclareLaunchArgument(
        'enable_comparison',
        default_value='true',
        description='Enable planner comparison metrics collection'
    )

    planner_config = LaunchConfiguration('planner')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # ==================== Launch Nav2 Controller with Selected Config ====================
    
    # 1. DWA (Gọi baseline, KHÔNG ĐỤNG RL)
    nav2_dwa_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_dir, 'launch', 'baseline_controller.launch.py')),
        condition=IfCondition(PythonExpression(["'", planner_config, "' == 'dwa'"])),
        launch_arguments={
            'mppi_params_file': os.path.join(pkg_dir, 'config', 'params_dwa.yaml'),
            'use_sim_time': use_sim_time
        }.items()
    )

    # 2. TEB (Gọi baseline, KHÔNG ĐỤNG RL)
    nav2_teb_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_dir, 'launch', 'baseline_controller.launch.py')),
        condition=IfCondition(PythonExpression(["'", planner_config, "' == 'teb'"])),
        launch_arguments={
            'mppi_params_file': os.path.join(pkg_dir, 'config', 'params_teb.yaml'),
            'use_sim_time': use_sim_time
        }.items()
    )

    # 3. RL-MPPI CỦA BẠN (Gọi code gốc, giữ nguyên hiện trạng 100%)
    nav2_ampcc_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_dir, 'launch', 'controller.launch.py')),
        condition=IfCondition(PythonExpression(["'", planner_config, "' == 'rl_mppi'"])),
        launch_arguments={
            'mppi_params_file': os.path.join(pkg_dir, 'config', 'mppi_params.yaml'),
            'use_sim_time': use_sim_time
        }.items()
    )

    # ==================== Comparison Metrics Node ====================
    # Node ghi dữ liệu (đã sửa lỗi crash)
    comparison_node = Node(
        package='controller',
        executable='planner_comparison.py',
        name='planner_comparison',
        condition=IfCondition(LaunchConfiguration('enable_comparison')),
        parameters=[{
            'test_duration': 120.0,
            'active_planner': planner_config,
            'data_dir': './planner_comparison_data',
        }],
        remappings=[
            ('odom', '/odometry/filtered'),
            ('global_path', '/plan'),
            ('scan', '/scan'),
        ],
        output='screen'
    )

    # ==================== Create Launch Description ====================
    return LaunchDescription([
        planner_arg,
        enable_comparison_arg,
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        
        # Include Nav2 controller with appropriate config
        nav2_dwa_launch,
        nav2_teb_launch,
        nav2_ampcc_launch,
        
        # Launch metrics collector
        comparison_node,
    ])
