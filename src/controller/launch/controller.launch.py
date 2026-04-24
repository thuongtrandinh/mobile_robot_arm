import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter, SetRemap
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    respawn = LaunchConfiguration("respawn")
    log_level = LaunchConfiguration("log_level")
    tuning_config = LaunchConfiguration("tuning_config")
    halo_drl_dir = LaunchConfiguration("halo_drl_dir")
    config = LaunchConfiguration("config")
    model_path = LaunchConfiguration("model_path")
    policy_env_name = LaunchConfiguration("policy_env_name")
    
    visualize_actions = LaunchConfiguration("visualize_actions")
    action_marker_topic = LaunchConfiguration("action_marker_topic")
    action_marker_frame = LaunchConfiguration("action_marker_frame")
    publish_debug_joint_state = LaunchConfiguration("publish_debug_joint_state")
    debug_joint_state_topic = LaunchConfiguration("debug_joint_state_topic")
    publish_policy_debug_status = LaunchConfiguration("publish_policy_debug_status")
    policy_debug_status_topic = LaunchConfiguration("policy_debug_status_topic")
    planner_scene_marker_topic = LaunchConfiguration("planner_scene_marker_topic")
    planner_scene_frame = LaunchConfiguration("planner_scene_frame")
    visualize_local_costmap = LaunchConfiguration("visualize_local_costmap")
    local_costmap_topic = LaunchConfiguration("local_costmap_topic")
    astar_local_map_topic = LaunchConfiguration("astar_local_map_topic")

    mppi_params_file = LaunchConfiguration("mppi_params_file")

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="true", description="Use simulation time if true"
    )
    respawn_arg = DeclareLaunchArgument(
        "respawn", default_value="false", description="Respawn node when it crashes"
    )
    log_level_arg = DeclareLaunchArgument(
        "log_level", default_value="info", description="Logging level"
    )
    tuning_config_arg = DeclareLaunchArgument(
        "tuning_config",
        default_value=PathJoinSubstitution([FindPackageShare("controller"), "config", "ddmr_mpc_config.yaml"]),
    )
    mppi_params_file_arg = DeclareLaunchArgument(
        "mppi_params_file",
        default_value=PathJoinSubstitution([FindPackageShare("controller"), "config", "mppi_params.yaml"]),
    )
    halo_drl_dir_arg = DeclareLaunchArgument(
        "halo_drl_dir", default_value="HALO_1/drl_moudle"
    )
    config_arg = DeclareLaunchArgument(
        "config", default_value="configs/mpc_rl.py"
    )
    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value=PathJoinSubstitution([FindPackageShare("controller"), "policy", "best_model.zip"]),
    )
    policy_env_name_arg = DeclareLaunchArgument(
        "policy_env_name", default_value="mpc_rl"
    )

    visualize_actions_arg = DeclareLaunchArgument("visualize_actions", default_value="true")
    action_marker_topic_arg = DeclareLaunchArgument("action_marker_topic", default_value="/policy/action_markers")
    action_marker_frame_arg = DeclareLaunchArgument("action_marker_frame", default_value="base_link")
    publish_debug_joint_state_arg = DeclareLaunchArgument("publish_debug_joint_state", default_value="true")
    debug_joint_state_topic_arg = DeclareLaunchArgument("debug_joint_state_topic", default_value="/debug/joint_state_req")
    publish_policy_debug_status_arg = DeclareLaunchArgument("publish_policy_debug_status", default_value="true")
    policy_debug_status_topic_arg = DeclareLaunchArgument("policy_debug_status_topic", default_value="/debug/policy_status")
    planner_scene_marker_topic_arg = DeclareLaunchArgument("planner_scene_marker_topic", default_value="/planner/debug_markers")
    planner_scene_frame_arg = DeclareLaunchArgument("planner_scene_frame", default_value="map")
    visualize_local_costmap_arg = DeclareLaunchArgument("visualize_local_costmap", default_value="true")
    local_costmap_topic_arg = DeclareLaunchArgument("local_costmap_topic", default_value="/a_star/local_costmap")
    astar_local_map_topic_arg = DeclareLaunchArgument("astar_local_map_topic", default_value="/map")

    # 1. GỌI NAV2 BRINGUP (CHẠY MPPI)
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    nav2_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': mppi_params_file,
            'autostart': 'true'
        }.items()
    )

    nav2_group = GroupAction([
        SetParameter(name='use_sim_time', value=use_sim_time),
        SetRemap(src='cmd_vel', dst='/diff_cont/cmd_vel_unstamped'),
        SetRemap(src='cmd_vel_nav', dst='/diff_cont/cmd_vel_unstamped'),
        nav2_cmd
    ])

    # 2. GỌI PLANNER NODE (Chỉ giữ A* smooth & local path service)
    ampcc_node = Node(
        package="controller",
        executable="ampcc_node",
        name="opt_planner",
        output="screen",
        parameters=[tuning_config, {"use_sim_time": use_sim_time}],
        respawn=respawn,
        arguments=["--ros-args", "--log-level", log_level],
    )

    # 3. GỌI RL BRIDGE (RL local goal -> A* service -> FollowPath/MPPI)
    rl_bridge_node = Node(
        package="controller",
        executable="rl_ocp_policy_bridge.py",
        name="rl_ocp_policy_bridge",
        output="screen",
        parameters=[
            tuning_config,
            {
                "use_sim_time": use_sim_time,
                "halo_drl_dir": halo_drl_dir,
                "config": config,
                "model_path": model_path,
                "policy_env_name": policy_env_name,
                "odom_topic": "/odometry/filtered",
                "scan_topic": "/scan",
                "goal_topic": "/goal_pose",
                "map_topic": "/map",
                "cmd_topic": "/diff_cont/cmd_vel",
                "planner_service": "/ocp_plann",
                "follow_path_action": "/follow_path",
                "controller_id": "FollowPath",
                "goal_checker_id": "general_goal_checker",
                "planner_half_width": 5.8,
                "planner_half_height": 9.8,
                "auto_relax_constraints": False,
                "visualize_actions": visualize_actions,
                "action_marker_topic": action_marker_topic,
                "action_marker_frame": action_marker_frame,
                "publish_debug_joint_state": publish_debug_joint_state,
                "debug_joint_state_topic": debug_joint_state_topic,
                "publish_policy_debug_status": publish_policy_debug_status,
                "policy_debug_status_topic": policy_debug_status_topic,
                "planner_scene_marker_topic": planner_scene_marker_topic,
                "planner_scene_frame": planner_scene_frame,
                "local_goal_topic": "/rl/local_goal",
                "local_path_topic": "/rl/local_path",
            }
        ],
        respawn=respawn,
        arguments=["--ros-args", "--log-level", log_level],
    )

    # 4. GỌI RVIZ VISUALIZER (Hiển thị các markers)
    rviz_visualizer_node = Node(
        package="controller",
        executable="ampcc_rviz_visualizer.py",
        name="ampcc_rviz_visualizer",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "visualize_actions": visualize_actions,
                "visualize_planner_scene": True,
                "visualize_local_costmap": visualize_local_costmap,
                "visualize_scan_obstacles": False,
                "visualize_joint_state_geometry": False,
                "visualize_astar_path": True,
                "visualize_astar_local_map": False,
                "action_marker_topic": action_marker_topic,
                "action_marker_frame": action_marker_frame,
                "planner_scene_marker_topic": planner_scene_marker_topic,
                "planner_scene_frame": planner_scene_frame,
                "map_topic": "/map",
                "scan_topic": "/scan",
                "joint_state_topic": debug_joint_state_topic,
                "astar_path_topic": "/a_star_path",
                "astar_local_map_topic": astar_local_map_topic,
                "local_costmap_topic": local_costmap_topic,
                "local_costmap_sample_step": 4,
                "local_costmap_max_cells": 3000,
                "local_costmap_min_cost": 10,
                "astar_local_map_sample_step": 4,
                "astar_local_map_max_cells": 2500,
                "astar_local_map_min_cost": 10,
            }
        ],
        respawn=respawn,
        arguments=["--ros-args", "--log-level", log_level],
    )

    return LaunchDescription([
        use_sim_time_arg,
        respawn_arg,
        log_level_arg,
        tuning_config_arg,
        mppi_params_file_arg,
        halo_drl_dir_arg,
        config_arg,
        model_path_arg,
        policy_env_name_arg,
        visualize_actions_arg,
        action_marker_topic_arg,
        action_marker_frame_arg,
        publish_debug_joint_state_arg,
        debug_joint_state_topic_arg,
        publish_policy_debug_status_arg,
        policy_debug_status_topic_arg,
        planner_scene_marker_topic_arg,
        planner_scene_frame_arg,
        visualize_local_costmap_arg,
        local_costmap_topic_arg,
        astar_local_map_topic_arg,
        
        nav2_group,
        ampcc_node,
        rl_bridge_node,
        rviz_visualizer_node,
    ])
