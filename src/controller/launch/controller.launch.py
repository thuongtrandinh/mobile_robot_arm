import os
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare
from nav2_common.launch import RewrittenYaml

def generate_launch_description():
    controller_prefix = get_package_prefix("controller")
    workspace_root = os.path.dirname(os.path.dirname(controller_prefix))
    custom_mppi_prefix = os.path.join(workspace_root, "install", "nav2_mppi_controller")
    custom_mppi_lib = os.path.join(custom_mppi_prefix, "lib")

    use_sim_time = LaunchConfiguration("use_sim_time")
    respawn = LaunchConfiguration("respawn")
    log_level = LaunchConfiguration("log_level")
    use_rviz = LaunchConfiguration("use_rviz")
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
        "use_sim_time", default_value="false", description="Use simulation time if true"
    )
    respawn_arg = DeclareLaunchArgument(
        "respawn", default_value="false", description="Respawn node when it crashes"
    )
    log_level_arg = DeclareLaunchArgument(
        "log_level", default_value="info", description="Logging level"
    )
    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="false",
        description="Launch RViz2 with the descriptions RViz config",
        choices=["true", "false"],
    )
    tuning_config_arg = DeclareLaunchArgument(
        "tuning_config",
        default_value=PathJoinSubstitution([FindPackageShare("controller"), "config", "planner_config.yaml"]),
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
    action_marker_frame_arg = DeclareLaunchArgument("action_marker_frame", default_value="map")
    publish_debug_joint_state_arg = DeclareLaunchArgument("publish_debug_joint_state", default_value="true")
    debug_joint_state_topic_arg = DeclareLaunchArgument("debug_joint_state_topic", default_value="/debug/joint_state_req")
    publish_policy_debug_status_arg = DeclareLaunchArgument("publish_policy_debug_status", default_value="true")
    policy_debug_status_topic_arg = DeclareLaunchArgument("policy_debug_status_topic", default_value="/debug/policy_status")
    planner_scene_marker_topic_arg = DeclareLaunchArgument("planner_scene_marker_topic", default_value="/planner/debug_markers")
    planner_scene_frame_arg = DeclareLaunchArgument("planner_scene_frame", default_value="map")
    visualize_local_costmap_arg = DeclareLaunchArgument("visualize_local_costmap", default_value="true")
    local_costmap_topic_arg = DeclareLaunchArgument("local_costmap_topic", default_value="/a_star/local_costmap")
    astar_local_map_topic_arg = DeclareLaunchArgument("astar_local_map_topic", default_value="/map")

    # 1. GỌI NAV2 NODES (CHẠY MPPI)
    #
    # Do not use a parent/global cmd_vel remap here. A global remap also rewires
    # behavior_server into the wheel controller, which creates conflicting
    # velocity commands when recoveries run. Keep controller_server -> smoother
    # on cmd_vel_nav, and only send the smoother output to diff_cont.
    nav2_remappings = [
        ("/tf", "tf"),
        ("/tf_static", "tf_static"),
        ("odom", "/odometry/filtered"),
    ]
    nav2_params = ParameterFile(
        RewrittenYaml(
            source_file=mppi_params_file,
            param_rewrites={"use_sim_time": use_sim_time, "autostart": "true"},
            convert_types=True,
        ),
        allow_substs=True,
    )

    nav2_buffered_logging = SetEnvironmentVariable(
        "RCUTILS_LOGGING_BUFFERED_STREAM", "1"
    )
    custom_mppi_ament_prefix = SetEnvironmentVariable(
        "AMENT_PREFIX_PATH",
        os.pathsep.join(
            path
            for path in [custom_mppi_prefix, os.environ.get("AMENT_PREFIX_PATH", "")]
            if path
        ),
    )
    custom_mppi_cmake_prefix = SetEnvironmentVariable(
        "CMAKE_PREFIX_PATH",
        os.pathsep.join(
            path
            for path in [custom_mppi_prefix, os.environ.get("CMAKE_PREFIX_PATH", "")]
            if path
        ),
    )
    custom_mppi_library_path = SetEnvironmentVariable(
        "LD_LIBRARY_PATH",
        os.pathsep.join(
            path
            for path in [custom_mppi_lib, os.environ.get("LD_LIBRARY_PATH", "")]
            if path
        ),
    )

    controller_server = Node(
        package="nav2_controller",
        executable="controller_server",
        output="screen",
        respawn=respawn,
        respawn_delay=2.0,
        parameters=[nav2_params],
        arguments=["--ros-args", "--log-level", log_level],
        remappings=nav2_remappings + [("cmd_vel", "cmd_vel_nav")],
    )

    smoother_server = Node(
        package="nav2_smoother",
        executable="smoother_server",
        name="smoother_server",
        output="screen",
        respawn=respawn,
        respawn_delay=2.0,
        parameters=[nav2_params],
        arguments=["--ros-args", "--log-level", log_level],
        remappings=nav2_remappings,
    )

    planner_server = Node(
        package="nav2_planner",
        executable="planner_server",
        name="planner_server",
        output="screen",
        respawn=respawn,
        respawn_delay=2.0,
        parameters=[nav2_params],
        arguments=["--ros-args", "--log-level", log_level],
        remappings=nav2_remappings,
    )

    behavior_server = Node(
        package="nav2_behaviors",
        executable="behavior_server",
        name="behavior_server",
        output="screen",
        respawn=respawn,
        respawn_delay=2.0,
        parameters=[nav2_params],
        arguments=["--ros-args", "--log-level", log_level],
        remappings=nav2_remappings + [("cmd_vel", "behavior_cmd_vel")],
    )

    bt_navigator = Node(
        package="nav2_bt_navigator",
        executable="bt_navigator",
        name="bt_navigator",
        output="screen",
        respawn=respawn,
        respawn_delay=2.0,
        parameters=[nav2_params],
        arguments=["--ros-args", "--log-level", log_level],
        remappings=nav2_remappings,
    )

    waypoint_follower = Node(
        package="nav2_waypoint_follower",
        executable="waypoint_follower",
        name="waypoint_follower",
        output="screen",
        respawn=respawn,
        respawn_delay=2.0,
        parameters=[nav2_params],
        arguments=["--ros-args", "--log-level", log_level],
        remappings=nav2_remappings,
    )

    velocity_smoother = Node(
        package="nav2_velocity_smoother",
        executable="velocity_smoother",
        name="velocity_smoother",
        output="screen",
        respawn=respawn,
        respawn_delay=2.0,
        parameters=[nav2_params],
        arguments=["--ros-args", "--log-level", log_level],
        remappings=nav2_remappings
        + [
            ("cmd_vel", "cmd_vel_nav"),
            ("cmd_vel_smoothed", "/diff_cont/cmd_vel_unstamped"),
        ],
    )

    lifecycle_manager_navigation = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_navigation",
        output="screen",
        arguments=["--ros-args", "--log-level", log_level],
        parameters=[
            {"use_sim_time": use_sim_time},
            {"autostart": True},
            {
                "node_names": [
                    "controller_server",
                    "smoother_server",
                    "planner_server",
                    "behavior_server",
                    "bt_navigator",
                    "waypoint_follower",
                    "velocity_smoother",
                ]
            },
        ],
    )

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
                "policy_hz": 5.0,
                "service_hz": 5.0,
                "follow_path_replan_min_interval_sec": 0.80,
                "follow_path_replan_path_delta": 0.35,
                "tf_timeout_sec": 0.10,
                "obstacle_sample_step": 16,
                "scan_filter_enabled": True,
                "scan_obstacle_max_range": 3.0,
                "scan_neighbor_window": 2,
                "scan_min_neighbor_count": 2,
                "scan_neighbor_max_delta": 0.18,
                "scan_persistence_hits": 2,
                "scan_persistence_decay_scans": 4,
                "scan_persistence_resolution": 0.12,
                "scan_obstacle_limit": 60,
                "use_map_static_obstacles": True,
                "map_static_sample_step_m": 0.20,
                "map_static_obstacle_limit": 300,
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
                "visualize_scan_obstacles": True,
                "visualize_joint_state_geometry": True,
                "visualize_astar_path": True,
                "visualize_astar_local_map": True,
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
                "debug_marker_lifetime_sec": 0.25,
            }
        ],
        respawn=respawn,
        arguments=["--ros-args", "--log-level", log_level],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        condition=IfCondition(use_rviz),
        arguments=[
            "-d",
            PathJoinSubstitution([
                FindPackageShare("descriptions"),
                "config",
                "rviz2.rviz",
            ]),
        ],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    return LaunchDescription([
        use_sim_time_arg,
        respawn_arg,
        log_level_arg,
        use_rviz_arg,
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
        
        nav2_buffered_logging,
        custom_mppi_ament_prefix,
        custom_mppi_cmake_prefix,
        custom_mppi_library_path,
        controller_server,
        smoother_server,
        planner_server,
        behavior_server,
        bt_navigator,
        waypoint_follower,
        velocity_smoother,
        lifecycle_manager_navigation,
        ampcc_node,
        rl_bridge_node,
        rviz_visualizer_node,
        rviz_node,
    ])
