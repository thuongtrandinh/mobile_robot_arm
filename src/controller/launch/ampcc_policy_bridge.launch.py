from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    respawn = LaunchConfiguration("respawn")
    log_level = LaunchConfiguration("log_level")
    halo_drl_dir = LaunchConfiguration("halo_drl_dir")
    config = LaunchConfiguration("config")
    model_path = LaunchConfiguration("model_path")
    policy_env_name = LaunchConfiguration("policy_env_name")
    use_wall_constraints = LaunchConfiguration("use_wall_constraints")
    use_demo_poly_obstacle = LaunchConfiguration("use_demo_poly_obstacle")
    visualize_actions = LaunchConfiguration("visualize_actions")
    action_marker_topic = LaunchConfiguration("action_marker_topic")
    action_marker_frame = LaunchConfiguration("action_marker_frame")
    publish_debug_joint_state = LaunchConfiguration("publish_debug_joint_state")
    debug_joint_state_topic = LaunchConfiguration("debug_joint_state_topic")
    publish_policy_debug_status = LaunchConfiguration("publish_policy_debug_status")
    policy_debug_status_topic = LaunchConfiguration("policy_debug_status_topic")

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation time if true",
    )

    respawn_arg = DeclareLaunchArgument(
        "respawn",
        default_value="false",
        description="Respawn node when it crashes",
    )

    log_level_arg = DeclareLaunchArgument(
        "log_level",
        default_value="info",
        description="Logging level: debug, info, warn, error, fatal",
    )

    halo_drl_dir_arg = DeclareLaunchArgument(
        "halo_drl_dir",
        default_value="HALO_1/drl_moudle",
        description="Path to HALO_1 drl_moudle directory",
    )

    config_arg = DeclareLaunchArgument(
        "config",
        default_value="configs/mpc_rl.py",
        description="Config path relative to halo_drl_dir, or absolute path",
    )

    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value=PathJoinSubstitution([
            FindPackageShare("controller"),
            "policy",
            "best_model.zip",
        ]),
        description="Absolute path to best_model.zip",
    )

    policy_env_name_arg = DeclareLaunchArgument(
        "policy_env_name",
        default_value="mpc_rl",
        description="Conda env name used by policy worker",
    )

    use_demo_poly_obstacle_arg = DeclareLaunchArgument(
        "use_demo_poly_obstacle",
        default_value="false",
        description="Enable demo polygon constraint in planner request",
    )

    use_wall_constraints_arg = DeclareLaunchArgument(
        "use_wall_constraints",
        default_value="false",
        description="Enable wall constraints generated from map bounds",
    )

    visualize_actions_arg = DeclareLaunchArgument(
        "visualize_actions",
        default_value="true",
        description="Publish action candidates and masks as RViz markers",
    )

    action_marker_topic_arg = DeclareLaunchArgument(
        "action_marker_topic",
        default_value="/policy/action_markers",
        description="MarkerArray topic for action visualization",
    )

    action_marker_frame_arg = DeclareLaunchArgument(
        "action_marker_frame",
        default_value="odom",
        description="Frame id used for action visualization markers",
    )

    publish_debug_joint_state_arg = DeclareLaunchArgument(
        "publish_debug_joint_state",
        default_value="true",
        description="Publish service request JointState for debugging",
    )

    debug_joint_state_topic_arg = DeclareLaunchArgument(
        "debug_joint_state_topic",
        default_value="/debug/joint_state_req",
        description="Topic name for debug JointState messages",
    )

    publish_policy_debug_status_arg = DeclareLaunchArgument(
        "publish_policy_debug_status",
        default_value="true",
        description="Publish compact policy status JSON for runtime verification",
    )

    policy_debug_status_topic_arg = DeclareLaunchArgument(
        "policy_debug_status_topic",
        default_value="/debug/policy_status",
        description="Topic for policy status JSON debug messages",
    )

    ampcc_node = Node(
        package="controller",
        executable="ampcc_node",
        name="opt_planner",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
        respawn=respawn,
        arguments=["--ros-args", "--log-level", log_level],
    )

    rl_bridge_node = Node(
        package="controller",
        executable="rl_ocp_policy_bridge.py",
        name="rl_ocp_policy_bridge",
        output="screen",
        parameters=[
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
                "planner_half_width": 5.8,
                "planner_half_height": 9.8,
                "use_wall_constraints": use_wall_constraints,
                "use_demo_poly_obstacle": use_demo_poly_obstacle,
                "visualize_actions": visualize_actions,
                "action_marker_topic": action_marker_topic,
                "action_marker_frame": action_marker_frame,
                "publish_debug_joint_state": publish_debug_joint_state,
                "debug_joint_state_topic": debug_joint_state_topic,
                "publish_policy_debug_status": publish_policy_debug_status,
                "policy_debug_status_topic": policy_debug_status_topic,
            }
        ],
        respawn=respawn,
        arguments=["--ros-args", "--log-level", log_level],
    )

    return LaunchDescription([
        use_sim_time_arg,
        respawn_arg,
        log_level_arg,
        halo_drl_dir_arg,
        config_arg,
        model_path_arg,
        policy_env_name_arg,
        use_demo_poly_obstacle_arg,
        use_wall_constraints_arg,
        visualize_actions_arg,
        action_marker_topic_arg,
        action_marker_frame_arg,
        publish_debug_joint_state_arg,
        debug_joint_state_topic_arg,
        publish_policy_debug_status_arg,
        policy_debug_status_topic_arg,
        ampcc_node,
        rl_bridge_node,
    ])
