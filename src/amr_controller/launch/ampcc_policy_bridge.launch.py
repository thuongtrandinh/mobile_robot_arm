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
        default_value="/home/thuong/LVTN/amr_ws/HALO_1/drl_moudle",
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
            FindPackageShare("amr_controller"),
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

    ampcc_node = Node(
        package="amr_controller",
        executable="ampcc_node",
        name="opt_planner",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
        respawn=respawn,
        arguments=["--ros-args", "--log-level", log_level],
    )

    rl_bridge_node = Node(
        package="amr_controller",
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
        ampcc_node,
        rl_bridge_node,
    ])
