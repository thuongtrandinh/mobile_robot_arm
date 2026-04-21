from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    respawn = LaunchConfiguration("respawn")
    log_level = LaunchConfiguration("log_level")
    tuning_config = LaunchConfiguration("tuning_config")

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
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

    tuning_config_arg = DeclareLaunchArgument(
        "tuning_config",
        default_value=PathJoinSubstitution([
            FindPackageShare("controller"),
            "config",
            "ddmr_mpc_config.yaml",
        ]),
        description="YAML file containing MPC and policy tuning parameters",
    )

    ampcc_node = Node(
        package="controller",
        executable="ampcc_node",
        name="opt_planner",
        output="screen",
        parameters=[tuning_config, {"use_sim_time": use_sim_time}],
        respawn=respawn,
        arguments=["--ros-args", "--log-level", log_level],
    )

    return LaunchDescription([
        use_sim_time_arg,
        respawn_arg,
        log_level_arg,
        tuning_config_arg,
        ampcc_node,
    ])
