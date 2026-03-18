from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_dir_arg = DeclareLaunchArgument(
        "model_dir",
        default_value="/home/thuong/LVTN/amr_ws/HALO/drl_moudle/train_data/run_04",
    )
    halo_drl_dir_arg = DeclareLaunchArgument(
        "halo_drl_dir",
        default_value="/home/thuong/LVTN/amr_ws/HALO/drl_moudle",
    )
    config_arg = DeclareLaunchArgument(
        "config",
        default_value="configs/mpc_rl.py",
    )

    bridge_node = Node(
        package="amr_planner",
        executable="rl_local_goal_bridge",
        name="rl_local_goal_bridge",
        output="screen",
        parameters=[
            {
                "model_dir": LaunchConfiguration("model_dir"),
                "halo_drl_dir": LaunchConfiguration("halo_drl_dir"),
                "config": LaunchConfiguration("config"),
                "odom_topic": "/odometry/filtered",
                "scan_topic": "/scan",
                "goal_topic": "/goal_pose",
                "cmd_topic": "/diff_cont/cmd_vel",
            }
        ],
    )

    return LaunchDescription([
        model_dir_arg,
        halo_drl_dir_arg,
        config_arg,
        bridge_node,
    ])
