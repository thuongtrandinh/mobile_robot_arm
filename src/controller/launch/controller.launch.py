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
    
    # [GIỮ NGUYÊN] Các cấu hình phục vụ Visualization
    visualize_actions = LaunchConfiguration("visualize_actions")
    action_marker_topic = LaunchConfiguration("action_marker_topic")
    action_marker_frame = LaunchConfiguration("action_marker_frame")

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

    # File cấu hình chung cho hệ thống
    tuning_config_arg = DeclareLaunchArgument(
        "tuning_config",
        default_value=PathJoinSubstitution([
            FindPackageShare("controller"),
            "config",
            "ddmr_mpc_config.yaml",
        ]),
        description="YAML file containing system tuning parameters",
    )
    
    # -------------------------------------------------------------
    # FILE CẤU HÌNH NAV2
    # -------------------------------------------------------------
    mppi_params_file_arg = DeclareLaunchArgument(
        "mppi_params_file",
        default_value=PathJoinSubstitution([
            FindPackageShare("controller"),
            "config",
            "mppi_params.yaml", 
        ]),
        description="YAML file containing Nav2 MPPI configuration",
    )
    mppi_params_file = LaunchConfiguration("mppi_params_file")

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

    # -------------------------------------------------------------
    # 1. GỌI NAV2 BRINGUP (CHỈ CHẠY CONTROLLER SERVER)
    # -------------------------------------------------------------
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    nav2_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': mppi_params_file,
            'autostart': 'true'
        }.items()
    )

    # BỌC NAV2 LẠI, ÉP DÙNG SIM TIME VÀ BẺ LÁI VẬN TỐC
    # diff_drive_controller đang dùng Twist (use_stamped_vel: false),
    # nên phải đẩy lệnh vào /diff_cont/cmd_vel_unstamped.
    nav2_group = GroupAction([
        # Ép mọi node trong group này dùng chung biến use_sim_time
        SetParameter(name='use_sim_time', value=use_sim_time),
        SetRemap(src='cmd_vel', dst='/diff_cont/cmd_vel_unstamped'),
        SetRemap(src='cmd_vel_nav', dst='/diff_cont/cmd_vel_unstamped'),
        nav2_cmd
    ])

    # -------------------------------------------------------------
    # 2. NODE PYTHON BRIDGE CỦA BẠN (MẠNG RL CAO CẤP)
    # -------------------------------------------------------------
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
                "use_nav2_action": True, 
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
        tuning_config_arg,
        mppi_params_file_arg,
        halo_drl_dir_arg,
        config_arg,
        model_path_arg,
        policy_env_name_arg,
        
        # CHỈ GỌI nav2_group (đã chứa sẵn nav2_cmd bên trong rồi)
        nav2_group,
        # rl_bridge_node,
    ])
