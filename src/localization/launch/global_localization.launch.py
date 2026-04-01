import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # ===========================
    # Launch Arguments
    # ===========================
    map_name_arg = DeclareLaunchArgument(
        "map_name",
        default_value="room_20x20",
        description="Map name (folder name in mapping/maps/)"
    )

    map_yaml_arg = DeclareLaunchArgument(
        "map_yaml",
        default_value="map.yaml",
        description="Map yaml filename in selected map folder"
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation time"
    )
    
    x_pos_arg = DeclareLaunchArgument(
        "x_pos",
        default_value="0.0",
        description="Initial X position"
    )
    
    y_pos_arg = DeclareLaunchArgument(
        "y_pos",
        default_value="0.0",
        description="Initial Y position"
    )
    
    yaw_arg = DeclareLaunchArgument(
        "yaw",
        default_value="0.0",
        description="Initial yaw angle (radians)"
    )

    # ===========================
    # Launch Configurations
    # ===========================
    map_name = LaunchConfiguration("map_name")
    use_sim_time = LaunchConfiguration("use_sim_time")
    map_yaml = LaunchConfiguration("map_yaml")
    x_pos = LaunchConfiguration("x_pos")
    y_pos = LaunchConfiguration("y_pos")
    yaw = LaunchConfiguration("yaw")

    # ===========================
    # Package paths and configs
    # ===========================
    localization_dir = get_package_share_directory("localization")
    
    ekf_config_file = os.path.join(localization_dir, "config", "ekf.yaml")
    amcl_config_file = os.path.join(localization_dir, "config", "amcl.yaml")
    
    map_path = PathJoinSubstitution([
        get_package_share_directory("mapping"),
        "maps",
        map_name,
        map_yaml
    ])

    # ===========================
    # Nodes
    # ===========================
    
    # EKF Node (Local Localization: fuses IMU + Wheel Odometry)
    # Publishes: /odometry/filtered, TF: odom -> base_footprint
    ekf_filter_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[
            ekf_config_file,
            {"use_sim_time": use_sim_time}
        ]
    )
    
    # Map Server (loads map for AMCL)
    nav2_map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[
            {"yaml_filename": map_path},
            {"use_sim_time": use_sim_time}
        ]
    )

    # AMCL (Global Localization: provides map -> odom TF)
    # Delayed start to ensure /scan and TF tree are ready
    nav2_amcl = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        emulate_tty=True,
        parameters=[
            amcl_config_file,
            {"use_sim_time": use_sim_time},
            {
                "initial_pose.x": x_pos,
                "initial_pose.y": y_pos,
                "initial_pose.yaw": yaw,
            }
        ]
    )
    
    # Lifecycle Manager (manages map_server and amcl lifecycle)
    nav2_lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[
            {"node_names": ["map_server", "amcl"]},
            {"use_sim_time": use_sim_time},
            {"autostart": True}
        ]
    )

    # ===========================
    # Launch Description
    # ===========================
    return LaunchDescription([
        map_name_arg,
        map_yaml_arg,
        use_sim_time_arg,
        x_pos_arg,
        y_pos_arg,
        yaw_arg,
        ekf_filter_node,
        nav2_map_server,
        nav2_amcl,
        nav2_lifecycle_manager,
    ])