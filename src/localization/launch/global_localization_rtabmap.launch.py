import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    localization_dir = get_package_share_directory("localization")
    descriptions_dir = get_package_share_directory("descriptions")
    rtabmap_launch_dir = get_package_share_directory("rtabmap_launch")
    workspace_root = os.path.abspath(os.path.join(localization_dir, "..", "..", "..", ".."))

    use_sim_time = LaunchConfiguration("use_sim_time")
    use_sim_time_enabled = (
        context.launch_configurations.get("use_sim_time", "false").strip().lower() == "true"
    )
    use_rviz_value = context.launch_configurations.get("use_rviz", "").strip().lower()
    ekf_override = context.launch_configurations.get("ekf_config", "").strip()

    default_sim_map_path = os.path.join(
        workspace_root, "src", "mapping", "maps", "room_20x20", "room_20x20_map.yaml"
    )
    default_real_db_path = os.path.join(
        workspace_root, "src", "mapping", "maps", "B3", "rtabmap_map.db"
    )

    map_path_override = context.launch_configurations.get("map", "").strip()
    selected_map_path = map_path_override or default_sim_map_path
    database_path_value = os.path.expanduser(
        context.launch_configurations.get("database_path", "").strip()
    ) or default_real_db_path

    if ekf_override:
        ekf_config_path = ekf_override
    elif use_sim_time_enabled:
        ekf_config_path = os.path.join(localization_dir, "config", "ekf_sim.yaml")
    else:
        ekf_config_path = os.path.join(localization_dir, "config", "ekf.yaml")

    if use_rviz_value in ("true", "false"):
        resolved_use_rviz = use_rviz_value
    else:
        resolved_use_rviz = "false" if use_sim_time_enabled else "true"

    rviz_config_path = os.path.join(descriptions_dir, "config", "rviz2.rviz")

    ekf_filter_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[
            ekf_config_path,
            {"use_sim_time": use_sim_time},
        ],
    )

    nodes = [ekf_filter_node]

    if use_sim_time_enabled:
        amcl_config = os.path.join(localization_dir, "config", "amcl.yaml")

        map_server_node = Node(
            package="nav2_map_server",
            executable="map_server",
            name="map_server",
            output="screen",
            parameters=[
                {"yaml_filename": selected_map_path},
                {"use_sim_time": use_sim_time},
            ],
        )

        amcl_node = Node(
            package="nav2_amcl",
            executable="amcl",
            name="amcl",
            output="screen",
            parameters=[
                amcl_config,
                {"use_sim_time": use_sim_time},
                {"initial_pose.x": LaunchConfiguration("x_pos")},
                {"initial_pose.y": LaunchConfiguration("y_pos")},
                {"initial_pose.yaw": LaunchConfiguration("yaw")},
            ],
            remappings=[
                ("/tf", "tf"),
                ("/tf_static", "tf_static"),
            ],
        )

        lifecycle_manager_node = Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_localization",
            output="screen",
            parameters=[
                {"node_names": ["map_server", "amcl"]},
                {"use_sim_time": use_sim_time},
                {"autostart": True},
            ],
        )

        nodes.extend([
            LogInfo(
                msg=(
                    f"[global_localization_rtabmap] Simulation mode with AMCL: "
                    f"map={selected_map_path}, ekf={ekf_config_path}, rviz={resolved_use_rviz}"
                )
            ),
            map_server_node,
            amcl_node,
            lifecycle_manager_node,
        ])
    else:
        cfg = LaunchConfiguration("cfg")
        namespace = LaunchConfiguration("namespace")

        rtabmap_localization = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(rtabmap_launch_dir, "launch", "rtabmap.launch.py")
            ),
            launch_arguments={
                "namespace": namespace,
                "use_sim_time": use_sim_time,
                "localization": "true",
                "cfg": cfg,
                "database_path": database_path_value,
                "frame_id": "base_footprint",
                "map_frame_id": "map",
                "map_topic": "/map",
                "odom_topic": "/odometry/filtered",
                "subscribe_rgb": "true",
                "subscribe_depth": "true",
                "rgb_topic": "/camera/color/image_raw",
                "depth_topic": "/camera/aligned_depth_to_color/image_raw",
                "camera_info_topic": "/camera/color/camera_info",
                "subscribe_scan": "true",
                "scan_topic": "/scan",
                "approx_sync": "true",
                "odom_sensor_sync": "true",
                "visual_odometry": "false",
                "icp_odometry": "false",
                "publish_tf_odom": "false",
                "publish_tf_map": "true",
                "rtabmap_viz": "false",
                "rviz": resolved_use_rviz,
                "rviz_cfg": rviz_config_path,
                "qos": "2",
                "qos_imu": "2",
                "qos_scan": "2",
                "qos_odom": "2",
                "qos_image": "2",
                "qos_camera_info": "2",
                "wait_for_transform": "0.5",
            }.items(),
        )

        nodes.extend([
            LogInfo(
                msg=(
                    f"[global_localization_rtabmap] Real mode with RTAB-Map: "
                    f"db={database_path_value}, ekf={ekf_config_path}, rviz={resolved_use_rviz}"
                )
            ),
            rtabmap_localization,
        ])

    return nodes


def generate_launch_description():
    workspace_root = "/home/hdt/LVTN/mobile_robot_arm"
    default_sim_map_path = os.path.join(
        workspace_root, "src", "mapping", "maps", "room_20x20", "room_20x20_map.yaml"
    )
    default_real_db_path = os.path.join(
        workspace_root, "src", "mapping", "maps", "B3", "rtabmap_map.db"
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="false: real mode uses RTAB-Map database in B3. true: simulation uses AMCL with room_20x20 map.",
        ),
        DeclareLaunchArgument(
            "ekf_config",
            default_value="",
            description="Optional EKF config override. Empty selects ekf.yaml or ekf_sim.yaml automatically.",
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="",
            description="Leave empty for auto mode: true on real robot, false in simulation.",
        ),
        DeclareLaunchArgument(
            "map",
            default_value=default_sim_map_path,
            description="AMCL map yaml used in simulation mode.",
        ),
        DeclareLaunchArgument(
            "database_path",
            default_value=default_real_db_path,
            description="RTAB-Map database path used in real mode.",
        ),
        DeclareLaunchArgument(
            "cfg",
            default_value=os.path.join(localization_dir := get_package_share_directory("localization"), "config", "rtabmap_localization.yaml"),
            description="RTAB-Map localization config file used in real mode.",
        ),
        DeclareLaunchArgument(
            "namespace",
            default_value="rtabmap",
            description="Namespace for RTAB-Map nodes in real mode.",
        ),
        DeclareLaunchArgument("x_pos", default_value="0.0"),
        DeclareLaunchArgument("y_pos", default_value="0.0"),
        DeclareLaunchArgument("yaw", default_value="0.0"),
        OpaqueFunction(function=launch_setup),
    ])
