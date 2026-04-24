import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    localization_dir = get_package_share_directory("localization")
    rtabmap_launch_dir = get_package_share_directory("rtabmap_launch")
    workspace_root = os.path.abspath(os.path.join(localization_dir, "..", "..", "..", ".."))
    src_dir = os.path.join(workspace_root, "src")
    maps_root = os.path.join(src_dir, "mapping", "maps")

    default_db_path = os.path.join(os.path.expanduser("~"), ".ros", "rtabmap_map.db")
    ekf_config = os.path.join(localization_dir, "config", "ekf.yaml")
    rviz_config_path = os.path.join(src_dir, "descriptions", "config", "rviz2.rviz")

    use_sim_time = LaunchConfiguration("use_sim_time")
    cfg = LaunchConfiguration("cfg")
    database_path = LaunchConfiguration("database_path")
    namespace = LaunchConfiguration("namespace")
    map_name = LaunchConfiguration("map_name")
    map_yaml = LaunchConfiguration("map_yaml")
    use_map_server = LaunchConfiguration("use_map_server")
    launch_rviz = LaunchConfiguration("launch_rviz")
    initial_x = LaunchConfiguration("initial_x")
    initial_y = LaunchConfiguration("initial_y")
    initial_yaw = LaunchConfiguration("initial_yaw")

    use_sim_time_arg = DeclareLaunchArgument("use_sim_time", default_value="false")
    cfg_arg = DeclareLaunchArgument(
        "cfg",
        default_value=os.path.join(localization_dir, "config", "rtabmap_localization.yaml")
    )
    database_path_arg = DeclareLaunchArgument("database_path", default_value=default_db_path)
    namespace_arg = DeclareLaunchArgument("namespace", default_value="rtabmap")
    map_name_arg = DeclareLaunchArgument("map_name", default_value="room_20x20")
    map_yaml_arg = DeclareLaunchArgument("map_yaml", default_value="room_20x20_map.yaml")
    use_map_server_arg = DeclareLaunchArgument("use_map_server", default_value="true")
    launch_rviz_arg = DeclareLaunchArgument("launch_rviz", default_value="true")
    initial_x_arg = DeclareLaunchArgument("initial_x", default_value="0.0")
    initial_y_arg = DeclareLaunchArgument("initial_y", default_value="0.0")
    initial_yaw_arg = DeclareLaunchArgument("initial_yaw", default_value="0.0")

    ekf_filter_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[ekf_config, {"use_sim_time": use_sim_time}],
    )

    map_server_node = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[
            {
                "yaml_filename": PathJoinSubstitution([maps_root, map_name, map_yaml]),
                "use_sim_time": use_sim_time,
            }
        ],
        condition=IfCondition(use_map_server),
    )

    lifecycle_manager_node = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization_map",
        output="screen",
        parameters=[
            {
                "node_names": ["map_server"],
                "use_sim_time": use_sim_time,
                "autostart": True,
            }
        ],
        condition=IfCondition(use_map_server),
    )

    rtabmap_localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(rtabmap_launch_dir, "launch", "rtabmap.launch.py")
        ),
        launch_arguments={
            "namespace": namespace,
            "use_sim_time": use_sim_time,
            "localization": "true",
            "cfg": cfg,
            "database_path": database_path,
            "frame_id": "base_footprint",
            "map_frame_id": "map",
            "odom_topic": "/odometry/filtered",
            "initial_pose": PythonExpression([
                "'",
                initial_x,
                " ",
                initial_y,
                " 0 0 0 ",
                initial_yaw,
                "'",
            ]),
            "subscribe_rgb": "true",
            "subscribe_depth": "true",
            "rgb_topic": "/camera/color/image_raw",
            "depth_topic": "/camera/depth/image_rect_raw",
            "camera_info_topic": "/camera/color/camera_info",
            "subscribe_scan": "true",
            "scan_topic": "/scan",
            "approx_sync": "true",
            "visual_odometry": "false",
            "icp_odometry": "false",
            "publish_tf_odom": "false",
            "publish_tf_map": "true",
            "rtabmap_viz": "false",
            "rviz": launch_rviz,
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

    return LaunchDescription([
        use_sim_time_arg,
        cfg_arg,
        database_path_arg,
        namespace_arg,
        map_name_arg,
        map_yaml_arg,
        use_map_server_arg,
        launch_rviz_arg,
        initial_x_arg,
        initial_y_arg,
        initial_yaw_arg,
        ekf_filter_node,
        map_server_node,
        lifecycle_manager_node,
        rtabmap_localization,
    ])
