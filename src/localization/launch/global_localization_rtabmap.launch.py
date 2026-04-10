import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    localization_dir = get_package_share_directory("localization")
    rtabmap_launch_dir = get_package_share_directory("rtabmap_launch")

    use_sim_time = LaunchConfiguration("use_sim_time")
    cfg = LaunchConfiguration("cfg")
    database_path = LaunchConfiguration("database_path")
    namespace = LaunchConfiguration("namespace")
    map_name = LaunchConfiguration("map_name")
    map_yaml = LaunchConfiguration("map_yaml")
    use_map_server = LaunchConfiguration("use_map_server")

    ekf_config = os.path.join(localization_dir, "config", "ekf_zed2.yaml")
    map_yaml_path = PathJoinSubstitution(
        [get_package_share_directory("mapping"), "maps", map_name, map_yaml]
    )

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock",
    )
    cfg_arg = DeclareLaunchArgument(
        "cfg",
        default_value=os.path.join(localization_dir, "config", "rtabmap_localization.yaml"),
        description="RTAB-Map config file for localization mode",
    )
    database_path_arg = DeclareLaunchArgument(
        "database_path",
        default_value="~/.ros/rtabmap_map.db",
        description="RTAB-Map database generated in mapping phase",
    )
    namespace_arg = DeclareLaunchArgument(
        "namespace",
        default_value="rtabmap",
        description="Namespace for RTAB-Map nodes",
    )
    map_name_arg = DeclareLaunchArgument(
        "map_name",
        default_value="room_20x20",
        description="Map folder name under mapping/maps",
    )
    map_yaml_arg = DeclareLaunchArgument(
        "map_yaml",
        default_value="room_20x20_map.yaml",
        description="Map yaml filename in selected map folder",
    )
    use_map_server_arg = DeclareLaunchArgument(
        "use_map_server",
        default_value="true",
        description="Load occupancy map with nav2 map_server",
    )

    ekf_filter_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[
            ekf_config,
            {"use_sim_time": use_sim_time},
        ],
    )

    map_server_node = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        condition=IfCondition(use_map_server),
        parameters=[
            {"yaml_filename": map_yaml_path},
            {"use_sim_time": use_sim_time},
        ],
    )

    lifecycle_manager_node = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_map_server",
        output="screen",
        condition=IfCondition(use_map_server),
        parameters=[
            {"node_names": ["map_server"]},
            {"use_sim_time": use_sim_time},
            {"autostart": True},
        ],
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
            "imu_topic": "/imu",
            "subscribe_scan": "true",
            "scan_topic": "/scan",
            "depth": "true",
            "subscribe_rgb": "true",
            "initial_pose": "0 0 0 0 0 0",  # Ép RTAB-Map hiểu robot xuất phát ở gốc tọa độ
            "rgb_topic": "/zed/zed_node/rgb/color/rect/image",
            "depth_topic": "/zed/zed_node/depth/depth_registered",
            "camera_info_topic": "/zed/zed_node/rgb/color/rect/camera_info",
            "approx_sync": "true",
            "visual_odometry": "false",
            "icp_odometry": "false",
            "publish_tf_odom": "false",
            "rtabmap_viz": "false",
            "rviz": "false",
            "qos": "2",
            "qos_imu": "2",
            "qos_scan": "2",
            "qos_odom": "2",
            "qos_image": "2",       # <--- THÊM DÒNG NÀY (Hạ chuẩn QoS ảnh)
            "qos_camera_info": "2", # <--- THÊM DÒNG NÀY (Hạ chuẩn QoS thông số camera)
            "wait_for_transform": "0.2",
        }.items(),
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            cfg_arg,
            database_path_arg,
            namespace_arg,
            map_name_arg,
            map_yaml_arg,
            use_map_server_arg,
            ekf_filter_node,
            map_server_node,
            lifecycle_manager_node,
            rtabmap_localization,
        ]
    )
