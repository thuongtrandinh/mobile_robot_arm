import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    localization_dir = get_package_share_directory("localization")
    mapping_dir = get_package_share_directory("mapping")
    rtabmap_launch_dir = get_package_share_directory("rtabmap_launch")

    use_sim_time = LaunchConfiguration("use_sim_time")
    cfg = LaunchConfiguration("cfg")
    database_path = LaunchConfiguration("database_path")
    namespace = LaunchConfiguration("namespace")
    rtabmap_args = LaunchConfiguration("rtabmap_args")
    use_rviz = LaunchConfiguration("use_rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    use_zed = LaunchConfiguration("use_zed")
    rgb_topic = LaunchConfiguration("rgb_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    # topic_queue_size = LaunchConfiguration("topic_queue_size")
    # sync_queue_size = LaunchConfiguration("sync_queue_size")
    # approx_sync_max_interval = LaunchConfiguration("approx_sync_max_interval")

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation clock",
    )
    cfg_arg = DeclareLaunchArgument(
        "cfg",
        default_value=os.path.join(mapping_dir, "config", "rtabmap_slam.yaml"),
        description="RTAB-Map config file for SLAM mode",
    )
    database_path_arg = DeclareLaunchArgument(
        "database_path",
        default_value="~/.ros/rtabmap_map.db",
        description="RTAB-Map database output in mapping phase",
    )
    namespace_arg = DeclareLaunchArgument(
        "namespace",
        default_value="rtabmap",
        description="Namespace for RTAB-Map nodes",
    )
    rtabmap_args_arg = DeclareLaunchArgument(
        "rtabmap_args",
        default_value="",
        description="Extra RTAB-Map args, e.g. '-d' to reset database on start",
    )
    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Launch RViz2",
    )
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value="/home/thuong/LVTN/amr_ws/src/descriptions/config/rviz2.rviz",
        description="RViz2 config file path",
    )
    use_zed_arg = DeclareLaunchArgument(
        "use_zed",
        default_value="true",
        description="Enable ZED2 RGBD input for RTAB-Map (requires matching camera topics)",
    )
    rgb_topic_arg = DeclareLaunchArgument(
        "rgb_topic",
        default_value="/zed/zed_node/rgb/color/rect/image",
        description="RGB image topic used when use_zed=true",
    )
    depth_topic_arg = DeclareLaunchArgument(
        "depth_topic",
        default_value="/zed/zed_node/depth/depth_registered",
        description="Depth image topic used when use_zed=true",
    )
    camera_info_topic_arg = DeclareLaunchArgument(
        "camera_info_topic",
        default_value="/zed/zed_node/rgb/color/rect/camera_info",
        description="Camera info topic used when use_zed=true",
    )
    # topic_queue_size_arg = DeclareLaunchArgument(
    #     "topic_queue_size",
    #     default_value="30",
    #     description="RTAB-Map topic queue size for each subscribed input",
    # )
    # sync_queue_size_arg = DeclareLaunchArgument(
    #     "sync_queue_size",
    #     default_value="30",
    #     description="RTAB-Map synchronization queue size",
    # )
    # approx_sync_max_interval_arg = DeclareLaunchArgument(
    #     "approx_sync_max_interval",
    #     default_value="0.2",
    #     description="Max interval (s) allowed for approximate synchronization",
    # )

    ekf_filter_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[
            os.path.join(localization_dir, "config", "ekf_zed2.yaml"),
            {"use_sim_time": use_sim_time},
        ],
    )

    rtabmap_slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(rtabmap_launch_dir, "launch", "rtabmap.launch.py")
        ),
        launch_arguments={
            "namespace": namespace,
            "use_sim_time": use_sim_time,
            "localization": "false",
            "cfg": cfg,
            "database_path": database_path,
            "args": rtabmap_args,
            "frame_id": "base_footprint",
            "map_frame_id": "map",
            "odom_topic": "/odometry/filtered",
            "imu_topic": "/imu",
            "subscribe_scan": "true",
            "scan_topic": "/scan",
            "depth": use_zed,
            "subscribe_rgb": use_zed,
            "rgb_topic": rgb_topic,
            "depth_topic": depth_topic,
            "camera_info_topic": camera_info_topic,
            "approx_sync": "true",
            # "approx_sync_max_interval": approx_sync_max_interval,
            # "topic_queue_size": topic_queue_size,
            # "sync_queue_size": sync_queue_size,
            "odom_sensor_sync": "false",
            "visual_odometry": "false",
            "icp_odometry": "false",
            "publish_tf_odom": "false",
            "publish_tf_map": "true",
            "rtabmap_viz": "false",
            "rviz": "false",
            "qos": "2",
            "qos_imu": "2",
            "qos_scan": "2",
            "qos_odom": "2",
            "wait_for_transform": "0.5",
            "qos_image": "2",
            "qos_camera_info": "2",
        }.items(),
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            cfg_arg,
            database_path_arg,
            namespace_arg,
            rtabmap_args_arg,
            use_rviz_arg,
            rviz_config_arg,
            use_zed_arg,
            rgb_topic_arg,
            depth_topic_arg,
            camera_info_topic_arg,
            # topic_queue_size_arg,
            # sync_queue_size_arg,
            # approx_sync_max_interval_arg,
            ekf_filter_node,
            rtabmap_slam,
            rviz_node,
        ]
    )
