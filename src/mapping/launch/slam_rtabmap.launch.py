import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    localization_dir = get_package_share_directory("localization")
    mapping_dir = get_package_share_directory("mapping")
    zed2_dir = get_package_share_directory("zed2")
    lidar_dir = get_package_share_directory("lidar")
    rtabmap_launch_dir = get_package_share_directory("rtabmap_launch")

    use_sim_time = LaunchConfiguration("use_sim_time")
    cfg = LaunchConfiguration("cfg")
    database_path = LaunchConfiguration("database_path")
    namespace = LaunchConfiguration("namespace")
    rtabmap_args = LaunchConfiguration("rtabmap_args")
    use_zed = LaunchConfiguration("use_zed")
    use_lidar = LaunchConfiguration("use_lidar")
    start_hardware = LaunchConfiguration("start_hardware")
    rgb_topic = LaunchConfiguration("rgb_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")

    ekf_config = os.path.join(mapping_dir, "config", "ekf_encoder_zed_vio.yaml")

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
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
    use_zed_arg = DeclareLaunchArgument(
        "use_zed",
        default_value="true",
        description="Enable ZED2 RGBD input for RTAB-Map (requires matching camera topics)",
    )
    use_lidar_arg = DeclareLaunchArgument(
        "use_lidar",
        default_value="true",
        description="Enable LiDAR input for RTAB-Map",
    )
    start_hardware_arg = DeclareLaunchArgument(
        "start_hardware",
        default_value="true",
        description="If use_sim_time=false and start_hardware=true, launch ZED2 and LiDAR drivers automatically",
    )
    rgb_topic_arg = DeclareLaunchArgument(
        "rgb_topic",
        default_value="/zed2/zed_node/rgb/color/rect/image",
        description="RGB image topic used when use_zed=true",
    )
    depth_topic_arg = DeclareLaunchArgument(
        "depth_topic",
        default_value="/zed2/zed_node/depth/depth_registered",
        description="Depth image topic used when use_zed=true",
    )
    camera_info_topic_arg = DeclareLaunchArgument(
        "camera_info_topic",
        default_value="/zed2/zed_node/rgb/color/rect/camera_info",
        description="Camera info topic used when use_zed=true",
    )

    zed2_hardware = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(zed2_dir, "launch", "zed2.launch.py")
        ),
        launch_arguments={
            "namespace": "",
            "camera_name": "zed2",
            "node_name": "zed_node",
            "use_sim_time": "false",
            "sim_mode": "false",
            "publish_svo_clock": "false",
        }.items(),
        condition=IfCondition(PythonExpression([
            "'", use_sim_time, "' == 'false' and '", start_hardware, "' == 'true' and '", use_zed, "' == 'true'"
        ])),
    )

    lidar_hardware = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(lidar_dir, "launch", "a2m8.launch.py")
        ),
        launch_arguments={
            "use_sim_time": "false",
            "frame_id": "laser",
        }.items(),
        condition=IfCondition(PythonExpression([
            "'", use_sim_time, "' == 'false' and '", start_hardware, "' == 'true' and '", use_lidar, "' == 'true'"
        ])),
    )

    # Disabled old EKF pipeline (IMU + encoder only), kept here for reference.
    ekf_filter_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[
            os.path.join(localization_dir, "config", "ekf.yaml"),
            {"use_sim_time": use_sim_time},
        ],
    )

    # zed2_rgbd_odometry = Node(
    #     package="rtabmap_odom",
    #     executable="rgbd_odometry",
    #     name="zed2_rgbd_odometry",
    #     output="screen",
    #     condition=IfCondition(use_zed),
    #     parameters=[
    #         {
    #             "use_sim_time": use_sim_time,
    #             "frame_id": "base_footprint",
    #             "odom_frame_id": "zed2_odom",
    #             "publish_tf": False,
    #             "publish_null_when_lost": False, # Thêm dòng này
    #             "wait_for_transform": 0.2,
    #             "approx_sync": True,
    #             "topic_queue_size": 30,
    #             "sync_queue_size": 30,
    #             "qos": 2,
    #             "qos_camera_info": 2,
    #             "qos_imu": 2,
    #             "Reg/Force3DoF": "true",
    #         }
    #     ],
    #     remappings=[
    #         ("rgb/image", rgb_topic),
    #         ("depth/image", depth_topic),
    #         ("rgb/camera_info", camera_info_topic),
    #         ("imu", "/imu"),
    #         ("odom", "/zed2/odom_vio"),
    #     ],
    # )

    # ekf_filter_node = Node(
    #     package="robot_localization",
    #     executable="ekf_node",
    #     name="ekf_filter_node",
    #     output="screen",
    #     parameters=[
    #         ekf_config,
    #         {"use_sim_time": use_sim_time},
    #     ],
    # )

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
            "approx_sync_max_interval": "0.05",
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
            "wait_for_transform": "0.2",
            "qos_image": "2",
            "qos_camera_info": "2",
        }.items(),
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            cfg_arg,
            database_path_arg,
            namespace_arg,
            rtabmap_args_arg,
            use_zed_arg,
            use_lidar_arg,
            start_hardware_arg,
            rgb_topic_arg,
            depth_topic_arg,
            camera_info_topic_arg,
            zed2_hardware,
            lidar_hardware,
            # zed2_rgbd_odometry,
            ekf_filter_node,
            rtabmap_slam,
        ]
    )
