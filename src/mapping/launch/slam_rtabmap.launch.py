import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_prefix, get_package_share_directory


def _prepare_rtabmap_database(context):
    db_path = os.path.expanduser(LaunchConfiguration("database_path").perform(context))
    db_dir = os.path.dirname(db_path)

    os.makedirs(db_dir, exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    return []


def generate_launch_description():
    localization_dir = get_package_share_directory("localization")
    descriptions_dir = get_package_share_directory("descriptions")
    mapping_dir = get_package_share_directory("mapping")
    mapping_prefix = get_package_prefix("mapping")
    workspace_dir = os.path.dirname(os.path.dirname(mapping_prefix))
    rtabmap_launch_dir = get_package_share_directory("rtabmap_launch")

    default_db_path = os.path.join(
        workspace_dir, "src", "mapping", "maps", "B3", "rtabmap_map.db"
    )
    default_rviz_config = os.path.join(descriptions_dir, "config", "rviz2.rviz")

    use_sim_time = LaunchConfiguration("use_sim_time")
    cfg = LaunchConfiguration("cfg")
    database_path = LaunchConfiguration("database_path")
    namespace = LaunchConfiguration("namespace")
    rtabmap_args = LaunchConfiguration("rtabmap_args")
    use_rviz = LaunchConfiguration("use_rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    
    # --- Đổi biến từ ZED 2 sang D435i ---
    use_camera = LaunchConfiguration("use_camera")
    rgb_topic = LaunchConfiguration("rgb_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")

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
        default_value=default_db_path,
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
        default_value=default_rviz_config,
        description="RViz2 config file path",
    )
    
    # --- ĐỒNG BỘ TOPIC D435i ---
    use_camera_arg = DeclareLaunchArgument(
        "use_camera",
        default_value="true",
        description="Enable D435i RGBD input for RTAB-Map",
    )
    rgb_topic_arg = DeclareLaunchArgument(
        "rgb_topic",
        default_value="/camera/color/image_raw",
        description="RGB image topic used when use_camera=true",
    )
    depth_topic_arg = DeclareLaunchArgument(
        "depth_topic",
        default_value="/camera/aligned_depth_to_color/image_raw",
        description="Depth image topic used when use_camera=true",
    )
    camera_info_topic_arg = DeclareLaunchArgument(
        "camera_info_topic",
        default_value="/camera/color/camera_info",
        description="Camera info topic used when use_camera=true",
    )

    # Nạp file config EKF mới (đã đổi tên ở bước trước)
    ekf_filter_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[
            os.path.join(mapping_dir, "config", "ekf.yaml"),
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
            
            "imu_topic": "/camera/imu",            # Đã trỏ đúng topic IMU của D435i
            "subscribe_scan": "true",
            "scan_topic": "/scan",
            "depth": use_camera,
            "subscribe_rgb": use_camera,
            "rgb_topic": rgb_topic,
            "depth_topic": depth_topic,
            "camera_info_topic": camera_info_topic,
            "approx_sync": "true",
            "odom_sensor_sync": "true",  
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
            "wait_for_transform": "0.5",           # D435i publish TF nhanh và ổn định, hạ xuống 0.5s để giảm độ trễ
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
            use_camera_arg,
            rgb_topic_arg,
            depth_topic_arg,
            camera_info_topic_arg,
            OpaqueFunction(function=_prepare_rtabmap_database),
            ekf_filter_node,
            rtabmap_slam,
            rviz_node,
        ]
    )
