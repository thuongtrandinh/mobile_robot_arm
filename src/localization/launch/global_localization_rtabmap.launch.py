import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # 1. Tìm đường dẫn Workspace root (Thư mục chứa src, install, build)
    # Lấy đường dẫn của gói này trong thư mục install
    pkg_localization_share = get_package_share_directory("localization")
    # Từ install/localization/share/localization nhảy ngược lên 4 cấp để ra Workspace Root
    workspace_root = os.path.abspath(os.path.join(pkg_localization_share, "..", "..", "..", ".."))
    
    # Định nghĩa thư mục src
    src_dir = os.path.join(workspace_root, "src")

    # 2. Khai báo các đường dẫn dựa trên thư mục /src
    # Đường dẫn tới file database RTAB-Map
    default_db_path = os.path.join(src_dir, "mapping", "maps", "B3", "rtabmap_map.db")
    
    # Đường dẫn tới các thư mục share (vẫn nên dùng share cho config để đảm bảo tính ổn định sau build)
    localization_dir = get_package_share_directory("localization")
    mapping_dir = get_package_share_directory("mapping")
    rtabmap_launch_dir = get_package_share_directory("rtabmap_launch")

    use_sim_time = LaunchConfiguration("use_sim_time")
    cfg = LaunchConfiguration("cfg")
    database_path = LaunchConfiguration("database_path")
    namespace = LaunchConfiguration("namespace")
    
    # Cấu hình EKF (Sử dụng file mới không có VO)
    ekf_config = os.path.join(mapping_dir, "config", "ekf.yaml")

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation clock",
    )
    cfg_arg = DeclareLaunchArgument(
        "cfg",
        default_value=os.path.join(localization_dir, "config", "rtabmap_localization.yaml"),
        description="RTAB-Map config file for localization mode",
    )
    database_path_arg = DeclareLaunchArgument(
        "database_path",
        default_value=default_db_path, # Sử dụng đường dẫn tương đối từ src đã tính ở trên
        description="RTAB-Map database path relative to workspace src",
    )
    namespace_arg = DeclareLaunchArgument(
        "namespace",
        default_value="rtabmap",
        description="Namespace for RTAB-Map nodes",
    )

    # 3. NODE EKF (Dung hợp Encoder + IMU D435i)
    ekf_filter_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[ekf_config, {"use_sim_time": use_sim_time}],
    )

    # 4. RTAB-MAP LOCALIZATION (Đã loại bỏ hoàn toàn VO)
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
            
            # Topic đồng bộ D435i
            "imu_topic": "/camera/imu",
            "rgb_topic": "/camera/color/image_raw",
            "depth_topic": "/camera/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/color/camera_info",
            
            "subscribe_scan": "true",
            "scan_topic": "/scan",
            "approx_sync": "true",
            "visual_odometry": "false",       # Loại bỏ Visual Odometry
            "icp_odometry": "false",
            "publish_tf_odom": "false",
            "publish_tf_map": "true",
            "rtabmap_viz": "false",
            "rviz": "false",
            "qos": "2",
            "qos_imu": "2",
            "qos_scan": "2",
            "qos_odom": "2",
            "qos_image": "2",
            "qos_camera_info": "2",
            "wait_for_transform": "0.5",
        }.items(),
    )

    return LaunchDescription(
        [
            use_sim_time_arg,
            cfg_arg,
            database_path_arg,
            namespace_arg,
            ekf_filter_node,
            rtabmap_localization,
        ]
    )
