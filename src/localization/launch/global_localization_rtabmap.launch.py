import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 1. Lấy đường dẫn các package cơ bản
    localization_dir = get_package_share_directory("localization")
    rtabmap_launch_dir = get_package_share_directory("rtabmap_launch")

    # 2. Xác định Workspace Root và thư mục src
    workspace_root = os.path.abspath(os.path.join(localization_dir, "..", "..", "..", ".."))
    src_dir = os.path.join(workspace_root, "src")

    # --- KHAI BÁO CÁC ĐƯỜNG DẪN CONFIG (ĐÃ TỐI ƯU) ---
    
    # Path Database Map (B3)
    default_db_path = os.path.join(src_dir, "mapping", "maps", "B3", "rtabmap_map.db")
    
    # Path EKF (Sử dụng file bạn đã copy vào localization/config)
    ekf_config = os.path.join(localization_dir, "config", "ekf.yaml")
    
    # Path RViz (Trỏ chính xác về thư mục descriptions trong src)
    rviz_config_path = os.path.join(src_dir, "descriptions", "config", "rviz2.rviz")

    # --- LAUNCH ARGUMENTS ---
    use_sim_time = LaunchConfiguration("use_sim_time")
    cfg = LaunchConfiguration("cfg")
    database_path = LaunchConfiguration("database_path")
    namespace = LaunchConfiguration("namespace")
    
    use_sim_time_arg = DeclareLaunchArgument("use_sim_time", default_value="false")
    cfg_arg = DeclareLaunchArgument(
        "cfg", 
        default_value=os.path.join(localization_dir, "config", "rtabmap_localization.yaml")
    )
    database_path_arg = DeclareLaunchArgument("database_path", default_value=default_db_path)
    namespace_arg = DeclareLaunchArgument("namespace", default_value="rtabmap")

    # 3. NODE EKF (Chỉ dùng Encoder + IMU thô để tính Odom lọc)
    ekf_filter_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[ekf_config, {"use_sim_time": use_sim_time}],
    )

    # 4. RTAB-MAP LOCALIZATION (Đã loại bỏ IMU trực tiếp, dùng RViz config riêng)
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
            
            # Chỉ EKF xử lý IMU, RTAB-Map chỉ nhận ảnh và scan
            "rgb_topic": "/camera/color/image_raw",
            "depth_topic": "/camera/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/color/camera_info",
            
            "subscribe_scan": "true",
            "scan_topic": "/scan",
            "approx_sync": "true",
            "visual_odometry": "false",
            "icp_odometry": "false",
            "publish_tf_odom": "false",
            "publish_tf_map": "true",
            
            # Tắt GUI mặc định và dùng RViz config bạn chỉ định
            "rtabmap_viz": "false",
            "rviz": "true",
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
        ekf_filter_node,
        rtabmap_localization,
    ])
