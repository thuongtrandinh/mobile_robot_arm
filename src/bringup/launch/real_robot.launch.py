import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # ==========================================
    # 1. ĐƯỜNG DẪN CÁC PACKAGE
    # ==========================================
    descriptions_pkg = get_package_share_directory('descriptions')
    lidar_pkg = get_package_share_directory('lidar')   
    zed_wrapper_pkg = get_package_share_directory('zed_wrapper') 
    
    # SỬA LỖI Ở ĐÂY: Trỏ trực tiếp lấy file config từ package zed_wrapper
    custom_zed_config_dir = os.path.join(zed_wrapper_pkg, 'config')

    # ==========================================
    # 2. ROBOT STATE PUBLISHER (URDF & TF TREE)
    # ==========================================
    xacro_file = os.path.join(descriptions_pkg, 'model', 'wheeled', 'urdf', 'mobile_robot.urdf.xacro')
    robot_description_content = ParameterValue(Command(['xacro ', xacro_file]), value_type=str)
    
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_content}]
    )

    # ==========================================
    # 3. MICRO-ROS (GIAO TIẾP MẠCH ĐIỀU KHIỂN ĐỘNG CƠ)
    # ==========================================
    micro_ros_agent_node = Node(
        package='micro_ros_agent',
        executable='micro_ros_agent',
        name='micro_ros_agent',
        arguments=['serial', 'b', '115200', '--dev', '/dev/uart'], # LƯU Ý: Đổi cổng nếu cần
        output='screen'
    )

    # ==========================================
    # 4. KHỞI CHẠY RPLIDAR A2M8
    # ==========================================
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(lidar_pkg, 'launch', 'a2m8.launch.py')
        ),
        launch_arguments={
            'baud_rate': '256000',              # Tốc độ truyền dữ liệu (phải khớp với cài đặt của Lidar)
            'serial_port': '/dev/rplidar',  # LƯU Ý: Cổng Lidar (khác cổng micro-ROS)
            'frame_id': 'laser'             
        }.items()
    )

    # ==========================================
    # 5. KHỞI CHẠY CAMERA ZED 2
    # ==========================================
    zed2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(zed_wrapper_pkg, 'launch', 'zed_camera.launch.py')
        ),
        launch_arguments={
            'camera_model': 'zed2',
            'camera_name': 'zed2',
            'publish_tf': 'true',       
            'publish_map_tf': 'false',  
            'config_path': custom_zed_config_dir # Nạp cấu hình từ biến đã sửa
        }.items()
    )

    # ==========================================
    # 6. KỊCH BẢN KHỞI CHẠY TUẦN TỰ (SEQUENCE)
    # ==========================================
    return LaunchDescription([
        # BƯỚC 1: Bật ngay bộ khung tọa độ (TF) và kết nối vi điều khiển
        robot_state_publisher_node,
        micro_ros_agent_node,
        
        # BƯỚC 2: Chờ 5 giây để mạch kết nối xong, sau đó bật Lidar
        TimerAction(
            period=5.0,
            actions=[lidar_launch]
        ),
        
        # BƯỚC 3: Chờ 5 giây rồi mới bật ZED 2 để tránh sốc điện cổng USB
        TimerAction(
            period=5.0,
            actions=[zed2_launch]
        )
    ])