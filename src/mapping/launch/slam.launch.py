import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Định nghĩa tham số sim time
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    # Lấy đường dẫn gói mapping
    mapping_dir = get_package_share_directory('mapping')

    # Đường dẫn tới các file YAML
    ekf_config_path = os.path.join(mapping_dir, 'config', 'ekf.yaml')
    slam_toolbox_config_path = os.path.join(mapping_dir, 'config', 'slam_toolbox.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'),

        # 1. Khởi chạy Robot Localization (EKF)
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[
                ekf_config_path, 
                {'use_sim_time': use_sim_time}
            ],
            # Remap topic đầu ra thành /odom để Nav2 và SLAM dùng chuẩn
            remappings=[("odometry/filtered", "odom")] 
        ),

        # 2. Khởi chạy SLAM Toolbox
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[
                slam_toolbox_config_path, 
                {'use_sim_time': use_sim_time}
            ]
        )
    ])
