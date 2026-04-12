import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    localization_dir = get_package_share_directory('localization')
    # Đường dẫn tới map .yaml bạn đã lưu từ SLAM Toolbox
    map_file = os.path.join(get_package_share_directory('mapping'), 'maps', 'small_warehouse', 'map.yaml')

    return LaunchDescription([
        # 1. EKF Node
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[os.path.join(localization_dir, 'config', 'ekf.yaml')],
            remappings=[('/odometry/filtered', '/odom')]
        ),
        # 2. Map Server (Tải bản đồ lên)
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{'yaml_filename': map_file}]
        ),
        # 3. AMCL Node (Định vị robot trên bản đồ)
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[os.path.join(localization_dir, 'config', 'amcl.yaml')]
        ),
        # 4. Lifecycle Manager (Kích hoạt các node Nav2)
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localization',
            output='screen',
            parameters=[{'autostart': True, 'node_names': ['map_server', 'amcl']}]
        )
    ])
