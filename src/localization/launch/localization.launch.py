import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    map_yaml_file = LaunchConfiguration('map', default=os.path.join(get_package_share_directory('mapping'), 'maps', 'small_warehouse', 'map.yaml')) # Trỏ tới map bạn đã quét
    
    localization_dir = get_package_share_directory('localization')
    
    # Khởi chạy EKF
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[os.path.join(localization_dir, 'config', 'ekf.yaml'), {'use_sim_time': use_sim_time}],
        remappings=[("odometry/filtered", "odom")]
    )

    # Khởi chạy Nav2 Map Server (để tải bản đồ đã lưu)
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename': map_yaml_file}, {'use_sim_time': use_sim_time}]
    )

    # Khởi chạy Nav2 AMCL (Định vị)
    amcl_node = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[os.path.join(localization_dir, 'config', 'amcl.yaml'), {'use_sim_time': use_sim_time}]
    )

    # Lifecycle Manager cho Nav2 (Bắt buộc để kích hoạt map_server và amcl)
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time},
                    {'autostart': True},
                    {'node_names': ['map_server', 'amcl']}]
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('map', default_value=map_yaml_file),
        ekf_node,
        map_server_node,
        amcl_node,
        lifecycle_manager
    ])
