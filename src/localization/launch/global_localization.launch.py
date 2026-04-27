import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    workspace_root = os.path.abspath(os.path.join(get_package_share_directory('localization'), '..', '..', '..', '..'))
    default_map_path = os.path.join(
        workspace_root,
        'src',
        'mapping',
        'maps',
        'room_20x20',
        'room_20x20_map.yaml',
    )

    # ===========================
    # Launch Arguments
    # ===========================
    # Cho phép truyền đường dẫn map từ terminal, nếu không truyền sẽ dùng default_map_path
    map_yaml_arg = DeclareLaunchArgument(
        "map",
        default_value=default_map_path,
        description="Full path to the map yaml file"
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation time"
    )
    
    # Các tham số vị trí ban đầu cho AMCL
    x_pos_arg = DeclareLaunchArgument("x_pos", default_value="0.0")
    y_pos_arg = DeclareLaunchArgument("y_pos", default_value="0.0")
    yaw_arg = DeclareLaunchArgument("yaw", default_value="0.0")

    # ===========================
    # Nodes Configuration
    # ===========================
    
    # 1. Map Server
    map_server_node = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[
            {"yaml_filename": LaunchConfiguration("map")}, # Lấy từ tham số 'map'
            {"use_sim_time": LaunchConfiguration("use_sim_time")}
        ]
    )

    # 2. AMCL
    amcl_config = os.path.join(get_package_share_directory('localization'), 'config', 'amcl.yaml')
    
    nav2_amcl = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        parameters=[
            amcl_config,
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
            {"initial_pose.x": LaunchConfiguration("x_pos")},
            {"initial_pose.y": LaunchConfiguration("y_pos")},
            {"initial_pose.yaw": LaunchConfiguration("yaw")}
        ]
    )
    
    # 3. Lifecycle Manager (Quản lý trạng thái map_server và amcl)
    nav2_lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[
            {"node_names": ["map_server", "amcl"]},
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
            {"autostart": True}
        ]
    )

    return LaunchDescription([
        map_yaml_arg,
        use_sim_time_arg,
        x_pos_arg,
        y_pos_arg,
        yaw_arg,
        map_server_node,
        nav2_amcl,
        nav2_lifecycle_manager
    ])
