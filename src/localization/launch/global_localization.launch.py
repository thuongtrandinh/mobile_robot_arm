import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

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

def launch_setup(context, *args, **kwargs):
    localization_dir = get_package_share_directory("localization")
    use_sim_time = LaunchConfiguration("use_sim_time")

    use_sim_time_str = context.launch_configurations.get("use_sim_time", "true").strip().lower()
    ekf_config_file = "ekf_sim.yaml" if use_sim_time_str == "true" else "ekf.yaml"
    ekf_config_path = os.path.join(localization_dir, "config", ekf_config_file)

    map_server_node = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[
            {"yaml_filename": LaunchConfiguration("map")},
            {"use_sim_time": use_sim_time},
        ],
    )

    ekf_filter_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[
            ekf_config_path,
            {"use_sim_time": use_sim_time},
        ],
    )

    amcl_config = os.path.join(localization_dir, "config", "amcl.yaml")
    nav2_amcl = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        parameters=[
            amcl_config,
            {"use_sim_time": use_sim_time},
            {"initial_pose.x": LaunchConfiguration("x_pos")},
            {"initial_pose.y": LaunchConfiguration("y_pos")},
            {"initial_pose.yaw": LaunchConfiguration("yaw")},
        ],
    )

    nav2_lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[
            {"node_names": ["map_server", "amcl"]},
            {"use_sim_time": use_sim_time},
            {"autostart": True},
        ],
    )

    return [
        map_server_node,
        ekf_filter_node,
        nav2_amcl,
        nav2_lifecycle_manager
    ])
