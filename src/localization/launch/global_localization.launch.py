import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    localization_dir = get_package_share_directory("localization")
    localization_prefix = get_package_prefix("localization")
    workspace_root = os.path.dirname(os.path.dirname(localization_prefix))
    maps_root = os.path.join(workspace_root, "src", "mapping", "maps")
    use_sim_time = LaunchConfiguration("use_sim_time")

    use_sim_time_value = LaunchConfiguration("use_sim_time").perform(context).strip().lower()
    ekf_config_file = "ekf_sim.yaml" if use_sim_time_value == "true" else "ekf.yaml"
    ekf_config_path = os.path.join(localization_dir, "config", ekf_config_file)

    map_server_node = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[
            {
                "yaml_filename": PathJoinSubstitution([
                    maps_root,
                    LaunchConfiguration("map_name"),
                    LaunchConfiguration("map_yaml"),
                ]),
                "use_sim_time": use_sim_time,
            },
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

    amcl_node = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        parameters=[
            os.path.join(localization_dir, "config", "amcl.yaml"),
            {"use_sim_time": use_sim_time},
            {"set_initial_pose": True},
            {"initial_pose.x": LaunchConfiguration("x_pos")},
            {"initial_pose.y": LaunchConfiguration("y_pos")},
            {"initial_pose.yaw": LaunchConfiguration("yaw")},
        ],
    )

    lifecycle_manager_node = Node(
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
        amcl_node,
        lifecycle_manager_node,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            choices=["true", "false"],
        ),
        DeclareLaunchArgument(
            "map_name",
            default_value="room_20x20",
            description="Map folder name under mapping/maps",
        ),
        DeclareLaunchArgument(
            "map_yaml",
            default_value="room_20x20_map.yaml",
            description="Map yaml filename in the selected map folder",
        ),
        DeclareLaunchArgument(
            "x_pos",
            default_value="0.0",
            description="Initial robot x position in the map frame",
        ),
        DeclareLaunchArgument(
            "y_pos",
            default_value="0.0",
            description="Initial robot y position in the map frame",
        ),
        DeclareLaunchArgument(
            "yaw",
            default_value="0.0",
            description="Initial robot yaw in radians in the map frame",
        ),
        OpaqueFunction(function=launch_setup),
    ])
