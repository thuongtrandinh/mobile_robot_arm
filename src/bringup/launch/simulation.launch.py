import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    launch_rviz = LaunchConfiguration("launch_rviz")
    world = LaunchConfiguration("world")
    localization_ekf_config = LaunchConfiguration("localization_ekf_config")
    localization_cfg = LaunchConfiguration("localization_cfg")
    localization_database_path = LaunchConfiguration("localization_database_path")
    localization_namespace = LaunchConfiguration("localization_namespace")
    localization_map_name = LaunchConfiguration("localization_map_name")
    localization_map_yaml = LaunchConfiguration("localization_map_yaml")
    localization_use_map_server = LaunchConfiguration("localization_use_map_server")

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="true: simulation mode with Gazebo. false: real hardware sensors",
        choices=["true", "false"],
    )
    launch_rviz_arg = DeclareLaunchArgument(
        "launch_rviz",
        default_value="true",
        description="Launch RViz in simulation mode",
        choices=["true", "false"],
    )
    world_arg = DeclareLaunchArgument(
        "world",
        default_value="room_20x20.world",
        description="Gazebo world used when use_sim_time=true",
    )
    localization_ekf_config_arg = DeclareLaunchArgument(
        "localization_ekf_config",
        default_value=os.path.join(
            get_package_share_directory("localization"),
            "config",
            "ekf_sim.yaml",
        ),
        description="EKF config file used by localization in simulation",
    )
    localization_cfg_arg = DeclareLaunchArgument(
        "localization_cfg",
        default_value=os.path.join(
            get_package_share_directory("localization"),
            "config",
            "rtabmap_localization.yaml",
        ),
        description="RTAB-Map localization config file",
    )
    localization_database_path_arg = DeclareLaunchArgument(
        "localization_database_path",
        default_value="~/.ros/rtabmap_map.db",
        description="RTAB-Map database path used in localization mode",
    )
    localization_namespace_arg = DeclareLaunchArgument(
        "localization_namespace",
        default_value="rtabmap",
        description="Namespace for RTAB-Map nodes",
    )
    localization_map_name_arg = DeclareLaunchArgument(
        "localization_map_name",
        default_value="room_20x20",
        description="Map folder name under mapping/maps",
    )
    localization_map_yaml_arg = DeclareLaunchArgument(
        "localization_map_yaml",
        default_value="room_20x20_map.yaml",
        description="Map yaml filename in the selected map folder",
    )
    localization_use_map_server_arg = DeclareLaunchArgument(
        "localization_use_map_server",
        default_value="true",
        description="Enable Nav2 map_server when running RTAB-Map localization",
        choices=["true", "false"],
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("descriptions"),
                "launch",
                "gazebo.launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "launch_rviz": launch_rviz,
            "world": world,
        }.items(),
    )

    localization_rtabmap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("localization"),
                "launch",
                "global_localization_rtabmap.launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "ekf_config": localization_ekf_config,
            "cfg": localization_cfg,
            "database_path": localization_database_path,
            "namespace": localization_namespace,
            "map_name": localization_map_name,
            "map_yaml": localization_map_yaml,
            "use_map_server": localization_use_map_server,
        }.items(),
    )

    return LaunchDescription([
        use_sim_time_arg,
        launch_rviz_arg,
        world_arg,
        localization_ekf_config_arg,
        localization_cfg_arg,
        localization_database_path_arg,
        localization_namespace_arg,
        localization_map_name_arg,
        localization_map_yaml_arg,
        localization_use_map_server_arg,
        gazebo,
        localization_rtabmap,
    ])
