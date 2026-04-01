import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    launch_rviz = LaunchConfiguration("launch_rviz")
    world = LaunchConfiguration("world")
    run_mapping = LaunchConfiguration("run_mapping")
    run_localization = LaunchConfiguration("run_localization")
    use_rtabmap_localization = LaunchConfiguration("use_rtabmap_localization")
    run_mpc = LaunchConfiguration("run_mpc")

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
    run_mapping_arg = DeclareLaunchArgument(
        "run_mapping",
        default_value="false",
        description="Run mapping pipeline",
        choices=["true", "false"],
    )
    run_localization_arg = DeclareLaunchArgument(
        "run_localization",
        default_value="false",
        description="Run localization pipeline",
        choices=["true", "false"],
    )
    use_rtabmap_localization_arg = DeclareLaunchArgument(
        "use_rtabmap_localization",
        default_value="true",
        description="true: use RTAB-Map localization, false: use AMCL localization",
        choices=["true", "false"],
    )
    run_mpc_arg = DeclareLaunchArgument(
        "run_mpc",
        default_value="false",
        description="Run MPC controller node",
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
        condition=IfCondition(use_sim_time),
    )

    zed2_hw = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("zed2"),
                "launch",
                "zed2.launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": "false",
            "sim_mode": "false",
            "publish_svo_clock": "false",
        }.items(),
        condition=UnlessCondition(use_sim_time),
    )

    lidar_hw = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("lidar"),
                "launch",
                "a2m8.launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": "false",
        }.items(),
        condition=UnlessCondition(use_sim_time),
    )

    mapping = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("mapping"),
                "launch",
                "slam.launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
        }.items(),
        condition=IfCondition(run_mapping),
    )

    localization_amcl = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("localization"),
                "launch",
                "global_localization.launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
        }.items(),
        condition=IfCondition(PythonExpression([
            "'", run_localization, "' == 'true' and '", use_rtabmap_localization, "' == 'false'"
        ])),
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
        }.items(),
        condition=IfCondition(PythonExpression([
            "'", run_localization, "' == 'true' and '", use_rtabmap_localization, "' == 'true'"
        ])),
    )

    mpc = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("controller"),
                "launch",
                "ampcc_controller.launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
        }.items(),
        condition=IfCondition(run_mpc),
    )

    return LaunchDescription([
        use_sim_time_arg,
        launch_rviz_arg,
        world_arg,
        run_mapping_arg,
        run_localization_arg,
        use_rtabmap_localization_arg,
        run_mpc_arg,
        gazebo,
        zed2_hw,
        lidar_hw,
        mapping,
        localization_amcl,
        localization_rtabmap,
        mpc,
    ])