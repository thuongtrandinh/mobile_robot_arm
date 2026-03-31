import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_slam = LaunchConfiguration("use_slam")
    run_lidar = LaunchConfiguration("run_lidar")
    run_localization = LaunchConfiguration("run_localization")
    run_controller = LaunchConfiguration("run_controller")
    run_zed2 = LaunchConfiguration("run_zed2")
    run_ekf = LaunchConfiguration("run_ekf")

    map_name = LaunchConfiguration("map_name")
    map_yaml = LaunchConfiguration("map_yaml")
    x_pos = LaunchConfiguration("x_pos")
    y_pos = LaunchConfiguration("y_pos")
    yaw = LaunchConfiguration("yaw")

    lidar_serial_port = LaunchConfiguration("lidar_serial_port")
    lidar_serial_baudrate = LaunchConfiguration("lidar_serial_baudrate")
    lidar_frame_id = LaunchConfiguration("lidar_frame_id")

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation clock. Must stay false for real robot.",
        choices=["true", "false"],
    )

    use_slam_arg = DeclareLaunchArgument(
        "use_slam",
        default_value="false",
        description="true: run SLAM. false: run AMCL localization on saved map.",
        choices=["true", "false"],
    )

    run_lidar_arg = DeclareLaunchArgument(
        "run_lidar",
        default_value="true",
        description="Start the RPLidar driver.",
        choices=["true", "false"],
    )

    run_localization_arg = DeclareLaunchArgument(
        "run_localization",
        default_value="false",
        description="Enable localization pipeline (AMCL or SLAM).",
        choices=["true", "false"],
    )

    run_controller_arg = DeclareLaunchArgument(
        "run_controller",
        default_value="false",
        description="Start AMPCC controller node.",
        choices=["true", "false"],
    )

    run_zed2_arg = DeclareLaunchArgument(
        "run_zed2",
        default_value="false",
        description="Start ZED2 camera wrapper.",
        choices=["true", "false"],
    )

    run_ekf_arg = DeclareLaunchArgument(
        "run_ekf",
        default_value="true",
        description="Run EKF in SLAM mode for /odometry/filtered.",
        choices=["true", "false"],
    )

    map_name_arg = DeclareLaunchArgument(
        "map_name",
        default_value="room_20x20",
        description="Map folder name under mapping/maps.",
    )

    map_yaml_arg = DeclareLaunchArgument(
        "map_yaml",
        default_value="map.yaml",
        description="Map YAML file inside selected map folder.",
    )

    x_pos_arg = DeclareLaunchArgument(
        "x_pos",
        default_value="0.0",
        description="Initial pose X for AMCL.",
    )

    y_pos_arg = DeclareLaunchArgument(
        "y_pos",
        default_value="0.0",
        description="Initial pose Y for AMCL.",
    )

    yaw_arg = DeclareLaunchArgument(
        "yaw",
        default_value="0.0",
        description="Initial yaw (rad) for AMCL.",
    )

    lidar_serial_port_arg = DeclareLaunchArgument(
        "lidar_serial_port",
        default_value="/dev/ttyUSB0",
        description="RPLidar serial device.",
    )

    lidar_serial_baudrate_arg = DeclareLaunchArgument(
        "lidar_serial_baudrate",
        default_value="256000",
        description="RPLidar serial baudrate.",
    )

    lidar_frame_id_arg = DeclareLaunchArgument(
        "lidar_frame_id",
        default_value="laser",
        description="TF frame id published by lidar node.",
    )



    ekf = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[
            os.path.join(
                get_package_share_directory("localization"),
                "config",
                "ekf_zed2.yaml",
            ),
            {"use_sim_time": use_sim_time},
        ],
        condition=IfCondition(
            PythonExpression([
                "'", run_localization, "' == 'true' and '", use_slam, "' == 'true' and '", run_ekf, "' == 'true'"
            ])
        ),
    )

    lidar_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("lidar"),
                "launch",
                "a2m8.launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "serial_port": lidar_serial_port,
            "serial_baudrate": lidar_serial_baudrate,
            "frame_id": lidar_frame_id,
        }.items(),
        condition=IfCondition(run_lidar),
    )
 
    controller = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("controller"),
                "launch",
                "ampcc_controller.launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "respawn": "false",
            "log_level": "info",
        }.items(),
        condition=IfCondition(run_controller),
    )

    zed2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("zed2"),
                "launch",
                "zed2.launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "sim_mode": "false",
            "publish_svo_clock": "false",
        }.items(),
        condition=IfCondition(run_zed2),
    )

    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("localization"),
                "launch",
                "global_localization.launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "map_name": map_name,
            "map_yaml": map_yaml,
            "x_pos": x_pos,
            "y_pos": y_pos,
            "yaw": yaw,
        }.items(),
        condition=IfCondition(
            PythonExpression(["'", run_localization, "' == 'true' and '", use_slam, "' == 'false'"])
        ),
    )

    slam = IncludeLaunchDescription(
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
        condition=IfCondition(
            PythonExpression(["'", run_localization, "' == 'true' and '", use_slam, "' == 'true'"])
        ),
    )

    return LaunchDescription([
        use_sim_time_arg,
        use_slam_arg,
        run_lidar_arg,
        run_localization_arg,
        run_controller_arg,
        run_zed2_arg,
        run_ekf_arg,
        map_name_arg,
        map_yaml_arg,
        x_pos_arg,
        y_pos_arg,
        yaw_arg,
        lidar_serial_port_arg,
        lidar_serial_baudrate_arg,
        lidar_frame_id_arg,
        ekf,
        lidar_driver,
        controller,
        zed2,
        localization,
        slam,
    ])