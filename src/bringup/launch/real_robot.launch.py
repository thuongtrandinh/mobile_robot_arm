import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    # Khai báo đường dẫn các gói
    descriptions_pkg = get_package_share_directory('descriptions')
    lidar_pkg = get_package_share_directory('lidar')   
    zed_wrapper_pkg = get_package_share_directory('zed_wrapper') 
    
    # Đường dẫn file cấu hình
    xacro_file = os.path.join(descriptions_pkg, 'model', 'wheeled', 'urdf', 'mobile_robot.urdf.xacro')
    controller_config = os.path.join(descriptions_pkg, 'model', 'wheeled', 'config', 'ros2_control.yaml')
    custom_zed_config_dir = os.path.join(zed_wrapper_pkg, 'config')

    # 1. ROBOT STATE PUBLISHER (Truyền is_sim:=false vào Xacro)
    robot_description_content = ParameterValue(
        Command(['xacro ', xacro_file, ' is_sim:=false']), 
        value_type=str
    )
    
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description_content}],
        output='screen'
    )

    # 2. ROS2 CONTROL NODE (Manager)
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[{'robot_description': robot_description_content}, controller_config],
        output='screen'
    )

    # 3. SPAWNERS (Bộ điều khiển)
    # Khởi tạo Joint State Broadcaster
    joint_broad_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_broad"],
    )

    # Khởi tạo Differential Drive Controller (Odom & Cmd_vel)
    diff_drive_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["diff_cont"],
        remappings=[
            ('/diff_cont/odom', '/odom'),
            ('/diff_cont/cmd_vel_unstamped', '/cmd_vel')
        ]
    )

    # 4. MICRO-ROS AGENT
    micro_ros_agent_node = Node(
        package='micro_ros_agent',
        executable='micro_ros_agent',
        arguments=['serial', '--dev', '/dev/uart', '-b', '115200'], 
        output='screen'
    )

    # 5. SENSORS (Lidar & ZED2)
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(lidar_pkg, 'launch', 'a2m8.launch.py')),
        launch_arguments={'baud_rate': '256000', 'serial_port': '/dev/rplidar', 'frame_id': 'laser'}.items()
    )

    zed2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(zed_wrapper_pkg, 'launch', 'zed_camera.launch.py')),
        launch_arguments={'camera_model': 'zed2', 'config_path': custom_zed_config_dir}.items()
    )

    return LaunchDescription([
        # Khởi động ngay lập tức
        robot_state_publisher_node,
        micro_ros_agent_node,
        controller_manager,
        
        # Đợi tuần tự để ổn định hệ thống
        TimerAction(period=5.0, actions=[joint_broad_spawner]),
        TimerAction(period=5.0, actions=[diff_drive_spawner]),
        
        # Khởi động cảm biến sau cùng
        TimerAction(period=5.0, actions=[lidar_launch]),
        TimerAction(period=10.0, actions=[zed2_launch])
    ])