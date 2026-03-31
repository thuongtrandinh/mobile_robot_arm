import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    descriptions_pkg = get_package_share_directory('descriptions')
    lidar_pkg = get_package_share_directory('lidar')   
    zed_wrapper_pkg = get_package_share_directory('zed_wrapper') 
    
    custom_zed_config_dir = os.path.join(zed_wrapper_pkg, 'config')

    # 1. ROBOT STATE PUBLISHER
    xacro_file = os.path.join(descriptions_pkg, 'model', 'wheeled', 'urdf', 'mobile_robot.urdf.xacro')
    robot_description_content = ParameterValue(Command(['xacro ', xacro_file]), value_type=str)
    
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_content}]
    )

    # 3. DIFF DRIVE CONTROLLER SPAWNER
    # Node này sẽ kích hoạt bộ điều khiển và tạo ra topic /diffcont/odom
    diff_drive_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["diff_drive_controller"],
        remappings=[
            ('/diff_drive_controller/odom', '/diffcont/odom')
        ]
    )

    # 4. MICRO-ROS AGENT
    micro_ros_agent_node = Node(
        package='micro_ros_agent',
        executable='micro_ros_agent',
        name='micro_ros_agent',
        arguments=['serial', '--dev', '/dev/uart', '-b', '115200'], 
        output='screen'
    )

    # 5. LIDAR & ZED (Giữ nguyên như của bạn)
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(lidar_pkg, 'launch', 'a2m8.launch.py')),
        launch_arguments={'baud_rate': '256000', 'serial_port': '/dev/rplidar', 'frame_id': 'laser'}.items()
    )

    zed2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(zed_wrapper_pkg, 'launch', 'zed_camera.launch.py')),
        launch_arguments={'camera_model': 'zed2', 'config_path': custom_zed_config_dir}.items()
    )

    # TRẢ VỀ LAUNCH DESCRIPTION (Phải bao gồm tất cả các node ở trên)
    return LaunchDescription([
        robot_state_publisher_node,
        micro_ros_agent_node,
        diff_drive_spawner,         # Thêm vào đây
        
        TimerAction(period=5.0, actions=[lidar_launch]),
        TimerAction(period=10.0, actions=[zed2_launch])
    ])