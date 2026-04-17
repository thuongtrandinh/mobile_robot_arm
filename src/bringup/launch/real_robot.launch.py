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
    realsense2_camera_pkg = get_package_share_directory('realsense2_camera') 
    
    # Đường dẫn file cấu hình
    xacro_file = os.path.join(descriptions_pkg, 'model', 'wheeled', 'urdf', 'mobile_robot.urdf.xacro')
    controller_config = os.path.join(descriptions_pkg, 'model', 'wheeled', 'config', 'ros2_control.yaml')
    
    # Đường dẫn config D435i
    d435i_config = os.path.join(descriptions_pkg, 'model', 'wheeled', 'config', 'd435i.yaml')

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

    # 2. ODOMETRY CALCULATOR NODE
    # Nút Python tính toán odometry từ joint_states và phát ra /diff_cont/odom cho EKF
    odom_calculator_node = Node(
        package='bringup',
        executable='odom_calculator.py',
        name='diff_cont',  # Đặt tên này để nó tự vào yaml đọc thông số wheel_radius, wheel_separation
        parameters=[controller_config],
        output='screen'
    )

    # 4. MICRO-ROS AGENT
    micro_ros_agent_node = Node(
        package='micro_ros_agent',
        executable='micro_ros_agent',
        arguments=['serial', '--dev', '/dev/uart', '-b', '115200'], 
        output='screen'
    )

    # 5. SENSORS (Lidar & D435i Camera)
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(lidar_pkg, 'launch', 'a2m8.launch.py')),
        launch_arguments={'baud_rate': '256000', 'serial_port': '/dev/rplidar', 'frame_id': 'laser'}.items()
    )

    d435i_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(realsense2_camera_pkg, 'launch', 'rs_launch.py')),
        launch_arguments={
            'camera_name': 'camera',
            'camera_namespace': 'camera',
            'config_file': d435i_config,
            'device_type': 'd435i',
            'output': 'screen',
        }.items()
    )

    return LaunchDescription([
        # Khởi động ngay lập tức
        robot_state_publisher_node,
        odom_calculator_node,
        # micro_ros_agent_node,
        
        # Khởi động cảm biến sau cùng
        TimerAction(period=5.0, actions=[lidar_launch]),
        TimerAction(period=10.0, actions=[d435i_launch])
    ])