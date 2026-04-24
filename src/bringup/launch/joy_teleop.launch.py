from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Node 1: Đọc phần cứng tay cầm joystick
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            parameters=[{'deadzone': 0.05}]
        ),
        # Node 2: Xử lý logic điều khiển JoyControl
        Node(
            package='teleoperation',
            executable='joy_control',
            name='joy_control',
            parameters=[
                {'linear_speed_limit': 0.3},
                {'angular_speed_limit': 0.5},
                {'smoothing_factor': 0.1}  # Bộ lọc làm mượt dừng
            ],
            # Remap topic để khớp với STM32 controller
            remappings=[('/cmd_vel', '/diff_cont/cmd_vel')]
        )
    ])
