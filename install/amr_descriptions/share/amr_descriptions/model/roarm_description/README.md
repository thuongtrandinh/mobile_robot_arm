# RoArm Robot Description

## Tổng quan
Package này chứa URDF description và cấu hình controllers cho robot arm RoArm.

## Cấu trúc file

### URDF Files
- `roarm.urdf.xacro` - URDF chính của robot (được generate từ CAD)
- `roarm.xacro` - File xacro tổng hợp, include tất cả các file cần thiết
- `roarm_gazebo.xacro` - Gazebo plugins và material properties
- `roarm_ros2_control.xacro` - Cấu hình ros2_control interface

### Config Files
- `roarm_controllers.yaml` - Cấu hình cho các controllers (joint_state_broadcaster, arm_controller)

### Launch Files
- `roarm_description.launch.py` - Chỉ load URDF và robot_state_publisher (không có RViz)
- `roarm_bringup.launch.py` - Load URDF + Controllers (cho robot thật)
- `roarm_gazebo.launch.py` - Launch đầy đủ với Gazebo simulation

## Joints
Robot có 4 joints chính:
1. `base_link_to_link1` - Base rotation (-π to π)
2. `link1_to_link2` - Shoulder joint (-π/2 to π/2)
3. `link2_to_link3` - Elbow joint (-1.0 to π)
4. `link3_to_gripper_link` - Gripper/Wrist joint (0 to 1.5)

## Sử dụng

### 1. Chỉ load URDF (không RViz, không Gazebo)
```bash
ros2 launch roarm_description roarm_description.launch.py
```
Điều này sẽ launch:
- robot_state_publisher
- joint_state_publisher_gui (để điều khiển các joints)

### 2. Load URDF + Controllers (cho robot thật)
```bash
ros2 launch roarm_description roarm_bringup.launch.py use_sim_time:=false
```
Điều này sẽ launch:
- robot_state_publisher
- ros2_control_node (controller manager)
- joint_state_broadcaster
- arm_controller (joint trajectory controller)

### 3. Simulation với Gazebo
```bash
ros2 launch roarm_description roarm_gazebo.launch.py
```
Điều này sẽ launch:
- Gazebo
- robot_state_publisher
- Spawn robot vào Gazebo
- Controllers (joint_state_broadcaster + arm_controller)

### 4. Xem trong RViz (nếu cần)
Mở terminal khác và chạy:
```bash
ros2 run rviz2 rviz2
```
Sau đó add:
- RobotModel (để xem robot)
- TF (để xem coordinate frames)

## Controllers

### Joint State Broadcaster
Publish trạng thái của tất cả các joints lên topic `/joint_states`

### Arm Controller (Joint Trajectory Controller)
- Topic: `/arm_controller/joint_trajectory`
- Type: `trajectory_msgs/JointTrajectory`

Ví dụ điều khiển:
```bash
ros2 topic pub /arm_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "{
  joint_names: ['base_link_to_link1', 'link1_to_link2', 'link2_to_link3', 'link3_to_gripper_link'],
  points: [
    {positions: [0.0, 0.0, 0.0, 0.0], time_from_start: {sec: 2}}
  ]
}" --once
```

## Kiểm tra Controllers
```bash
# List tất cả controllers
ros2 control list_controllers

# Kiểm tra hardware interface
ros2 control list_hardware_interfaces
```

## Dependencies
- ros2_control
- ros2_controllers
- gazebo_ros2_control (cho simulation)
- joint_state_publisher_gui (optional)
- robot_state_publisher
- xacro

## Notes
- File `roarm.urdf.xacro` được generate từ CAD và không nên chỉnh sửa trực tiếp
- Tất cả customization nên được thêm vào `roarm.xacro`, `roarm_gazebo.xacro`, hoặc `roarm_ros2_control.xacro`
- Các file này đã được clean up để không phụ thuộc vào package `mobile_robot` nữa
