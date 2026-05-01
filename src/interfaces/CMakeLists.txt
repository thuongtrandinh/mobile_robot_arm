cmake_minimum_required(VERSION 3.8)
project(interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

set(msg_files
  "msg/ControlVar.msg"
  "msg/DynaObstacle.msg"
  "msg/HumanArray.msg"
  "msg/HumanState.msg"
  "msg/JointState.msg"
  "msg/ObstacleArray.msg"
  "msg/ObstacleState.msg"
  "msg/Point.msg"
  "msg/PolyState.msg"
  "msg/RobotState.msg"
  "msg/WallState.msg"
)

set(srv_files
  "srv/OcpLocalPlann.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES std_msgs geometry_msgs nav_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
