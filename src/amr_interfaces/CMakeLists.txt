cmake_minimum_required(VERSION 3.8)
project(amr_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

# Only build messages without circular dependencies
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ControlVar.msg"
  "msg/DynaObstacle.msg"
  "msg/HumanState.msg"
  "msg/HumanStateArray.msg"
  "msg/ObstacleState.msg"
  "msg/ObstacleStateArray.msg"
  "msg/Point.msg"
  "msg/WallState.msg"
  DEPENDENCIES std_msgs nav_msgs
)

ament_package()
