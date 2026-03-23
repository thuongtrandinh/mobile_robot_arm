cmake_minimum_required(VERSION 3.8)
project(amr_controller)

# ---------- Compiler settings ----------
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------- Dependencies ----------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(amr_interfaces REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# ---------- Library: amr_controller_core ----------
add_library(amr_controller_core SHARED
  src/astar.cc
  src/lookahead.cc
  src/mpc.cc
  src/planner.cc
  src/smooth.cc
)

target_include_directories(amr_controller_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(amr_controller_core
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  amr_interfaces
)

target_link_libraries(amr_controller_core
  ${OpenCV_LIBS}
  casadi
)

target_compile_definitions(amr_controller_core PUBLIC ROS_BUILD)

# ---------- Executable: ampcc_node ----------
add_executable(ampcc_node
  src/main.cc
)

target_include_directories(ampcc_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${EIGEN3_INCLUDE_DIRS}
)

target_compile_definitions(ampcc_node PRIVATE ROS_BUILD)

target_link_libraries(ampcc_node amr_controller_core)

ament_target_dependencies(ampcc_node
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  tf2
  amr_interfaces
)

# ---------- Install ----------
install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY policy
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/eval_ppo.py
    scripts/rl_ocp_policy_bridge.py
    scripts/rl_policy_worker.py
  DESTINATION lib/${PROJECT_NAME}
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config")
  install(
    DIRECTORY config
    DESTINATION share/${PROJECT_NAME}
  )
endif()

install(
  TARGETS amr_controller_core
  EXPORT amr_controller_coreTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(
  TARGETS ampcc_node
  DESTINATION lib/${PROJECT_NAME}
)

# ---------- Export ----------
ament_export_targets(amr_controller_coreTargets HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_libraries(amr_controller_core)
ament_export_dependencies(rclcpp std_msgs geometry_msgs nav_msgs Eigen3)
ament_export_dependencies(amr_interfaces)

# ---------- Testing ----------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
