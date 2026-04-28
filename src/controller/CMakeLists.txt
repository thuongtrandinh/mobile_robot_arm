cmake_minimum_required(VERSION 3.8)
project(controller)

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
find_package(interfaces REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TIFF REQUIRED)
find_package(GDAL REQUIRED)

# ---------- Library: controller_core ----------
add_library(controller_core SHARED
  src/astar.cc
  src/lookahead.cc
  src/planner.cc
  src/smooth.cc
)

target_include_directories(controller_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(controller_core
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  interfaces
)

target_link_libraries(controller_core
  ${OpenCV_LIBS}
  TIFF::TIFF
  GDAL::GDAL
)

target_compile_definitions(controller_core PUBLIC ROS_BUILD)

# ---------- Executable: ampcc_node ----------
add_executable(ampcc_node
  src/main.cc
)

target_include_directories(ampcc_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${EIGEN3_INCLUDE_DIRS}
)

target_compile_definitions(ampcc_node PRIVATE ROS_BUILD)

target_link_libraries(ampcc_node controller_core)

ament_target_dependencies(ampcc_node
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  tf2
  interfaces
)

# ---------- Install ----------
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY policy
  DESTINATION share/${PROJECT_NAME}
)

set(CONTROLLER_SCRIPT_FILES
  scripts/ampcc_rviz_visualizer.py
  scripts/custom_env_visualizer_node.py
  scripts/eval_ppo.py
  scripts/rl_ocp_policy_bridge.py
  scripts/rl_policy_worker.py
)

set(CONTROLLER_EXISTING_SCRIPT_FILES "")
foreach(_script IN LISTS CONTROLLER_SCRIPT_FILES)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_script}")
    list(APPEND CONTROLLER_EXISTING_SCRIPT_FILES ${_script})
  else()
    message(STATUS "Skip missing script in install(PROGRAMS): ${_script}")
  endif()
endforeach()

if(CONTROLLER_EXISTING_SCRIPT_FILES)
  install(
    PROGRAMS ${CONTROLLER_EXISTING_SCRIPT_FILES}
    DESTINATION lib/${PROJECT_NAME}
  )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config")
  install(
    DIRECTORY config
    DESTINATION share/${PROJECT_NAME}
  )
endif()

# ---------- Testing ----------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
