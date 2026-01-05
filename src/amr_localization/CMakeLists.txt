cmake_minimum_required(VERSION 3.8)
project(amr_localization)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(include)

# C++ executables commented out - source files not present
# add_executable(aruco_detector_sim src/aruco_detector_sim.cpp)
# ament_target_dependencies(aruco_detector_sim
#   rclcpp sensor_msgs geometry_msgs std_msgs visualization_msgs
#   cv_bridge tf2 tf2_ros tf2_geometry_msgs
# )
# target_link_libraries(aruco_detector_sim ${OpenCV_LIBRARIES} yaml-cpp)
# 
# # New: ArUco Map Localizer - uses pre-defined marker positions for absolute localization
# add_executable(aruco_map_localizer src/aruco_map_localizer.cpp)
# ament_target_dependencies(aruco_map_localizer
#   rclcpp sensor_msgs geometry_msgs visualization_msgs
#   cv_bridge tf2 tf2_geometry_msgs
# )
# target_link_libraries(aruco_map_localizer ${OpenCV_LIBRARIES} yaml-cpp)
# 
# add_executable(kalman_filter src/kalman_filter.cpp)
# ament_target_dependencies(kalman_filter rclcpp nav_msgs sensor_msgs)
# 
# add_executable(imu_republisher src/imu_republisher.cpp)
# ament_target_dependencies(imu_republisher rclcpp sensor_msgs)
# 
# add_executable(odom_republisher src/odom_republisher.cpp)
# ament_target_dependencies(odom_republisher rclcpp nav_msgs)
# 
# add_executable(odometry_motion_model src/odometry_motion_model.cpp)
# ament_target_dependencies(odometry_motion_model rclcpp tf2 nav_msgs geometry_msgs)

ament_python_install_package(agv_localization)

# Commented out - targets don't exist
# install(
#   DIRECTORY include
#   DESTINATION include
# )
# 
# install(TARGETS
#   odometry_motion_model
#   LIBRARY DESTINATION lib
#   ARCHIVE DESTINATION lib
#   RUNTIME DESTINATION bin
# )
# 
# install(TARGETS
#   aruco_detector_sim
#   aruco_map_localizer
#   kalman_filter
#   imu_republisher
#   odom_republisher
#   odometry_motion_model
#   DESTINATION lib/${PROJECT_NAME}
# )

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  agv_localization/kalman_filter.py
  agv_localization/imu_republisher.py
  agv_localization/odom_republisher.py
  agv_localization/odometry_motion_model.py
  agv_localization/fusion_monitor.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

