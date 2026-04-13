cmake_minimum_required(VERSION 3.8)
project(ocp_planner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Find dependencies for ROS2
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(interfaces REQUIRED)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(casadi REQUIRED)
find_package(CURL REQUIRED)

################################################
## Build ##
################################################

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# ROS2 node executable
add_executable(${PROJECT_NAME}_node
  src/main.cc
  src/astar.cc
  src/smooth.cc
  src/lookahead.cc
  src/planner.cc
  src/mpc.cc
)

ament_target_dependencies(${PROJECT_NAME}_node
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  interfaces
)

target_compile_definitions(${PROJECT_NAME}_node PRIVATE ROS_BUILD)

target_link_libraries(${PROJECT_NAME}_node
  ${OpenCV_LIBRARIES}
  casadi
  CURL::libcurl
  dw
)

################################################
################# pybind11 #####################
################################################

<<<<<<< HEAD
<<<<<<< HEAD
option(BUILD_OCP_PYTHON_BINDINGS "Build pybind11 module for ocp_planner" ON)

if(BUILD_OCP_PYTHON_BINDINGS)
  # Prefer conda Python when available, otherwise fall back to system Python.
  if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/bin/python")
    set(Python3_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
  endif()

  find_package(Python3 COMPONENTS Interpreter Development.Module)
  if(Python3_FOUND)
    # Add pybind11 subdirectory
    add_subdirectory(extern/pybind11)

    # Build Python binding module (STANDALONE - no ROS dependency)
    pybind11_add_module(${PROJECT_NAME}_py
      src/ocp_bind.cc
      src/astar.cc
      src/smooth.cc
      src/lookahead.cc
      src/planner.cc
      src/mpc.cc
    )

    target_link_libraries(${PROJECT_NAME}_py
      PRIVATE
      ${OpenCV_LIBRARIES}
      casadi::casadi
    )
  else()
    message(WARNING "Python3 (Interpreter + Development.Module) not found. Skipping ocp_planner pybind module.")
  endif()
endif()
=======
<<<<<<< HEAD
# Use conda environment Python when available; otherwise fall back to system Python.
if(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
  set(Python3_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
endif()
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
=======
option(BUILD_OCP_PYTHON_BINDINGS "Build pybind11 module for ocp_planner" ON)
>>>>>>> 3ea914c (feat: fix error for ocp)
=======
option(BUILD_OCP_PYTHON_BINDINGS "Build pybind11 module for ocp_planner" OFF)
>>>>>>> a922a9a (fix: disable pybind11 python bindings due to missing extern/pybind11 submodule)

if(BUILD_OCP_PYTHON_BINDINGS)
  # Prefer conda Python when available, otherwise fall back to system Python.
  if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/bin/python")
    set(Python3_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
  endif()

  find_package(Python3 COMPONENTS Interpreter Development.Module)
  if(Python3_FOUND)
    # Add pybind11 subdirectory
    add_subdirectory(extern/pybind11)

    # Build Python binding module (STANDALONE - no ROS dependency)
    pybind11_add_module(${PROJECT_NAME}_py
      src/ocp_bind.cc
      src/astar.cc
      src/smooth.cc
      src/lookahead.cc
      src/planner.cc
      src/mpc.cc
    )

    target_link_libraries(${PROJECT_NAME}_py
      PRIVATE
      ${OpenCV_LIBRARIES}
      casadi::casadi
    )
  else()
    message(WARNING "Python3 (Interpreter + Development.Module) not found. Skipping ocp_planner pybind module.")
  endif()
endif()
>>>>>>> 4f6363e (feat: fix error for ocp)

################################################
## Install ##
################################################

install(TARGETS ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

if(TARGET ${PROJECT_NAME}_py)
  # Install Python binding to lib directory
  install(TARGETS ${PROJECT_NAME}_py
    LIBRARY DESTINATION lib
  )
endif()

ament_package()
