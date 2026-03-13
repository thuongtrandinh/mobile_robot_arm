cmake_minimum_required(VERSION 3.10)
project(ocp_planner_standalone)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Find only essential dependencies (NO ROS)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# Find Casadi from conda environment
set(CASADI_INCLUDE_DIR "$ENV{CONDA_PREFIX}/lib/python3.8/site-packages/casadi/include")
set(CASADI_LIB_DIR "$ENV{CONDA_PREFIX}/lib/python3.8/site-packages/casadi")

find_library(CASADI_LIBRARY 
    NAMES casadi
    PATHS ${CASADI_LIB_DIR}
    NO_DEFAULT_PATH
)

if(NOT CASADI_LIBRARY)
    message(STATUS "Trying to find casadi in system paths...")
    find_library(CASADI_LIBRARY casadi)
endif()

if(NOT CASADI_LIBRARY)
    message(FATAL_ERROR "Casadi library not found! Searched in: ${CASADI_LIB_DIR}")
else()
    message(STATUS "Found Casadi: ${CASADI_LIBRARY}")
endif()

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CASADI_INCLUDE_DIR}
)

################################################
################# pybind11 #####################
################################################

# Use conda environment Python
set(PYTHON_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
set(PYTHON_INCLUDE_DIRECTORY "$ENV{CONDA_PREFIX}/include/python3.8")

# Add pybind11
add_subdirectory(extern/pybind11)

# Build Python binding module
pybind11_add_module(ocp_planner_py
  src/ocp_bind.cc
  src/astar.cc
  src/smooth.cc
  src/lookahead.cc
  src/planner.cc
  src/mpc.cc
)

target_link_libraries(ocp_planner_py
  PRIVATE
  ${OpenCV_LIBRARIES}
  ${CASADI_LIBRARY}
  # dw library if needed
)

# Install to a known location
install(TARGETS ocp_planner_py
  LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../../drl_moudle
)
