cmake_minimum_required(VERSION 3.8)
project(controller)

# ---------- Compiler settings ----------
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------- ACADOS integration ----------
option(CONTROLLER_ENABLE_ACADOS "Enable ACADOS generated solver integration" ON)
set(ACADOS_INSTALL_DIR "$ENV{HOME}/acados")
set(ACADOS_GENERATED_CODE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/c_generated_code")

if(CONTROLLER_ENABLE_ACADOS)
  set(ACADOS_INCLUDE_DIRS
    ${ACADOS_INSTALL_DIR}/include
    ${ACADOS_INSTALL_DIR}/include/acados
    ${ACADOS_INSTALL_DIR}/include/blasfeo/include
    ${ACADOS_INSTALL_DIR}/include/hpipm/include
    ${ACADOS_GENERATED_CODE_DIR}
    ${ACADOS_GENERATED_CODE_DIR}/ddmr_model
  )

  include_directories(${ACADOS_INCLUDE_DIRS})

  if(EXISTS "${ACADOS_INSTALL_DIR}/lib")
    link_directories(${ACADOS_INSTALL_DIR}/lib)
  endif()
endif()

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
  src/mpc.cc
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
  casadi
)

target_compile_definitions(controller_core PUBLIC ROS_BUILD)

# ---------- Executable: ampcc_node ----------
add_executable(ampcc_node
  src/main.cc
)

if(CONTROLLER_ENABLE_ACADOS)
  set(ACADOS_GENERATED_SOURCES
    ${ACADOS_GENERATED_CODE_DIR}/acados_solver_ddmr.c
    ${ACADOS_GENERATED_CODE_DIR}/ddmr_model/ddmr_expl_ode_fun.c
    ${ACADOS_GENERATED_CODE_DIR}/ddmr_model/ddmr_expl_vde_forw.c
    ${ACADOS_GENERATED_CODE_DIR}/ddmr_model/ddmr_expl_vde_adj.c
  )

  foreach(_src IN LISTS ACADOS_GENERATED_SOURCES)
    if(EXISTS "${_src}")
      target_sources(ampcc_node PRIVATE ${_src})
    else()
      message(STATUS "ACADOS source not found yet: ${_src}")
    endif()
  endforeach()
endif()

target_include_directories(ampcc_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${EIGEN3_INCLUDE_DIRS}
)

target_compile_definitions(ampcc_node PRIVATE ROS_BUILD)

target_link_libraries(ampcc_node controller_core)

if(CONTROLLER_ENABLE_ACADOS)
  target_link_libraries(ampcc_node
    acados
    blasfeo
    hpipm
  )
endif()

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
    scripts/ampcc_rviz_visualizer.py
    scripts/custom_env_visualizer_node.py
    scripts/eval_ppo.py
    scripts/generate_acados_mpc.py
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
  TARGETS controller_core
  EXPORT controller_coreTargets
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
ament_export_targets(controller_coreTargets HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_libraries(controller_core)
ament_export_dependencies(rclcpp std_msgs geometry_msgs nav_msgs Eigen3)
ament_export_dependencies(interfaces)

# ---------- Testing ----------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
