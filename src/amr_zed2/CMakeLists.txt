cmake_minimum_required(VERSION 3.16)
project(amr_zed2 LANGUAGES NONE)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(ament_cmake REQUIRED)

# Install packages using setup.py
set(${PROJECT_NAME}_INSTALL_OPTION "")
set(${PROJECT_NAME}_INSTALL_OPTION "${${PROJECT_NAME}_INSTALL_OPTION} --no-deps")
set(${PROJECT_NAME}_INSTALL_OPTION "${${PROJECT_NAME}_INSTALL_OPTION} --single-version-externally-managed")

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(
  FILES resource/${PROJECT_NAME}
  DESTINATION share/ament_index/resource_index/packages
)

ament_package()
