cmake_minimum_required(VERSION 3.16)
project(yolo)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install executable scripts
install(PROGRAMS
  scripts/multi_object_tracking
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch, config, models
install(
  DIRECTORY launch config models
  DESTINATION share/${PROJECT_NAME}
)

# Install resource marker
install(
  FILES resource/${PROJECT_NAME}
  DESTINATION share/ament_index/resource_index/packages
)

ament_package()
