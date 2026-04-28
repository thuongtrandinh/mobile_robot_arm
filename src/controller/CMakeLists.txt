cmake_minimum_required(VERSION 3.8)
project(controller)

# ---------- Dependencies ----------
find_package(ament_cmake REQUIRED)

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
