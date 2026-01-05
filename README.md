# amr_framework

## Project Overview
> **Note**: This repository provides a modular framework for autonomous mobile robot (AMR) development, including navigation, localization, mapping, and hardware control.  
> Built with ROS 2 for scalable and maintainable robotics applications.

## Installation and Setup Instructions

### Prerequisites
- ROS 2 (Humble/Iron recommended)
- colcon build tool
- RPLidar ROS 2 driver

### Build Instructions
```bash
cd ~/amr_ws
colcon build
source install/setup.bash
```

## Repository Structure Overview

```
amr_ws/
└── src/
    ├── amr_hardware/                # Hardware interfaces and drivers
    │   ├── launch/
    │   ├── config/
    │   └── src/ (motor drivers, sensors, communication)
    ├── amr_descriptions/            # Robot URDF/Xacro models
    │   ├── launch/description.launch.py
    │   ├── urdf/amr.urdf.xacro
    │   ├── config/
    │   └── meshes/, materials/
    ├── amr_localization/            # Localization algorithms
    │   ├── launch/localization.launch.py
    │   ├── config/localization.yaml
    │   └── src/ (AMCL, EKF, sensor fusion)
    ├── amr_mapping/                 # SLAM and mapping
    │   ├── launch/mapping.launch.py
    │   ├── config/slam_config.yaml
    │   └── src/ (cartographer, slam_toolbox)
    ├── amr_planner/                 # Path planning and navigation
    │   ├── launch/planner.launch.py
    │   ├── config/nav2_params.yaml
    │   └── src/ (global/local planners)
    ├── amr_stm32/                   # STM32 firmware and interfaces
    │   ├── firmware/
    │   ├── scripts/
    │   └── config/
    ├── amr_teleop/                  # Teleoperation control
    │   ├── launch/teleop.launch.py
    │   └── src/ (keyboard, joystick control)
    ├── rplidar_ros/                 # RPLidar sensor driver
    │   └── launch/, config/, src/
    └── amr_bringup/                 # Launch all systems
        └── launch/bringup_all.launch.py               
```

## Contribution Guidelines

### Branching Model
- `main`: Stable, production-ready code. Maintainers only.
- `dev`: Active development and testing. Reviewed before merging.
- Feature branches (from `dev`):
  - Naming: `feature/<description>`
  - Bugfixes: `bugfix/<description>`
  - Examples: `feature/navigation-stack`, `bugfix/lidar-driver`

### Roles and Permissions
- Only maintainers can push directly to `main` and `dev`
- All other contributors:
  - Create a feature or bugfix branch from `dev`
  - Push changes and open a pull request (PR)

### Pull Request Workflow
- PRs must target `dev`
- Add reviewers and link relevant issues
- PRs require at least one approval before merging
- Maintainers merge to `main` after validation

## Contributors

**AMR Framework Team**
- [Thuong Tran Dinh](mailto:thuong.trandinh@hcmutcmut.edu.vn)

