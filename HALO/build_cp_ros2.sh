#!/bin/bash

# Build script for ROS2 Humble

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh  # hoặc anaconda3
conda activate mpc_rl

# Source ROS2 Humble
source /opt/ros/humble/setup.bash

# Build using colcon
cd /home/thuong/LVTN/amr_ws/HALO
colcon build --packages-select ocp_planner --symlink-install --cmake-args -Dcasadi_DIR=/usr/local/lib/cmake/casadi

# Remove old Python binding file
rm -f drl_moudle/ocp_planner_py.cpython-38-x86_64-linux-gnu.so

# Copy new Python binding file
cp install/lib/ocp_planner_py.cpython-38-x86_64-linux-gnu.so drl_moudle/

echo "Build completed successfully!"
echo "Python binding copied to drl_moudle/"
