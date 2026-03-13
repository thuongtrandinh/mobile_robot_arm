#!/bin/bash

# Build script for ROS2 Humble

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mpc_rl
source /opt/ros/humble/setup.bash

cd /home/thuong/LVTN/amr_ws/HALO
colcon build --packages-select ocp_planner --symlink-install --cmake-args -Dcasadi_DIR=/usr/local/lib/cmake/casadi

rm -f drl_moudle/ocp_planner_py.cpython-38-x86_64-linux-gnu.so
cp install/lib/ocp_planner_py.cpython-38-x86_64-linux-gnu.so drl_moudle/
