#!/bin/bash
set -e

# Build script for ROS2 Humble
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mpc_rl
source /opt/ros/humble/setup.bash

# Reuse external ROS interfaces package from the parent workspace.
if [[ -f "${SCRIPT_DIR}/../install/setup.bash" ]]; then
	source "${SCRIPT_DIR}/../install/setup.bash"
fi

cd "${SCRIPT_DIR}"
colcon build --packages-select ocp_planner --symlink-install --cmake-args -Dcasadi_DIR=/usr/local/lib/cmake/casadi

rm -f drl_moudle/ocp_planner_py.cpython-38-x86_64-linux-gnu.so

if [[ -f install/lib/ocp_planner_py.cpython-38-x86_64-linux-gnu.so ]]; then
	cp install/lib/ocp_planner_py.cpython-38-x86_64-linux-gnu.so drl_moudle/
elif [[ -f install/ocp_planner/lib/ocp_planner_py.cpython-38-x86_64-linux-gnu.so ]]; then
	cp install/ocp_planner/lib/ocp_planner_py.cpython-38-x86_64-linux-gnu.so drl_moudle/
else
	cp build/ocp_planner/ocp_planner_py.cpython-38-x86_64-linux-gnu.so drl_moudle/
fi

echo "Build completed successfully"
echo "Python binding copied to drl_moudle/"
