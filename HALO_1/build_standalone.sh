#!/bin/bash
set -e

# Standalone build script (NO ROS REQUIRED)
echo "Building MPC Planner (Standalone - No ROS)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mpc_rl

cd "${SCRIPT_DIR}/src/ocp_planner"
mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"
make install

echo ""
echo "Build completed"
echo "Python binding installed to: drl_moudle/ocp_planner_py.cpython-38-x86_64-linux-gnu.so"
echo ""
echo "Now you can run training:"
echo "  cd ${SCRIPT_DIR}/drl_moudle"
echo "  conda activate mpc_rl"
echo "  python train_ppo.py"
