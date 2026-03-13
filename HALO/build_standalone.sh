#!/bin/bash

# Standalone build script (NO ROS REQUIRED)

echo "Building MPC Planner (Standalone - No ROS)"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mpc_rl

# Create build directory
cd /home/thuong/LVTN/amr_ws/HALO/src/ocp_planner
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Install (copies .so to drl_moudle/)
make install

echo ""
echo "✅ Build completed!"
echo "Python binding installed to: drl_moudle/ocp_planner_py.cpython-38-x86_64-linux-gnu.so"
echo ""
echo "Now you can run training:"
echo "  cd ~/LVTN/HALO/drl_moudle"
echo "  conda activate mpc_rl"
echo "  python train_ppo.py"
