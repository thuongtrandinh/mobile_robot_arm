#!/bin/bash
set -e

# Backward-compatible entrypoint: use ROS2 build flow.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/build_cp_ros2.sh"
