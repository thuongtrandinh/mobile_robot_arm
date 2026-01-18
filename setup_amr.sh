#!/bin/bash
# Disable Fast-DDS shared memory transport to avoid RTPS_TRANSPORT_SHM errors
export FASTRTPS_DEFAULT_PROFILES_FILE="$HOME/amr_ws/.fastdds_profile.xml"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Source ROS2 workspace
source /opt/ros/humble/setup.bash
source $HOME/amr_ws/install/setup.bash

echo "✅ AMR workspace loaded (SHM transport disabled)"
