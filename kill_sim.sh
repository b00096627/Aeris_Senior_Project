#!/bin/bash

echo "[+] Shutting down tmux session..."
tmux kill-session -t px4_sim 2>/dev/null

echo "[+] Hunting down simulation zombies..."

# 1. Kill PX4 and Gazebo (Harmonic/Garden uses 'gz' and PX4 uses 'ruby' scripts to launch)
pkill -9 -f px4
pkill -9 -f "gz sim"
pkill -9 -f ruby

# 2. Kill the SUMO and Python bridge
pkill -9 -x sumo
pkill -9 -f bridge.py

# 3. Kill the DDS Agent
pkill -9 -x MicroXRCEAgent

# 4. Kill ROS 2 and bridges
pkill -9 -f ros2
pkill -9 -f ros_gz_bridge
pkill -9 -f foxglove_bridge

# 5. Kill QGroundControl
pkill -9 -f QGroundControl

echo "[+] Simulation environment completely sanitized. Check btop to verify."
