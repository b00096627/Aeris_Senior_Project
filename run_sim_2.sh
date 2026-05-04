#!/bin/bash

# Default Parameters
WORLD_DIR="crossroad"
GZ_WORLD_FILE="crossroad_world"
GZ_HEADLESS_FLAG=""

# Parse Command Line Arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d|--dir) WORLD_DIR="$2"; shift ;;
        -w|--world) GZ_WORLD_FILE="$2"; shift ;;
        --gz-headless) GZ_HEADLESS_FLAG="HEADLESS=1" ;;
        -h|--help) 
            echo "Usage: ./run_sim.sh [options]"
            echo "Options:"
            echo "  -d, --dir <name>      Folder name in sim_environments (default: straight_road)"
            echo "  -w, --world <name>    Gazebo .sdf world name (default: airport_road)"
            echo "  --gz-headless         Run Gazebo without the GUI"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

SESSION="px4_sim"

tmux has-session -t $SESSION 2>/dev/null
if [ $? == 0 ]; then
    echo "Session $SESSION already exists. Attaching..."
    tmux attach-session -t $SESSION
    exit 0
fi

echo "Starting parameterized simulation stack..."

# 0. Watchdog
tmux new-session -d -s $SESSION -n "Watchdog"
tmux send-keys -t $SESSION:0 "btop" C-m

# 1. DDS Agent (Must speak FastDDS natively)
tmux new-window -t $SESSION:1 -n "DDS"
tmux send-keys -t $SESSION:1 "cd ~/Micro-XRCE-DDS-Agent/" C-m
tmux send-keys -t $SESSION:1 "MicroXRCEAgent udp4 -p 8888" C-m

# 2. PX4 & Gazebo (Uses dynamic variables)
tmux new-window -t $SESSION:2 -n "PX4_GZ"
tmux send-keys -t $SESSION:2 "source /opt/ros/humble/setup.bash" C-m
tmux send-keys -t $SESSION:2 "source ~/px4_ros_com_ros2/install/setup.bash" C-m
tmux send-keys -t $SESSION:2 "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp" C-m
tmux send-keys -t $SESSION:2 "export FASTRTPS_DEFAULT_PROFILES_FILE=/home/aeris/fastdds_eth.xml" C-m
tmux send-keys -t $SESSION:2 "export FASTDDS_DEFAULT_PROFILES_FILE=/home/aeris/fastdds_eth.xml" C-m
tmux send-keys -t $SESSION:2 "export GZ_SIM_RESOURCE_PATH=\$GZ_SIM_RESOURCE_PATH:~/PX4-Autopilot/Tools/simulation/gz/models:~/PX4-Autopilot/Tools/simulation/gz/worlds" C-m
tmux send-keys -t $SESSION:2 "export IGN_GAZEBO_RESOURCE_PATH=\$IGN_GAZEBO_RESOURCE_PATH:~/PX4-Autopilot/Tools/simulation/gz/models:~/PX4-Autopilot/Tools/simulation/gz/worlds" C-m
tmux send-keys -t $SESSION:2 "cd ~/PX4-Autopilot" C-m
tmux send-keys -t $SESSION:2 "$GZ_HEADLESS_FLAG PX4_GZ_WORLD=$GZ_WORLD_FILE make px4_sitl gz_x500_dual_cam" C-m

# 3. SUMO Traffic Bridge + EMV ROS2 publisher (Waits 60 seconds to let drone spawn first)
tmux new-window -t $SESSION:3 -n "SUMO"
tmux send-keys -t $SESSION:3 "source /opt/ros/humble/setup.bash" C-m
tmux send-keys -t $SESSION:3 "source ~/Decision_Node/install/setup.bash" C-m
tmux send-keys -t $SESSION:3 "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp" C-m
tmux send-keys -t $SESSION:3 "export FASTRTPS_DEFAULT_PROFILES_FILE=/home/aeris/fastdds_eth.xml" C-m
tmux send-keys -t $SESSION:3 "export FASTDDS_DEFAULT_PROFILES_FILE=/home/aeris/fastdds_eth.xml" C-m
tmux send-keys -t $SESSION:3 "echo 'Waiting 60 seconds for Gazebo and PX4 to initialize...'" C-m
tmux send-keys -t $SESSION:3 "sleep 100" C-m
tmux send-keys -t $SESSION:3 "cd /home/aeris/sim_environments/bridges/ && python3 sumo_gz_bridge.py $WORLD_DIR" C-m

# 4. QGroundControl
tmux new-window -t $SESSION:4 -n "QGC"
tmux send-keys -t $SESSION:4 "echo 'Waiting 45 seconds for Gazebo and PX4 to initialize...'" C-m
tmux send-keys -t $SESSION:4 "sleep 45" C-m
tmux send-keys -t $SESSION:4 "cd ~ && ./QGroundControl-x86_64.AppImage" C-m

# 5. Dynamic ROS 2 Camera Bridge
tmux new-window -t $SESSION:5 -n "ROS2_Cam"
tmux send-keys -t $SESSION:5 "echo 'Waiting 65 seconds for Gazebo and PX4 to initialize...'" C-m
tmux send-keys -t $SESSION:5 "sleep 65" C-m
tmux send-keys -t $SESSION:5 "source /opt/ros/humble/setup.bash" C-m
tmux send-keys -t $SESSION:5 "source ~/px4_ros_com_ros2/install/setup.bash" C-m
tmux send-keys -t $SESSION:5 "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp" C-m
tmux send-keys -t $SESSION:5 "export FASTRTPS_DEFAULT_PROFILES_FILE=/home/aeris/fastdds_eth.xml" C-m
tmux send-keys -t $SESSION:5 "export FASTDDS_DEFAULT_PROFILES_FILE=/home/aeris/fastdds_eth.xml" C-m
tmux send-keys -t $SESSION:5 "ros2 run ros_gz_bridge parameter_bridge \
/world/$GZ_WORLD_FILE/model/x500_dual_cam_0/link/camera_link/sensor/camera_10/image@sensor_msgs/msg/Image[gz.msgs.Image \
/world/$GZ_WORLD_FILE/model/x500_dual_cam_0/link/camera_link/sensor/camera_90/image@sensor_msgs/msg/Image[gz.msgs.Image \
/world/$GZ_WORLD_FILE/model/x500_dual_cam_0/link/camera_link/sensor/depth_front/depth_image@sensor_msgs/msg/Image[gz.msgs.Image \
/world/$GZ_WORLD_FILE/model/x500_dual_cam_0/link/camera_link/sensor/depth_front/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo \
/world/$GZ_WORLD_FILE/model/x500_dual_cam_0/link/camera_link/sensor/depth_front/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked \
--ros-args \
-r /world/$GZ_WORLD_FILE/model/x500_dual_cam_0/link/camera_link/sensor/camera_10/image:=/camera/front_10 \
-r /world/$GZ_WORLD_FILE/model/x500_dual_cam_0/link/camera_link/sensor/camera_90/image:=/camera/down_90 \
-r /world/$GZ_WORLD_FILE/model/x500_dual_cam_0/link/camera_link/sensor/depth_front/depth_image:=/camera/depth_front/image_raw \
-r /world/$GZ_WORLD_FILE/model/x500_dual_cam_0/link/camera_link/sensor/depth_front/camera_info:=/camera/depth_front/camera_info \
-r /world/$GZ_WORLD_FILE/model/x500_dual_cam_0/link/camera_link/sensor/depth_front/points:=/camera/depth_front/points" C-m

# 6. Foxglove Viewer
tmux new-window -t $SESSION:6 -n "Foxglove"
tmux send-keys -t $SESSION:6 "echo 'Waiting 69 seconds for Gazebo and PX4 to initialize...'" C-m
tmux send-keys -t $SESSION:6 "sleep 69" C-m
tmux send-keys -t $SESSION:6 "source /opt/ros/humble/setup.bash" C-m
tmux send-keys -t $SESSION:6 "source ~/px4_ros_com_ros2/install/setup.bash" C-m
tmux send-keys -t $SESSION:6 "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp" C-m
tmux send-keys -t $SESSION:6 "export FASTRTPS_DEFAULT_PROFILES_FILE=/home/aeris/fastdds_eth.xml" C-m
tmux send-keys -t $SESSION:6 "export FASTDDS_DEFAULT_PROFILES_FILE=/home/aeris/fastdds_eth.xml" C-m
tmux send-keys -t $SESSION:6 "ros2 run foxglove_bridge foxglove_bridge --ros-args -p address:=0.0.0.0 -p port:=8765" C-m


tmux attach-session -t $SESSION