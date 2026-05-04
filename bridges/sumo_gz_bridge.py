import os
import subprocess
import math
import time
import random
import argparse
import yaml
import libsumo
import traci.constants as tc

from gz.transport13 import Node as GzNode
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.pose_v_pb2 import Pose_V

try:
    import rclpy
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Header, Float64
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import NavSatFix
    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False

from emv_pipeline import (
    _EMV_MSGS_AVAILABLE, EMV_CFG,
    CongestionSummary, DecisionCommand, TargetWaypoint, RerouteStatus,
    set_initial_route,
    publish_congestion, publish_ambulance_gps,
    run_decision_pipeline, run_rerouter, publish_rerouter_status,
    init_pilot_car_ghost,
)

# --- CAR ASSETS ---
RED_CAR_SDF_PATH    = "/home/aeris/sim_environments/3d_assets/red_car.sdf"
BLUE_CAR_SDF_PATH   = "/home/aeris/sim_environments/3d_assets/blue_car.sdf"
GREEN_CAR_SDF_PATH  = "/home/aeris/sim_environments/3d_assets/green_car.sdf"
PURPLE_CAR_SDF_PATH = "/home/aeris/sim_environments/3d_assets/purple_car.sdf"
AMBULANCE_SDF_PATH  = "/home/aeris/sim_environments/3d_assets/ambulance.sdf"

CAR_SDF_PATHS = [
    RED_CAR_SDF_PATH, BLUE_CAR_SDF_PATH, GREEN_CAR_SDF_PATH, PURPLE_CAR_SDF_PATH
]


def main():
    parser = argparse.ArgumentParser(description="Run the SUMO-Gazebo bridge.")
    parser.add_argument("env_name", type=str, help="The environment folder (e.g., straight_road)")
    args = parser.parse_args()

    env_name   = args.env_name
    base_path  = f"/home/aeris/sim_environments/{env_name}"
    sumo_cfg   = os.path.join(base_path, "sumo_files", f"{env_name}.sumocfg")
    calib_file = os.path.join(base_path, "calibration", f"{env_name}_calib.yaml")

    if not os.path.exists(calib_file):
        print(f"[-] Error: Calibration file not found at {calib_file}")
        return

    with open(calib_file, 'r') as file:
        config = yaml.safe_load(file)

    OFFSET_X    = config['calibration']['offset_x']
    OFFSET_Y    = config['calibration']['offset_y']
    OFFSET_Z    = config['calibration']['offset_z']
    STEP_LENGTH = config['simulation']['step_length']
    RT_STEP     = STEP_LENGTH * 60.0 / 60.0  # 1000 sim-seconds = 60 real seconds
    WORLD_NAME  = config['simulation']['gazebo_world_name']

    gz_node = GzNode()
    print("[+] Gazebo Transport Node initialized.")

    drone_gz_pos = [0.0, 0.0]

    sumo_cmd = [
        "sumo-gui", "-c", sumo_cfg,
        "--step-length", str(STEP_LENGTH),
        "--start",
        "--collision.action", "none",
    ]
    libsumo.start(sumo_cmd)
    print(f"[+] Libsumo loaded {sumo_cfg}. Waiting for traffic...")

    ros_node       = None
    pub_apf        = None
    pub_congestion = None
    pub_gps        = None
    pub_decision   = None
    pub_clearance  = None
    pub_reroute    = None
    pub_waypoint   = None
    pub_status     = None

    if _ROS2_AVAILABLE:
        _RELIABLE_QOS = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        rclpy.init()
        ros_node = rclpy.create_node("sumo_gz_bridge_node")
        pub_apf  = ros_node.create_publisher(Odometry, "/apf/goal_odom", _RELIABLE_QOS)
        print("[+] ROS2 bridge node started. Publishes: /apf/goal_odom")

        if _EMV_MSGS_AVAILABLE:
            pub_congestion = ros_node.create_publisher(
                CongestionSummary, "/drone/congestion_summary", 10)
            pub_gps = ros_node.create_publisher(
                NavSatFix, "/ambulance/gps", 10)
            pub_decision = ros_node.create_publisher(
                DecisionCommand, "/ambulance/decision", 10)
            pub_clearance = ros_node.create_publisher(
                DecisionCommand, "/drone/clearance_trigger", 10)
            pub_reroute = ros_node.create_publisher(
                DecisionCommand, "/rerouter/command", 10)
            pub_waypoint = ros_node.create_publisher(
                TargetWaypoint, "/ambulance/target_waypoint", 10)
            pub_status = ros_node.create_publisher(
                RerouteStatus, "/rerouter/status", 10)
            print("[+] EMV publishers created.")

    active_cars       = {}
    last_apf_pub_time = 0.0
    last_pilot_pos    = None  # (gz_x, gz_y, t) for velocity finite-difference
    hidden_cars       = set()

    emv_state = {
        "mode":                  "SURVEY",
        "clearance_start_time":  0.0,
        "cached_alternates":     [],
        "last_fetch_time":       0.0,
        "current_route_cost":    0.0,
        "last_densities":        [],
        "initial_route_set":     False,
        "pilot_car_ghost_init":  False,
    }

    rerouter_state = {
        "state":                      "IDLE",
        "active_route_name":          "",
        "waypoints":                  [],
        "current_waypoint_index":     0,
        "current_lat":                0.0,
        "current_lon":                0.0,
        "reroute_start_time":         0.0,
        "last_waypoint_advance_time": 0.0,
        "last_failure_reason":        "",
    }

    last_congestion_pub_time  = 0.0
    last_decision_eval_time   = 0.0
    last_rerouter_status_time = 0.0
    ambulance_gps             = (0.0, 0.0)

    try:
        while True:
            loop_start_time = time.time()

            newly_spawned   = []
            cars_to_despawn = []
            apf_msg         = None

            if libsumo.simulation.getMinExpectedNumber() <= 0:
                break

            libsumo.simulationStep()

            # --- SPAWN NEW CARS ---
            for car_id in libsumo.simulation.getDepartedIDList():
                if car_id == "pilot_car":
                    continue
                if car_id in active_cars:
                    continue
                v_class = libsumo.vehicle.getVehicleClass(car_id)
                libsumo.vehicle.subscribe(car_id, [tc.VAR_POSITION, tc.VAR_ANGLE])
                active_cars[car_id] = time.time()
                newly_spawned.append((car_id, v_class))

            # --- DESPAWN FINISHED CARS ---
            for car_id in libsumo.simulation.getArrivedIDList():
                if car_id in active_cars:
                    del active_cars[car_id]
                    cars_to_despawn.append(car_id)

            sub_results     = libsumo.vehicle.getAllSubscriptionResults()
            active_snapshot = dict(active_cars)

            # ── EMV: Initial route when ambulance spawns ──────────────────────
            amb_id = EMV_CFG["ambulance_vehicle_id"]
            if (_EMV_MSGS_AVAILABLE
                    and amb_id in libsumo.vehicle.getIDList()
                    and not emv_state["initial_route_set"]):
                set_initial_route()
                emv_state["initial_route_set"] = True
                libsumo.vehicle.subscribe(amb_id, [tc.VAR_POSITION, tc.VAR_ANGLE])

            # ── PILOT CAR: one-time ghost init on first spawn ─────────────────
            if ("pilot_car" in libsumo.vehicle.getIDList()
                    and not emv_state["pilot_car_ghost_init"]):
                init_pilot_car_ghost()
                emv_state["pilot_car_ghost_init"] = True

            # ── EMV: Publish congestion + GPS every 0.5s ─────────────────────
            if (_EMV_MSGS_AVAILABLE
                    and amb_id in libsumo.vehicle.getIDList()
                    and (loop_start_time - last_congestion_pub_time)
                        >= EMV_CFG["congestion_pub_interval_s"]):
                publish_congestion(ros_node, pub_congestion, emv_state)
                ambulance_gps = publish_ambulance_gps(ros_node, pub_gps)
                last_congestion_pub_time = loop_start_time

            # ── EMV: Decision pipeline + rerouter every 3s ───────────────────
            if (_EMV_MSGS_AVAILABLE
                    and amb_id in libsumo.vehicle.getIDList()
                    and (loop_start_time - last_decision_eval_time)
                        >= EMV_CFG["decision_eval_interval_s"]):
                run_decision_pipeline(
                    ros_node,
                    (pub_decision, pub_clearance, pub_reroute),
                    emv_state, rerouter_state,
                    loop_start_time)
                run_rerouter(ros_node, pub_waypoint,
                              rerouter_state, ambulance_gps)
                last_decision_eval_time = loop_start_time

            # ── EMV: Rerouter status every 1s ────────────────────────────────
            if (_EMV_MSGS_AVAILABLE
                    and (loop_start_time - last_rerouter_status_time)
                        >= EMV_CFG["rerouter_status_interval_s"]):
                publish_rerouter_status(ros_node, pub_status,
                                        rerouter_state, ambulance_gps)
                last_rerouter_status_time = loop_start_time

            # --- PILOT CAR: publish Gazebo-frame position + velocity as APF goal ---
            if ros_node is not None:
                try:
                    px, py = libsumo.vehicle.getPosition("pilot_car")
                    if not (px > 100000 or px < -100000):
                        gz_x  = px + OFFSET_X
                        gz_y  = py + OFFSET_Y
                        t_now = loop_start_time
                        drone_gz_pos[0] = gz_x
                        drone_gz_pos[1] = gz_y

                        try:
                            sumo_heading_deg = libsumo.vehicle.getAngle("pilot_car")
                        except Exception:
                            sumo_heading_deg = 0.0
                        gz_yaw = math.radians(90 - sumo_heading_deg)

                        vx_ned = 0.0
                        vy_ned = 0.0
                        if last_pilot_pos is not None:
                            last_x, last_y, last_t = last_pilot_pos
                            dt = t_now - last_t
                            if dt > 0.01:
                                vgz_x  = (gz_x - last_x) / dt
                                vgz_y  = (gz_y - last_y) / dt
                                vx_ned = vgz_y
                                vy_ned = vgz_x

                        last_pilot_pos = (gz_x, gz_y, t_now)

                        odom                          = Odometry()
                        odom.header                   = Header()
                        odom.header.frame_id          = "map"
                        odom.child_frame_id           = "pilot_car"
                        odom.pose.pose.position.x     = float(gz_y)
                        odom.pose.pose.position.y     = float(gz_x)
                        odom.pose.pose.position.z     = -30.0 if emv_state["mode"] == "SURVEY" else -5.0
                        odom.pose.pose.orientation.z  = math.sin(gz_yaw / 2.0)
                        odom.pose.pose.orientation.w  = math.cos(gz_yaw / 2.0)
                        odom.twist.twist.linear.x     = float(vx_ned)
                        odom.twist.twist.linear.y     = float(vy_ned)
                        odom.twist.twist.linear.z     = 0.0
                        apf_msg                       = odom
                except Exception:
                    pass

                if apf_msg is not None and (loop_start_time - last_apf_pub_time) >= 0.2:
                    apf_msg.header.stamp = ros_node.get_clock().now().to_msg()
                    pub_apf.publish(apf_msg)
                    last_apf_pub_time = loop_start_time

            # --- SPAWN NEW CARS IN GAZEBO ---
            for car_id, v_class in newly_spawned:
                chosen_sdf = AMBULANCE_SDF_PATH if v_class == "emergency" else random.choice(CAR_SDF_PATHS)
                cmd = [
                    "gz", "service", "-s", f"/world/{WORLD_NAME}/create",
                    "--reqtype", "gz.msgs.EntityFactory", "--reptype", "gz.msgs.Boolean",
                    # Randomize the spawn deep underground to prevent physics collisions!
                    "--req", f"sdf_filename: '{chosen_sdf}', name: '{car_id}', pose: {{position: {{x: {random.randint(-1000, 1000)}.0, y: {random.randint(-1000, 1000)}.0, z: -50.0}}}}"
                ]
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # --- DESPAWN FINISHED CARS FROM GAZEBO ---
            for car_id in cars_to_despawn:
                cmd = [
                    "gz", "service", "-s", f"/world/{WORLD_NAME}/remove",
                    "--reqtype", "gz.msgs.Entity", "--reptype", "gz.msgs.Boolean",
                    "--req", f"name: '{car_id}', type: MODEL"
                ]
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # --- BATCHED POSE UPDATE ---
            batch_req = Pose_V()

            for car_id, spawn_time in active_snapshot.items():
                if time.time() - spawn_time < 1.5:
                    continue
                if car_id not in sub_results:
                    continue

                sumo_x, sumo_y = sub_results[car_id][tc.VAR_POSITION]
                sumo_heading   = sub_results[car_id][tc.VAR_ANGLE]

                gz_x   = sumo_x + OFFSET_X
                gz_y   = sumo_y + OFFSET_Y
                gz_yaw = math.radians(90 - sumo_heading)

                dist = math.hypot(gz_x - drone_gz_pos[0], gz_y - drone_gz_pos[1])

                if dist > 250.0:
                    if car_id not in hidden_cars:
                        p               = batch_req.pose.add()
                        p.name          = car_id
                        p.position.x    = gz_x
                        p.position.y    = gz_y
                        p.position.z    = -50.0
                        p.orientation.z = math.sin(gz_yaw / 2.0)
                        p.orientation.w = math.cos(gz_yaw / 2.0)
                        hidden_cars.add(car_id)
                    continue

                if car_id in hidden_cars:
                    hidden_cars.discard(car_id)

                p               = batch_req.pose.add()
                p.name          = car_id
                p.position.x    = gz_x
                p.position.y    = gz_y
                p.position.z    = OFFSET_Z
                p.orientation.z = math.sin(gz_yaw / 2.0)
                p.orientation.w = math.cos(gz_yaw / 2.0)

            if len(batch_req.pose) > 0:
                gz_node.request(f"/world/{WORLD_NAME}/set_pose_vector", batch_req, Pose_V, Boolean, 200)

            elapsed_time = time.time() - loop_start_time
            if elapsed_time < RT_STEP:
                time.sleep(RT_STEP - elapsed_time)

    finally:
        libsumo.close()
        if ros_node is not None:
            ros_node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
