import math
import time
import libsumo

# --- EMV MSGS (optional — guarded at runtime) ---
_EMV_MSGS_AVAILABLE = False
CongestionSummary   = None
DecisionCommand     = None
TargetWaypoint      = None
RerouteStatus       = None

try:
    from emv_msgs.msg import (CongestionSummary, DecisionCommand,
                               TargetWaypoint, RerouteStatus)
    from emv_msgs.srv import SumoRouteRequest
    _EMV_MSGS_AVAILABLE = True
except ImportError:
    print("[WARN] emv_msgs not found — EMV pipeline disabled")

# --- EMV CONFIGURATION ---
EMV_CFG = {
    "gridlock_threshold":           0.20,
    "alt_severe_mean_threshold":    0.85,
    "alt_severe_max_threshold":     0.95,
    "min_reroute_gain_ratio":       0.05,
    "clearance_timeout_s":          30.0,
    "eval_min_s":                   2.0,
    "eval_max_s":                   10.0,
    "speed_max_mps":                15.0,
    "route_mean_density_scale":     1.8,
    "route_max_density_scale":      1.2,
    "no_shoulder_penalty":          20.0,
    "junction_penalty_scale":       8.0,
    "ambulance_vehicle_id":         "ambulance1",
    "destination_edge":             "501781289#14.290",
    "monitored_edge":               "501781289#4",
    "drone_speed_mps":              5.0,
    "waypoint_acceptance_radius_m": 15.0,
    "waypoint_timeout_s":           60.0,
    "congestion_pub_interval_s":    0.5,
    "decision_eval_interval_s":     3.0,
    "rerouter_status_interval_s":   1.0,
    "reroute_density_threshold":    0.60,
    "reroute_cooldown_s":           15.0,
}


# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres."""
    R  = 6_371_000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = (math.sin(dp / 2) ** 2
          + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return R * 2.0 * math.asin(math.sqrt(a))


def get_edge_density(edge_id):
    """Returns (mean_density, max_density) for a SUMO edge."""
    try:
        n = libsumo.edge.getLaneNumber(edge_id)
    except Exception:
        return 0.0, 0.0
    densities = []
    for i in range(n):
        lane_id = f"{edge_id}_{i}"
        try:
            spd = libsumo.lane.getLastStepMeanSpeed(lane_id)
            ff  = libsumo.lane.getMaxSpeed(lane_id)
            d   = 1.0 - (spd / ff) if ff > 0 else 0.0
            densities.append(max(0.0, min(1.0, d)))
        except Exception:
            pass
    if not densities:
        return 0.0, 0.0
    return (sum(densities) / len(densities), max(densities))


def score_route_cost(edge_ids):
    """A* cost function on a list of SUMO edge IDs."""
    total = 0.0
    for eid in edge_ids:
        if eid.startswith(":"):
            continue
        try:
            length        = libsumo.lane.getLength(eid + "_0")
            mean_d, max_d = get_edge_density(eid)
            mult          = (1.0
                             + EMV_CFG["route_mean_density_scale"] * mean_d
                             + EMV_CFG["route_max_density_scale"]  * max_d)
            total        += length * mult
        except Exception:
            total += 300.0
    return total




def update_sumo_traffic_weights():
    """Paints live traffic onto SUMO's routing engine via adaptTraveltime.
    Must be called before findRoute(..., routingMode=3) so Dijkstra sees
    congested edges as expensive and naturally prefers clear side roads."""
    for edge_id in libsumo.edge.getIDList():
        if edge_id.startswith(":"):
            continue
        try:
            vehicle_count    = libsumo.edge.getLastStepVehicleNumber(edge_id)
            edge_length      = libsumo.edge.getLength(edge_id)
            speed_limit      = libsumo.edge.getSpeed(edge_id)
            if speed_limit <= 0 or edge_length <= 0:
                continue
            max_capacity     = edge_length / 5.0
            density          = min(1.0, vehicle_count / max_capacity)
            base_travel_time = edge_length / speed_limit
            live_travel_time = base_travel_time * (1.0 + density * 10.0)
            libsumo.edge.adaptTraveltime(edge_id, live_travel_time)
        except Exception:
            continue


def set_initial_route():
    """Called once when ambulance1 enters simulation (t=100).
    Applies the convoy route locked by init_pilot_car_ghost() at t=95."""
    if not _convoy_route:
        print("[EMV] No convoy route locked yet — ambulance route unchanged")
        return
    amb_id = EMV_CFG["ambulance_vehicle_id"]
    try:
        libsumo.vehicle.setRoute(amb_id, _convoy_route)
        print(f"[EMV] Ambulance locked to convoy route: {len(_convoy_route)} edges")
    except Exception as exc:
        print(f"[EMV] Ambulance setRoute failed: {exc}")


# ── ROS2 PUBLISHING FUNCTIONS ─────────────────────────────────────────────────

def publish_congestion(ros_node, pub_congestion, emv_state):
    """Reads lane densities from pilot car's current edge; publishes CongestionSummary."""
    if pub_congestion is None:
        return

    if PILOT_CAR_ID not in libsumo.vehicle.getIDList():
        return
    try:
        edge_id = libsumo.vehicle.getRoadID(PILOT_CAR_ID)
        if not edge_id or edge_id.startswith(":"):
            return  # inside junction — skip this tick
    except Exception:
        return

    try:
        actual_count = libsumo.edge.getLaneNumber(edge_id)
    except Exception:
        actual_count = 3

    densities = []
    lane_ids  = []
    for i in range(actual_count):
        lane_id = f"{edge_id}_{i}"
        try:
            spd = libsumo.lane.getLastStepMeanSpeed(lane_id)
            ff  = libsumo.lane.getMaxSpeed(lane_id)
            d   = 1.0 - (spd / ff) if ff > 0 else 0.0
            d   = max(0.0, min(1.0, d))
        except Exception:
            d = 0.0
        densities.append(d)
        lane_ids.append(f"{edge_id}_lane_{i}")

    mean_d = sum(densities) / len(densities) if densities else 0.0
    max_d  = max(densities)                  if densities else 0.0

    emv_state["last_densities"] = densities

    msg = CongestionSummary()
    msg.header.stamp      = ros_node.get_clock().now().to_msg()
    msg.header.frame_id   = "map"
    msg.lane_ids          = lane_ids
    msg.lane_density      = [float(d) for d in densities]
    msg.mean_density      = float(mean_d)
    msg.max_density       = float(max_d)
    msg.odom_velocity_mps = [EMV_CFG["drone_speed_mps"], 0.0, 0.0]
    msg.lane_count        = actual_count
    msg.junction_type     = "straight"
    pub_congestion.publish(msg)


def publish_ambulance_gps(ros_node, pub_gps):
    """Reads ambulance SUMO position; publishes NavSatFix. Returns (lat, lon)."""
    if pub_gps is None:
        return (0.0, 0.0)
    amb_id = EMV_CFG["ambulance_vehicle_id"]
    try:
        x, y     = libsumo.vehicle.getPosition(amb_id)
        lon, lat = libsumo.simulation.convertGeo(x, y)

        from sensor_msgs.msg import NavSatFix
        gps = NavSatFix()
        gps.header.stamp    = ros_node.get_clock().now().to_msg()
        gps.header.frame_id = "map"
        gps.latitude        = float(lat)
        gps.longitude       = float(lon)
        gps.status.status   = 0
        pub_gps.publish(gps)
        return (float(lat), float(lon))
    except Exception:
        return (0.0, 0.0)


def run_decision_pipeline(ros_node, publishers, emv_state, _rerouter_state, now):
    """
    Three-state EMV pipeline:
      mean_d < 0.20              → SURVEY
      0.20 <= mean_d < 0.80      → CLEAR (bluelight, moderate congestion)
      mean_d >= 0.80             → REROUTE attempt, then CLEAR fallback
    """
    pub_decision, pub_clearance, _ = publishers

    densities       = emv_state["last_densities"]
    if not densities:
        return

    clear_threshold  = EMV_CFG["gridlock_threshold"]
    reroute_threshold = EMV_CFG["reroute_density_threshold"]
    mean_d           = sum(densities) / len(densities)
    rho_str          = " ".join(f"{d:.2f}" for d in densities)

    if mean_d < clear_threshold:
        in_clear_lockout = (
            emv_state["mode"] == "CLEAR"
            and (now - emv_state["clearance_start_time"])
                < EMV_CFG["clearance_timeout_s"]
        )
        if in_clear_lockout:
            remaining = EMV_CFG["clearance_timeout_s"] - (now - emv_state["clearance_start_time"])
            _activate_clear(ros_node, pub_decision, pub_clearance,
                            emv_state, mean_d, rho_str, clear_threshold, now)
            print(f"[CLEAR] holding lockout {remaining:.0f}s remaining")
            return
        if emv_state["mode"] != "SURVEY":
            emv_state["mode"] = "SURVEY"
            apply_pilot_car_mode("SURVEY")
        print(f"[SURVEY] lanes={len(densities)} rho=[{rho_str}] mean={mean_d:.2f}")
        _pub_decision_cmd(ros_node, pub_decision,
                          mode="SURVEY",
                          action="Monitor — path clear",
                          reason=f"mean={mean_d:.2f}<{clear_threshold}")
    elif mean_d >= reroute_threshold:
        if _attempt_reroute(ros_node, pub_decision, emv_state, mean_d, rho_str, now):
            return
        # Reroute failed — fall through to CLEAR
        _activate_clear(ros_node, pub_decision, pub_clearance,
                        emv_state, mean_d, rho_str, reroute_threshold, now)
    else:
        # clear_threshold <= mean_d < reroute_threshold: bluelight only, no reroute
        _activate_clear(ros_node, pub_decision, pub_clearance,
                        emv_state, mean_d, rho_str, clear_threshold, now)


def _activate_clear(ros_node, pub_decision, pub_clearance,
                    emv_state, mean_d, rho_str, threshold, now):
    if emv_state["mode"] != "CLEAR":
        emv_state["clearance_start_time"] = now
        emv_state["mode"] = "CLEAR"
        apply_pilot_car_mode("CLEAR")
    elapsed = now - emv_state["clearance_start_time"]
    print(f"[CLEAR] elapsed={elapsed:.0f}s rho=[{rho_str}] mean={mean_d:.2f}")
    _pub_decision_cmd(ros_node, pub_decision,
                      mode="CLEAR",
                      action="Activate drone clearance",
                      clearance_lane=0, buzzer=True,
                      elapsed=elapsed,
                      reason=f"mean={mean_d:.2f}>={threshold}")
    _pub_decision_cmd(ros_node, pub_clearance,
                      mode="CLEAR",
                      action="Drone clearance trigger",
                      clearance_lane=0, buzzer=True,
                      elapsed=elapsed,
                      reason=f"mean={mean_d:.2f}>={threshold}")


def _attempt_reroute(ros_node, pub_decision, emv_state, mean_d, rho_str, now):
    """
    Tries to compute an alternate route that avoids the pilot car's current congested edge.
    Applies the new route to both vehicles and teleports the pilot car one edge ahead of
    the ambulance to resume scouting. Returns True if a reroute was applied, False otherwise.
    """
    global _convoy_route

    last_reroute = emv_state.setdefault("last_reroute_time", 0.0)
    if now - last_reroute < EMV_CFG["reroute_cooldown_s"]:
        return False

    if PILOT_CAR_ID not in libsumo.vehicle.getIDList():
        return False

    amb_id = EMV_CFG["ambulance_vehicle_id"]
    if amb_id not in libsumo.vehicle.getIDList():
        return False

    try:
        congested_edge = libsumo.vehicle.getRoadID(PILOT_CAR_ID)
        if not congested_edge or congested_edge.startswith(":"):
            return False
    except Exception:
        return False

    try:
        amb_edge = libsumo.vehicle.getRoadID(amb_id)
        if not amb_edge or amb_edge.startswith(":"):
            raw      = libsumo.vehicle.getRoute(amb_id)
            amb_edge = raw[0] if raw else ""
        if not amb_edge:
            return False
    except Exception:
        return False

    dest       = EMV_CFG["destination_edge"]
    new_edges  = None
    orig_time  = 0.0

    # Paint the whole map with live traffic before asking for a detour.
    # congested_edge gets its density-based weight first, then we stack a
    # hard 9999s penalty on top to guarantee Dijkstra avoids it.
    update_sumo_traffic_weights()
    try:
        orig_time = libsumo.edge.getTraveltime(congested_edge)
    except Exception:
        orig_time = 30.0

    try:
        libsumo.edge.adaptTraveltime(congested_edge, 9999.0)
        candidates = []
        for mode in [3, 0]:  # 3=live weights, 0=free-flow fallback
            try:
                result = libsumo.simulation.findRoute(amb_edge, dest, routingMode=mode)
                edges  = [e for e in result.edges if not e.startswith(":")]
                if (edges
                        and congested_edge not in edges
                        and not any(c["edges"] == edges for c in candidates)):
                    candidates.append({"edges": edges, "cost": score_route_cost(edges)})
            except Exception:
                continue
        if candidates:
            best      = min(candidates, key=lambda c: c["cost"])
            new_edges = best["edges"]
    finally:
        try:
            libsumo.edge.adaptTraveltime(congested_edge, orig_time)
        except Exception:
            pass

    if not new_edges:
        print(f"[REROUTE] No detour found from {amb_edge} avoiding {congested_edge}")
        return False

    try:
        libsumo.vehicle.setRoute(amb_id, new_edges)
    except Exception as exc:
        print(f"[REROUTE] Ambulance setRoute failed: {exc}")
        return False

    # Build 2-point GPS waypoints so _teleport_pilot_car can place the pilot car
    # one edge ahead of the ambulance on the new route.
    waypoints = []
    try:
        amb_x, amb_y   = libsumo.vehicle.getPosition(amb_id)
        amb_lon, amb_lat = libsumo.simulation.convertGeo(amb_x, amb_y)
        waypoints.append((float(amb_lat), float(amb_lon)))
        if len(new_edges) > 1:
            shape = libsumo.lane.getShape(f"{new_edges[1]}_0")
            if shape:
                wp_x, wp_y       = shape[0]
                wp_lon, wp_lat   = libsumo.simulation.convertGeo(wp_x, wp_y)
                waypoints.append((float(wp_lat), float(wp_lon)))
    except Exception:
        pass

    if waypoints:
        _teleport_pilot_car(new_edges, waypoints, ahead_idx=min(1, len(waypoints) - 1))

    _convoy_route = new_edges
    emv_state["mode"]             = "REROUTING"
    emv_state["last_reroute_time"] = now
    apply_pilot_car_mode("SURVEY")

    print(f"[REROUTE] New route: {len(new_edges)} edges, avoided={congested_edge},"
          f" mean={mean_d:.2f} rho=[{rho_str}]")

    _pub_decision_cmd(ros_node, pub_decision,
                      mode="REROUTING",
                      action="Obstacle detected ahead, calculating detour",
                      reason=f"avoided={congested_edge} mean={mean_d:.2f}")
    return True


def run_rerouter(ros_node, pub_waypoint, rerouter_state, ambulance_gps):
    """Checks GPS progress, advances waypoints, publishes next TargetWaypoint."""
    st = rerouter_state
    if st["state"] != "NAVIGATING":
        return

    idx  = st["current_waypoint_index"]
    wpts = st["waypoints"]
    n    = len(wpts)
    if idx >= n:
        st["state"] = "COMPLETED"
        print(f"[REROUTER] COMPLETED {st['active_route_name']}")
        return

    t_lat, t_lon = wpts[idx]
    dist = haversine(ambulance_gps[0], ambulance_gps[1], t_lat, t_lon)

    radius = EMV_CFG["waypoint_acceptance_radius_m"]
    if dist <= radius:
        st["current_waypoint_index"]     += 1
        st["last_waypoint_advance_time"]  = time.monotonic()
        idx = st["current_waypoint_index"]
        print(f"[REROUTER] Waypoint {idx}/{n} reached")

        if idx >= n:
            st["state"] = "COMPLETED"
            print(f"[REROUTER] COMPLETED {st['active_route_name']}")
            return

        t_lat, t_lon = wpts[idx]
        _teleport_pilot_car([], wpts, ahead_idx=idx + 1)

    if pub_waypoint is not None:
        wp = TargetWaypoint()
        wp.header.stamp    = ros_node.get_clock().now().to_msg()
        wp.header.frame_id = "map"
        wp.latitude        = float(t_lat)
        wp.longitude       = float(t_lon)
        wp.waypoint_index  = idx
        wp.total_waypoints = n
        wp.route_name      = st["active_route_name"]
        pub_waypoint.publish(wp)

    timeout = EMV_CFG["waypoint_timeout_s"]
    elapsed = time.monotonic() - st["last_waypoint_advance_time"]
    if elapsed >= timeout:
        st["state"]               = "FAILED"
        st["last_failure_reason"] = (f"Waypoint {idx} timeout {elapsed:.0f}s"
                                      f" dist={dist:.1f}m")
        print(f"[REROUTER] FAILED: {st['last_failure_reason']}")


def publish_rerouter_status(ros_node, pub_status, rerouter_state, ambulance_gps):
    """Publishes RerouteStatus heartbeat at 1 Hz."""
    if pub_status is None:
        return
    st   = rerouter_state
    idx  = st["current_waypoint_index"]
    n    = len(st["waypoints"])
    dist = 0.0
    if st["state"] == "NAVIGATING" and idx < n:
        t_lat, t_lon = st["waypoints"][idx]
        dist = haversine(ambulance_gps[0], ambulance_gps[1], t_lat, t_lon)

    msg = RerouteStatus()
    msg.header.stamp           = ros_node.get_clock().now().to_msg()
    msg.state                  = st["state"]
    msg.active_route_name      = st["active_route_name"]
    msg.total_waypoints        = n
    msg.current_waypoint_index = idx
    msg.distance_to_next_m     = float(dist)
    msg.current_lat            = float(ambulance_gps[0])
    msg.current_lon            = float(ambulance_gps[1])
    msg.reason                 = st["last_failure_reason"]
    elapsed = (time.monotonic() - st["reroute_start_time"]
               if st["reroute_start_time"] > 0 else 0.0)
    msg.elapsed_s = elapsed
    pub_status.publish(msg)


# ── INTERNAL HELPERS ──────────────────────────────────────────────────────────

PILOT_CAR_ID   = "pilot_car"
_convoy_route  = []   # locked at pilot_car spawn; applied to ambulance at its spawn


def init_pilot_car_ghost():
    """Called once on pilot_car spawn (t=95). Locks convoy route while vehicle is on E14,
    then switches to ghost_car type so traffic doesn't yield to it."""
    global _convoy_route
    dest = EMV_CFG["destination_edge"]
    try:
        current = libsumo.vehicle.getRoadID(PILOT_CAR_ID)
        if not current or current.startswith(":"):
            raw     = libsumo.vehicle.getRoute(PILOT_CAR_ID)
            current = raw[0] if raw else ""
        if current:
            update_sumo_traffic_weights()
            candidates = []
            for mode in [3, 0]:  # 3=live weights, 0=free-flow fallback
                try:
                    result = libsumo.simulation.findRoute(
                        current, dest, routingMode=mode)
                    edges = [e for e in result.edges if not e.startswith(":")]
                    if edges and not any(c["edges"] == edges for c in candidates):
                        candidates.append({"edges": edges,
                                           "cost":  score_route_cost(edges)})
                except Exception:
                    continue
            if candidates:
                best = min(candidates, key=lambda c: c["cost"])
                _convoy_route = best["edges"]
                libsumo.vehicle.setRoute(PILOT_CAR_ID, _convoy_route)
                print(f"[PILOT] Convoy route locked at spawn:"
                      f" {len(_convoy_route)} edges cost={best['cost']:.1f}")
        libsumo.vehicle.setType(PILOT_CAR_ID, "ghost_car")
        print("[PILOT] Ghost mode init — switched to ghost_car (no bluelight, minGap=0)")
    except Exception as exc:
        print(f"[PILOT] init_pilot_car_ghost failed: {exc}")


def apply_pilot_car_mode(mode):
    """Switches pilot_car vType based on EMV mode.
    CLEAR → emergency type (bluelight on, cars yield); anything else → ghost_car (pass-through).
    """
    if PILOT_CAR_ID not in libsumo.vehicle.getIDList():
        return
    try:
        if mode == "CLEAR":
            libsumo.vehicle.setType(PILOT_CAR_ID, "emergency")
            print("[PILOT] Clearance ACTIVE — switched to emergency (bluelight on)")
        else:
            libsumo.vehicle.setType(PILOT_CAR_ID, "ghost_car")
            print(f"[PILOT] Ghost ({mode}) — switched to ghost_car (no bluelight)")
    except Exception as exc:
        print(f"[PILOT] apply_pilot_car_mode({mode}) failed: {exc}")


def _teleport_pilot_car(edges, waypoints, ahead_idx=1):
    """
    Teleports pilot_car to waypoints[ahead_idx] on the new route so the
    drone is immediately guided one step ahead of the ambulance.
    """
    if not waypoints:
        return
    idx = min(ahead_idx, len(waypoints) - 1)
    t_lat, t_lon = waypoints[idx]
    try:
        x, y = libsumo.simulation.convertGeo(t_lon, t_lat, fromGeo=True)
        edge_id, _, _ = libsumo.simulation.convertRoad(x, y, isGeo=False)
        snap_edge = edge_id if (edge_id and not edge_id.startswith(":")) else ""
        if edges:
            libsumo.vehicle.setRoute("pilot_car", edges)
        libsumo.vehicle.moveToXY(
            "pilot_car", snap_edge, 0, x, y, -1001, 2,
        )
        print(f"[PILOT] moveToXY → wp[{idx}] ({t_lat:.5f}, {t_lon:.5f})"
              f" edge={snap_edge}")
    except Exception as exc:
        print(f"[PILOT] moveToXY failed: {exc}")



def _pub_decision_cmd(ros_node, publisher, mode, action,
                       clearance_lane=0, buzzer=False,
                       elapsed=0.0, reason=""):
    """Builds and publishes a DecisionCommand."""
    if publisher is None:
        return
    cmd = DecisionCommand()
    cmd.header.stamp         = ros_node.get_clock().now().to_msg()
    cmd.mode                 = mode
    cmd.action_description   = action
    cmd.clearance_lane_index = clearance_lane
    cmd.buzzer_on            = buzzer
    cmd.clearance_elapsed_s  = float(elapsed)
    cmd.reason               = reason
    cmd.route_waypoint_lats  = []
    cmd.route_waypoint_lons  = []
    cmd.route_cost           = 0.0
    cmd.selected_route_name  = ""
    publisher.publish(cmd)
