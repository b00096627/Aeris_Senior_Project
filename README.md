# AERIS — Autonomous Emergency Response Intelligence System

An autonomous drone-based system designed to help emergency medical 
vehicles navigate urban traffic faster. The drone flies ahead of the 
ambulance, analyzes live traffic from the air, and either reroutes 
the vehicle or clears the lane — all without human intervention.

---

## System Architecture

**Drone (Jetson Orin Nano)**
- RT-DETR real-time vehicle detection
- Custom APF-based autonomous flight planner
- Congestion summary generation and transmission

**EMV Onboard Computer**
- Criticality-aware routing engine
- Lane clearance decision logic

**Simulation Stack (PC)**
- SUMO — urban traffic simulation
- Gazebo — 3D physics environment
- PX4 SITL — drone flight controller
- Custom bridges connecting all three in real time

---

## Repository Structure

| Folder | Description |
|---|---|
| `apf_planner/` | Custom APF-based local planner (Jetson) |
| `bridges/` | SUMO-Gazebo-PX4 real-time bridges |
| `crossroad/` | SUMO network files and road calibration |
| `3d_assets/` | Vehicle and ambulance models for Gazebo |
| `run_sim_2.sh` | Main simulation launcher (PC) |
| `kill_sim.sh` | Simulation teardown script |

> ⚠️ Full deployment guide and Jetson-side CV pipeline coming soon.

---

## Team
Ahmed Gheyas · Dewansh Agrawal · Sayyed Sinan Hadhadh · Mayur Jhamnani  
Supervised by Prof. Dr. Mohamed Hassan  
American University of Sharjah — Senior Design Project, May 2026
