#!/usr/bin/env python3
"""
Artificial Potential Field planner for PX4 on ROS2 Humble.

Behavior:
  1. On launch, drone arms and takes off to takeoff_alt
  2. After takeoff, drone hovers in place
  3. Waits for waypoints published to /apf/goal
  4. Each waypoint causes the drone to fly to it using APF avoidance
  5. After reaching a waypoint, drone hovers there until next waypoint

Send waypoints via:
  ros2 topic pub --once /apf/goal geometry_msgs/msg/PointStamped \
    '{point: {x: 10.0, y: 0.0, z: -3.0}}'
"""

import threading
import time

import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Float32


class APFPlanner(Node):

    def __init__(self):
        super().__init__('apf_planner')

        # ------------------------------------------------------------------ #
        # Parameters                                                         #
        # ------------------------------------------------------------------ #
        self.declare_parameter('k_att', 0.5)
        self.declare_parameter('k_rep', 5.0)
        self.declare_parameter('d_max', 4.0)
        self.declare_parameter('v_max', 2.0)
        self.declare_parameter('vz_max', 1.0)
        self.declare_parameter('goal_tolerance', 3.0)
        self.declare_parameter('takeoff_alt', -3.0)
        self.declare_parameter('cloud_topic', '/camera/depth_front/points')
        self.declare_parameter('ground_filter_z', 2.0)
        self.declare_parameter('ceil_filter_z', -3.0)
        self.declare_parameter('danger_radius', 1.5)
        self.declare_parameter('danger_speed', 2.0)
        self.declare_parameter('verbose_period', 1.0)
        self.declare_parameter('up_bias_strength', 2.0)
        self.declare_parameter('up_preference', 3.0)
        self.declare_parameter('feedforward_gain', 0.8)
        self.declare_parameter('cloud_stale_threshold', 2.0)
        self.declare_parameter('cloud_blind_threshold', 5.0)
        self.declare_parameter('blind_speed_factor', 1.0)
        self.declare_parameter('recovery_duration_s', 2.0)
        self.declare_parameter('recovery_softness', 0.3)
        self.declare_parameter('command_smoothing_alpha', 0.20)
        self.declare_parameter('repulsion_smoothing_alpha', 0.25)
        self.declare_parameter('max_accel', 8.0)
        self.declare_parameter('danger_blend_power', 1.5)
        self.declare_parameter('cloud_process_hz', 3.0)
        self.declare_parameter('max_raw_cloud_points', 6000)
        self.declare_parameter('max_cloud_points', 1000)
        self.declare_parameter('cloud_sample_stride', 1)
        self.declare_parameter('voxel_size', 0.25)
        self.declare_parameter('use_gpu', True)
        self.declare_parameter('local_position_accept_hz', 50.0)
        self.declare_parameter('status_accept_hz', 5.0)
        self.declare_parameter('tracking_speed_match_gain', 1.0)
        self.declare_parameter('tracking_along_gain', 0.8)
        self.declare_parameter('tracking_lateral_gain', 0.8)
        self.declare_parameter('tracking_position_gain', 0.9)
        self.declare_parameter('tracking_max_correction_speed', 5.0)
        self.declare_parameter('tracking_speed_margin', 5.0)
        self.declare_parameter('tracking_slow_speed_threshold', 0.4)
        self.declare_parameter('tracking_hold_radius', 2.0)
        self.declare_parameter('tracking_deadband_radius', 0.75)
        self.declare_parameter('tracking_brake_gain', 0.2)
        self.declare_parameter('tracking_slow_max_speed', 4.0)
        self.declare_parameter('tracking_enabled', False)
        self.declare_parameter('auto_tracking_enabled', True)
        self.declare_parameter('tracking_start_radius', 10.0)
        self.declare_parameter('goal_odom_accept_hz', 5.0)
        self.declare_parameter('speed_sync_radius', 5.0)

        self.k_att = self.get_parameter('k_att').value
        self.k_rep = self.get_parameter('k_rep').value
        self.d_max = self.get_parameter('d_max').value
        self.v_max = self.get_parameter('v_max').value
        self.vz_max = self.get_parameter('vz_max').value
        self.tol = self.get_parameter('goal_tolerance').value
        self.takeoff_alt = self.get_parameter('takeoff_alt').value
        self.ground_filter_z = self.get_parameter('ground_filter_z').value
        self.ceil_filter_z = self.get_parameter('ceil_filter_z').value
        self.danger_radius = self.get_parameter('danger_radius').value
        self.danger_speed = self.get_parameter('danger_speed').value
        self.verbose_period = self.get_parameter('verbose_period').value
        self.up_bias_strength = self.get_parameter('up_bias_strength').value
        self.up_preference = self.get_parameter('up_preference').value
        self.feedforward_gain = self.get_parameter('feedforward_gain').value
        self.cloud_stale_threshold = self.get_parameter('cloud_stale_threshold').value
        self.cloud_blind_threshold = self.get_parameter('cloud_blind_threshold').value
        self.blind_speed_factor = self.get_parameter('blind_speed_factor').value
        self.recovery_duration_s = self.get_parameter('recovery_duration_s').value
        self.recovery_softness = self.get_parameter('recovery_softness').value
        self.command_smoothing_alpha = self.get_parameter('command_smoothing_alpha').value
        self.repulsion_smoothing_alpha = self.get_parameter('repulsion_smoothing_alpha').value
        self.max_accel = self.get_parameter('max_accel').value
        self.danger_blend_power = self.get_parameter('danger_blend_power').value
        self.cloud_process_hz = self.get_parameter('cloud_process_hz').value
        self.max_raw_cloud_points = int(self.get_parameter('max_raw_cloud_points').value)
        self.max_cloud_points = int(self.get_parameter('max_cloud_points').value)
        self.cloud_sample_stride = int(self.get_parameter('cloud_sample_stride').value)
        self.voxel_size = self.get_parameter('voxel_size').value
        self.use_gpu = bool(self.get_parameter('use_gpu').value)
        self.local_position_accept_hz = self.get_parameter('local_position_accept_hz').value
        self.status_accept_hz = self.get_parameter('status_accept_hz').value
        self.tracking_speed_match_gain = self.get_parameter('tracking_speed_match_gain').value
        self.tracking_along_gain = self.get_parameter('tracking_along_gain').value
        self.tracking_lateral_gain = self.get_parameter('tracking_lateral_gain').value
        self.tracking_position_gain = self.get_parameter('tracking_position_gain').value
        self.tracking_max_correction_speed = self.get_parameter('tracking_max_correction_speed').value
        self.tracking_speed_margin = self.get_parameter('tracking_speed_margin').value
        self.tracking_slow_speed_threshold = self.get_parameter('tracking_slow_speed_threshold').value
        self.tracking_hold_radius = self.get_parameter('tracking_hold_radius').value
        self.tracking_deadband_radius = self.get_parameter('tracking_deadband_radius').value
        self.tracking_brake_gain = self.get_parameter('tracking_brake_gain').value
        self.tracking_slow_max_speed = self.get_parameter('tracking_slow_max_speed').value
        self.tracking_enabled = bool(self.get_parameter('tracking_enabled').value)
        self.auto_tracking_enabled = bool(self.get_parameter('auto_tracking_enabled').value)
        self.tracking_start_radius = self.get_parameter('tracking_start_radius').value
        self.goal_odom_accept_hz = self.get_parameter('goal_odom_accept_hz').value
        self.speed_sync_radius = self.get_parameter('speed_sync_radius').value
        cloud_topic = self.get_parameter('cloud_topic').value

        self.command_smoothing_alpha = float(np.clip(self.command_smoothing_alpha, 0.01, 1.0))
        self.repulsion_smoothing_alpha = float(np.clip(self.repulsion_smoothing_alpha, 0.01, 1.0))
        self.max_accel = max(float(self.max_accel), 0.1)
        self.cloud_process_hz = float(np.clip(self.cloud_process_hz, 0.5, 30.0))
        self.max_raw_cloud_points = max(self.max_raw_cloud_points, 500)
        self.max_cloud_points = max(self.max_cloud_points, 100)
        self.cloud_sample_stride = max(self.cloud_sample_stride, 1)
        self.local_position_accept_hz = float(np.clip(self.local_position_accept_hz, 1.0, 100.0))
        self.status_accept_hz = float(np.clip(self.status_accept_hz, 1.0, 100.0))
        self.tracking_speed_match_gain = max(float(self.tracking_speed_match_gain), 0.0)
        self.tracking_max_correction_speed = max(float(self.tracking_max_correction_speed), 0.1)
        self.tracking_speed_margin = max(float(self.tracking_speed_margin), 0.0)
        self.tracking_slow_speed_threshold = max(float(self.tracking_slow_speed_threshold), 0.0)
        self.tracking_hold_radius = max(float(self.tracking_hold_radius), 0.1)
        self.tracking_deadband_radius = max(float(self.tracking_deadband_radius), 0.0)
        self.tracking_brake_gain = max(float(self.tracking_brake_gain), 0.0)
        self.tracking_slow_max_speed = max(float(self.tracking_slow_max_speed), 0.1)
        self.tracking_start_radius = max(float(self.tracking_start_radius), 0.1)
        self.goal_odom_accept_hz = float(np.clip(self.goal_odom_accept_hz, 0.5, 5.0))
        self.speed_sync_radius = max(float(self.speed_sync_radius), 0.1)

        self.cp = None
        self.gpu_enabled = False
        if self.use_gpu:
            try:
                import cupy as cp
                _ = cp.cuda.runtime.getDeviceCount()
                self.cp = cp
                self.gpu_enabled = True
            except Exception as exc:
                self.get_logger().warn(
                    f'GPU requested but CuPy/CUDA is not available ({exc}); using CPU cloud filtering')

        # ------------------------------------------------------------------ #
        # State                                                              #
        # ------------------------------------------------------------------ #
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.yaw = 0.0
        self.obstacles_body = np.empty((0, 3), dtype=np.float32)
        self.armed = False
        self.nav_state = 0
        self.arming_state = 0
        self.offboard_counter = 0
        self.stuck_counter = 0
        self.v_filt = np.zeros(3)
        self.rep_filt_body = np.zeros(3)
        self.last_control_time = None
        self.has_taken_off = False
        self.hover_position = np.zeros(3)
        self.goal_received = False
        self.goal = np.zeros(3)
        self.goal_velocity = np.zeros(3)
        self.static_goal = None
        self.waiting_for_auto_tracking = False
        self.last_goal_odom_time = None
        self.last_goal_odom_accept_time = None
        self.last_position_accept_time = None
        self.last_status_accept_time = None
        self.last_cloud_rx_time = None
        self.last_cloud_time = None
        self.latest_cloud_msg = None
        self.latest_cloud_id = 0
        self.processed_cloud_id = 0
        self.cloud_lock = threading.Lock()
        self.cloud_worker_stop = threading.Event()
        self.cloud_worker_wake = threading.Event()
        self.cloud_stale_warned = False
        self.was_blind = False
        self.blind_recovery_until = None

        # Diagnostic counters. cloud_rx counts incoming PointCloud2 messages;
        # cloud_proc counts clouds actually converted/filtered by APF.
        self.cloud_rx_count = 0
        self.cloud_proc_count = 0
        self.cloud_used_count = 0
        self.cloud_overwrite_count = 0
        self.cloud_last_process_ms = 0.0
        self.position_rx_count = 0
        self.status_rx_count = 0
        self.position_count = 0
        self.status_count = 0

        # ------------------------------------------------------------------ #
        # QoS                                                                #
        # ------------------------------------------------------------------ #
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        goal_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        telemetry_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Subscribers
        self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1',
            self.local_pos_cb, px4_qos)
        self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v2',
            self.status_cb, px4_qos)
        self.create_subscription(
            PointCloud2, cloud_topic,
            self.cloud_cb, sensor_qos)
        self.create_subscription(
            PointStamped, '/apf/goal',
            self.goal_cb, goal_qos)
        self.create_subscription(
            Odometry, '/apf/goal_odom',
            self.goal_odom_cb, goal_qos)
        self.create_subscription(
            Bool, '/apf/tracking_enabled',
            self.tracking_enabled_cb, goal_qos)

        # Publishers
        self.pub_offboard = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', px4_qos)
        self.pub_setpoint = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', px4_qos)
        self.pub_command = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', px4_qos)
        self.pub_obs_dist = self.create_publisher(
            Float32, '/apf/min_obstacle_distance', telemetry_qos)

        # Timers
        self.create_timer(0.05, self.control_loop)
        self.create_timer(self.verbose_period, self.status_print)
        self.cloud_worker_thread = threading.Thread(
            target=self._cloud_worker_loop,
            name='apf_cloud_worker',
            daemon=True,
        )
        self.cloud_worker_thread.start()

        # Banner
        self.get_logger().info('=' * 60)
        self.get_logger().info('APF planner ready.')
        self.get_logger().info(f'  Cloud topic: {cloud_topic}')
        self.get_logger().info('  /apf/goal QoS: RELIABLE, VOLATILE, depth=1')
        self.get_logger().info('  PointCloud2 QoS: BEST_EFFORT, VOLATILE, depth=1')
        self.get_logger().info(f'  k_att={self.k_att}  k_rep={self.k_rep}  d_max={self.d_max}')
        self.get_logger().info(f'  v_max={self.v_max}  vz_max={self.vz_max}')
        self.get_logger().info(f'  goal_tolerance={self.tol} fixed')
        self.get_logger().info(
            f'  danger_radius={self.danger_radius}  danger_speed={self.danger_speed}')
        self.get_logger().info(
            f'  smoothing_alpha={self.command_smoothing_alpha}  '
            f'repulsion_alpha={self.repulsion_smoothing_alpha}  max_accel={self.max_accel}')
        self.get_logger().info(
            f'  cloud_stale_threshold={self.cloud_stale_threshold}  '
            f'cloud_blind_threshold={self.cloud_blind_threshold}  '
            f'blind_speed_factor={self.blind_speed_factor}')
        self.get_logger().info(
            f'  cloud_process_hz={self.cloud_process_hz}  '
            f'max_raw_cloud_points={self.max_raw_cloud_points}  '
            f'max_cloud_points={self.max_cloud_points}  voxel_size={self.voxel_size}')
        self.get_logger().info(
            f'  use_gpu={self.use_gpu}  gpu_enabled={self.gpu_enabled}')
        self.get_logger().info(
            f'  local_position_accept_hz={self.local_position_accept_hz}  '
            f'status_accept_hz={self.status_accept_hz}')
        self.get_logger().info(
            f'  tracking_speed_match_gain={self.tracking_speed_match_gain}  '
            f'tracking_max_correction_speed={self.tracking_max_correction_speed}  '
            f'tracking_speed_margin={self.tracking_speed_margin}')
        self.get_logger().info(
            f'  tracking_slow_speed_threshold={self.tracking_slow_speed_threshold}  '
            f'tracking_hold_radius={self.tracking_hold_radius}  '
            f'tracking_deadband_radius={self.tracking_deadband_radius}')
        self.get_logger().info(
            f'  tracking_enabled={self.tracking_enabled}  '
            f'auto_tracking_enabled={self.auto_tracking_enabled}  '
            f'tracking_start_radius={self.tracking_start_radius}  '
            f'goal_odom_accept_hz={self.goal_odom_accept_hz}  '
            f'speed_sync_radius={self.speed_sync_radius}')
        self.get_logger().info(
            '  Publish the spawn waypoint first. /apf/goal_odom will auto-take-over '
            'after the drone is near that point.')
        self.get_logger().info('=' * 60)

    # ======================================================================= #
    # Callbacks                                                               #
    # ======================================================================= #

    def local_pos_cb(self, msg: VehicleLocalPosition):
        self.position_rx_count += 1
        now = time.monotonic()
        if self.last_position_accept_time is not None:
            min_period = 1.0 / self.local_position_accept_hz
            if now - self.last_position_accept_time < min_period:
                return
        self.last_position_accept_time = now

        self.position = np.array([msg.x, msg.y, msg.z], dtype=float)
        self.velocity = np.array([msg.vx, msg.vy, msg.vz], dtype=float)
        self.yaw = float(msg.heading)
        self.position_count += 1

    def status_cb(self, msg: VehicleStatus):
        self.status_rx_count += 1
        now = time.monotonic()
        if self.last_status_accept_time is not None:
            min_period = 1.0 / self.status_accept_hz
            if now - self.last_status_accept_time < min_period:
                return
        self.last_status_accept_time = now

        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state
        self.armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        self.status_count += 1

    def goal_cb(self, msg: PointStamped):
        self.goal = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=float)
        self.static_goal = self.goal.copy()
        self.goal_velocity = np.zeros(3)
        self.last_goal_odom_time = None
        self.tracking_enabled = False
        self.waiting_for_auto_tracking = self.auto_tracking_enabled
        self.goal_received = True
        self.get_logger().info(
            f'>>> NEW GOAL received: NED '
            f'[{self.goal[0]:.2f}, {self.goal[1]:.2f}, {self.goal[2]:.2f}] '
            f'(fly-to-spawn first, auto_tracking={self.waiting_for_auto_tracking})')

    def goal_odom_cb(self, msg: Odometry):
        """Goal with velocity, used when tracking a moving target."""
        now_wall = time.monotonic()
        if self.last_goal_odom_accept_time is not None:
            min_period = 1.0 / self.goal_odom_accept_hz
            if now_wall - self.last_goal_odom_accept_time < min_period:
                return
        self.last_goal_odom_accept_time = now_wall

        odom_goal = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ], dtype=float)
        odom_velocity = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        ], dtype=float)

        if not self.tracking_enabled:
            if not (self.auto_tracking_enabled and self.waiting_for_auto_tracking):
                return

            if self.static_goal is None:
                return

            dist_to_static_goal = float(np.linalg.norm(self.static_goal - self.position))
            if dist_to_static_goal > self.tracking_start_radius:
                return

            self.tracking_enabled = True
            self.waiting_for_auto_tracking = False
            self.get_logger().info(
                f'>>> Pilot car detected - auto tracking ENABLED '
                f'(drone is {dist_to_static_goal:.1f}m from spawn point)')

        self.goal = odom_goal
        self.goal_velocity = 0.7 * self.goal_velocity + 0.3 * odom_velocity
        self.goal_received = True
        self.last_goal_odom_time = self.get_clock().now()

    def tracking_enabled_cb(self, msg: Bool):
        self.tracking_enabled = bool(msg.data)
        if self.tracking_enabled:
            self.waiting_for_auto_tracking = False
            self.get_logger().info('>>> Ambulance velocity tracking ENABLED')
        else:
            self.waiting_for_auto_tracking = self.auto_tracking_enabled and self.goal_received
            self.last_goal_odom_time = None
            self.goal_velocity = np.zeros(3)
            self.get_logger().info(
                f'>>> Ambulance velocity tracking DISABLED '
                f'(auto_tracking_wait={self.waiting_for_auto_tracking})')

    def cloud_cb(self, msg: PointCloud2):
        self.cloud_rx_count += 1
        self.last_cloud_rx_time = self.get_clock().now()

        with self.cloud_lock:
            if self.latest_cloud_id != self.processed_cloud_id:
                self.cloud_overwrite_count += 1
            self.latest_cloud_msg = msg
            self.latest_cloud_id += 1
        self.cloud_worker_wake.set()

    def _cloud_worker_loop(self):
        period = 1.0 / self.cloud_process_hz
        next_process_time = 0.0

        while not self.cloud_worker_stop.is_set():
            timeout = max(0.0, next_process_time - time.monotonic())
            self.cloud_worker_wake.wait(timeout)
            self.cloud_worker_wake.clear()

            if self.cloud_worker_stop.is_set():
                break

            now = time.monotonic()
            if now < next_process_time:
                continue

            with self.cloud_lock:
                if self.latest_cloud_msg is None:
                    next_process_time = now + period
                    continue
                if self.latest_cloud_id == self.processed_cloud_id:
                    next_process_time = now + period
                    continue
                msg = self.latest_cloud_msg
                cloud_id = self.latest_cloud_id

            self._process_cloud_msg(msg, cloud_id)
            next_process_time = time.monotonic() + period

    def _process_cloud_msg(self, msg: PointCloud2, cloud_id: int):
        start_monotonic = time.monotonic()

        try:
            body = self._extract_obstacles_body(msg)
        except Exception as exc:
            self.get_logger().warn(f'Point cloud processing failed: {exc}')
            body = np.empty((0, 3), dtype=np.float32)

        self.last_cloud_time = self.get_clock().now()
        self.cloud_last_process_ms = (time.monotonic() - start_monotonic) * 1000.0
        self.cloud_proc_count += 1
        with self.cloud_lock:
            self.processed_cloud_id = max(self.processed_cloud_id, cloud_id)
        self.cloud_stale_warned = False

        if body.shape[0] > 0:
            self.cloud_used_count += 1
        self._set_obstacles_and_publish(body)

    def _extract_obstacles_body(self, msg: PointCloud2) -> np.ndarray:
        pts = self._cloud_to_xyz_array(msg)
        if pts.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)

        if self.gpu_enabled:
            return self._extract_obstacles_body_gpu(pts)

        return self._extract_obstacles_body_cpu(pts)

    def _extract_obstacles_body_cpu(self, pts: np.ndarray) -> np.ndarray:
        finite_mask = np.isfinite(pts).all(axis=1)
        pts = pts[finite_mask]
        if pts.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)

        body = np.column_stack([
            pts[:, 0],
            -pts[:, 1],
            -pts[:, 2],
        ]).astype(np.float32, copy=False)

        z_mask = (body[:, 2] < self.ground_filter_z) & (body[:, 2] > self.ceil_filter_z)
        body = body[z_mask]
        if body.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)

        d = np.linalg.norm(body, axis=1)
        body = body[d < self.d_max]
        if body.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)

        if body.shape[0] > self.max_cloud_points:
            body = self._voxel_downsample(body, self.voxel_size)
        if body.shape[0] > self.max_cloud_points:
            step = int(np.ceil(body.shape[0] / self.max_cloud_points))
            body = body[::step]

        return body.astype(np.float32, copy=False)

    def _extract_obstacles_body_gpu(self, pts: np.ndarray) -> np.ndarray:
        cp = self.cp
        try:
            gpts = cp.asarray(pts, dtype=cp.float32)

            finite_mask = cp.isfinite(gpts).all(axis=1)
            gpts = gpts[finite_mask]
            if int(gpts.shape[0]) == 0:
                return np.empty((0, 3), dtype=np.float32)

            body = cp.empty_like(gpts)
            body[:, 0] = gpts[:, 0]
            body[:, 1] = -gpts[:, 1]
            body[:, 2] = -gpts[:, 2]

            z_mask = (body[:, 2] < self.ground_filter_z) & (body[:, 2] > self.ceil_filter_z)
            body = body[z_mask]
            if int(body.shape[0]) == 0:
                return np.empty((0, 3), dtype=np.float32)

            d = cp.linalg.norm(body, axis=1)
            body = body[d < self.d_max]
            if int(body.shape[0]) == 0:
                return np.empty((0, 3), dtype=np.float32)

            if int(body.shape[0]) > self.max_cloud_points:
                d = cp.linalg.norm(body, axis=1)
                keep_n = min(self.max_cloud_points, int(body.shape[0]))
                keep_idx = cp.argpartition(d, keep_n - 1)[:keep_n]
                body = body[keep_idx]

            return cp.asnumpy(body).astype(np.float32, copy=False)
        except Exception as exc:
            self.get_logger().warn(f'GPU cloud filtering failed ({exc}); falling back to CPU')
            self.gpu_enabled = False
            return self._extract_obstacles_body_cpu(pts)

    def _cloud_to_xyz_array(self, msg: PointCloud2) -> np.ndarray:
        point_count = max(1, int(msg.width) * int(msg.height))
        stride = max(
            self.cloud_sample_stride,
            int(np.ceil(point_count / self.max_raw_cloud_points)),
        )

        if msg.height > 1 and msg.width > 1 and stride > 1:
            return self._cloud_to_xyz_array_fallback(msg, stride)

        try:
            raw = pc2.read_points_numpy(
                msg, field_names=('x', 'y', 'z'), skip_nans=False)
        except AttributeError:
            return self._cloud_to_xyz_array_fallback(msg, stride)
        except Exception as exc:
            self.get_logger().warn(f'Point cloud conversion failed: {exc}')
            return np.empty((0, 3), dtype=np.float32)

        if raw.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        if raw.dtype.fields:
            pts = np.column_stack([raw['x'], raw['y'], raw['z']])
        else:
            pts = np.asarray(raw).reshape(-1, 3)

        if pts.shape[0] > self.max_raw_cloud_points:
            stride = max(stride, int(np.ceil(pts.shape[0] / self.max_raw_cloud_points)))
            pts = pts[::stride]

        return np.asarray(pts, dtype=np.float32)

    def _cloud_to_xyz_array_fallback(self, msg: PointCloud2, stride: int) -> np.ndarray:
        try:
            if msg.height > 1 and msg.width > 1:
                grid_step = max(
                    1,
                    int(np.ceil(np.sqrt((msg.width * msg.height) / self.max_raw_cloud_points))),
                )
                uvs = [
                    (u, v)
                    for v in range(0, msg.height, grid_step)
                    for u in range(0, msg.width, grid_step)
                ]
                raw = pc2.read_points(
                    msg, field_names=('x', 'y', 'z'), skip_nans=False, uvs=uvs)
                return np.asarray([[p[0], p[1], p[2]] for p in raw], dtype=np.float32)

            sampled = [
                [p[0], p[1], p[2]]
                for i, p in enumerate(pc2.read_points(
                    msg, field_names=('x', 'y', 'z'), skip_nans=False))
                if i % stride == 0
            ]
            return np.asarray(sampled, dtype=np.float32)
        except Exception as exc:
            self.get_logger().warn(f'Point cloud fallback conversion failed: {exc}')
            return np.empty((0, 3), dtype=np.float32)

    def _set_obstacles_and_publish(self, body: np.ndarray):
        self.obstacles_body = body

        min_d_msg = Float32()
        if body.shape[0] > 0:
            min_d_msg.data = float(np.min(np.linalg.norm(body, axis=1)))
        else:
            min_d_msg.data = float('nan')
        self.pub_obs_dist.publish(min_d_msg)

    # ======================================================================= #
    # Cloud freshness helper                                                  #
    # ======================================================================= #

    def _check_cloud_freshness(self):
        """
        Returns 'fresh', 'stale', or 'blind' based on the last processed cloud.
        Empty-but-fresh processed clouds mean the sensor is alive and the path is clear.
        """
        if self.last_cloud_time is None:
            return 'blind'

        age = (self.get_clock().now() - self.last_cloud_time).nanoseconds / 1e9

        if age > self.cloud_blind_threshold:
            if not self.cloud_stale_warned:
                self.get_logger().warn(
                    f'>>> CLOUD BLIND: no point cloud for {age:.1f}s - flying cautiously')
                self.cloud_stale_warned = True
            self.obstacles_body = np.empty((0, 3), dtype=np.float32)
            return 'blind'

        if age > self.cloud_stale_threshold:
            self.obstacles_body = np.empty((0, 3), dtype=np.float32)
            return 'stale'

        return 'fresh'

    # ======================================================================= #
    # APF Core                                                                #
    # ======================================================================= #

    def compute_velocity(self):
        """Returns (v_ned, dist_to_goal, mode_string)."""
        to_goal = self.goal - self.position
        dist_to_goal = float(np.linalg.norm(to_goal))
        in_tracking_mode = self._tracking_goal_is_fresh()

        if not in_tracking_mode and dist_to_goal < self.tol:
            return np.zeros(3), dist_to_goal, 'GOAL_REACHED'

        cloud_status = self._check_cloud_freshness()

        now = self.get_clock().now()
        if cloud_status == 'blind':
            self.was_blind = True
        elif self.was_blind and cloud_status == 'fresh':
            self.blind_recovery_until = now + Duration(seconds=self.recovery_duration_s)
            self.was_blind = False
            self.get_logger().info(
                f'>>> Cloud restored - entering {self.recovery_duration_s:.1f}s recovery')

        recovering = (
            self.blind_recovery_until is not None and
            now < self.blind_recovery_until
        )
        rep_scale = self.recovery_softness if recovering else 1.0
        effective_k_rep = self.k_rep * rep_scale
        effective_danger_speed = self.danger_speed * rep_scale
        effective_up_bias = self.up_bias_strength * rep_scale

        F_att = self.k_att * to_goal
        F_rep_body = np.zeros(3)
        n_rep = 0
        up_bias = 0.0
        danger_velocity_ned = None
        danger_weight = 0.0
        min_d = None

        if self.obstacles_body.shape[0] > 0:
            d = np.linalg.norm(self.obstacles_body, axis=1)
            d = np.maximum(d, 0.1)
            min_d = float(np.min(d))

            if min_d < self.danger_radius:
                closest = self.obstacles_body[np.argmin(d)]
                flee_dir_body = -closest / (np.linalg.norm(closest) + 1e-6)
                if flee_dir_body[2] > 0:
                    flee_dir_body[2] = -abs(flee_dir_body[2]) * 0.5
                flee_dir_body = flee_dir_body / (np.linalg.norm(flee_dir_body) + 1e-6)
                danger_velocity_ned = self._body_to_ned(flee_dir_body)
                urgency = float(np.clip(1.0 - (min_d / self.danger_radius), 0.0, 1.0))
                danger_weight = urgency ** self.danger_blend_power
                speed = effective_danger_speed * (0.4 + 0.6 * urgency)
                danger_velocity_ned = danger_velocity_ned * speed

            mask = d < self.d_max
            if np.any(mask):
                obs = self.obstacles_body[mask]
                d_m = d[mask]
                mag = effective_k_rep * (1.0 / d_m - 1.0 / self.d_max) / (d_m ** 2)
                unit_away = -obs / d_m[:, None]

                downward = unit_away[:, 2] > 0
                unit_away[downward, 2] *= 0.3
                upward = unit_away[:, 2] < 0
                unit_away[upward, 2] *= self.up_preference

                raw_rep_body = np.sum(unit_away * mag[:, None], axis=0)
                alpha = self.repulsion_smoothing_alpha
                self.rep_filt_body = (1.0 - alpha) * self.rep_filt_body + alpha * raw_rep_body
                F_rep_body = self.rep_filt_body
                n_rep = int(np.sum(mask))

                forward_pts = obs[(obs[:, 0] > 0) & (np.abs(obs[:, 1]) < obs[:, 0])]
                if forward_pts.shape[0] > 5:
                    min_fwd_d = float(np.min(np.linalg.norm(forward_pts, axis=1)))
                    blockage = float(np.clip(1.0 - (min_fwd_d / self.d_max), 0.0, 1.0))
                    up_bias = -effective_up_bias * blockage
        else:
            alpha = self.repulsion_smoothing_alpha
            self.rep_filt_body = (1.0 - alpha) * self.rep_filt_body

        F_rep_ned = self._body_to_ned(F_rep_body)
        F_total = F_att + F_rep_ned
        F_total[2] += up_bias

        v_xy = F_total[:2]
        speed_sync_active = (
            in_tracking_mode and
            float(np.linalg.norm(to_goal[:2])) <= self.speed_sync_radius
        )
        if speed_sync_active:
            v_xy = v_xy + self.feedforward_gain * self.goal_velocity[:2]

        v_des = np.array([
            v_xy[0],
            v_xy[1],
            float(np.clip(F_total[2], -self.vz_max, self.vz_max)),
        ])
        v_des = self._limit_velocity(v_des)

        if danger_velocity_ned is not None and danger_weight > 0.0:
            danger_velocity_ned = self._limit_velocity(danger_velocity_ned)
            v_des = (1.0 - danger_weight) * v_des + danger_weight * danger_velocity_ned
            v_des = self._limit_velocity(v_des)

        if cloud_status == 'blind':
            v_des = v_des * self.blind_speed_factor

        if n_rep > 0:
            mode = f'APF(rep_pts={n_rep}, up_bias={up_bias:+.1f})'
        else:
            mode = 'APF(clear)'

        if in_tracking_mode:
            goal_speed = float(np.linalg.norm(self.goal_velocity[:2]))
            xy_error = float(np.linalg.norm(to_goal[:2]))
            track_state = 'SYNC' if speed_sync_active else 'POSITION_FIRST'
            mode = f'TRACK/{track_state}(v_goal={goal_speed:.1f}, e_xy={xy_error:.1f})|{mode}'
        if danger_velocity_ned is not None:
            mode = f'{mode}|DANGER_BLEND(d={min_d:.2f},w={danger_weight:.2f})'
        if recovering:
            mode = f'{mode}|RECOVERY'
        if cloud_status != 'fresh':
            mode = f'{mode}|{cloud_status.upper()}'

        return v_des, dist_to_goal, mode

    def _tracking_goal_is_fresh(self) -> bool:
        return (
            self.tracking_enabled and
            self.last_goal_odom_time is not None and
            (self.get_clock().now() - self.last_goal_odom_time).nanoseconds < 1e9
        )

    def _body_to_ned(self, vec_body: np.ndarray) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return R @ vec_body

    @staticmethod
    def _limit_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm > max_norm:
            return vec * (max_norm / (norm + 1e-6))
        return vec

    def _limit_velocity(self, v_ned: np.ndarray) -> np.ndarray:
        limited = np.array(v_ned, dtype=float)
        spd_xy = float(np.linalg.norm(limited[:2]))
        if spd_xy > self.v_max:
            limited[:2] = limited[:2] * (self.v_max / spd_xy)
        limited[2] = float(np.clip(limited[2], -self.vz_max, self.vz_max))
        return limited

    @staticmethod
    def _voxel_downsample(pts: np.ndarray, voxel_size: float) -> np.ndarray:
        keys = np.floor(pts / voxel_size).astype(np.int32)
        _, idx = np.unique(keys, axis=0, return_index=True)
        return pts[idx]

    def _smooth_velocity(self, v_target: np.ndarray) -> np.ndarray:
        now = self.get_clock().now()
        if self.last_control_time is None:
            dt = 0.05
        else:
            dt = (now - self.last_control_time).nanoseconds / 1e9
            dt = float(np.clip(dt, 0.001, 0.2))
        self.last_control_time = now

        alpha = self.command_smoothing_alpha
        filtered_target = (1.0 - alpha) * self.v_filt + alpha * v_target
        delta = filtered_target - self.v_filt
        delta_norm = float(np.linalg.norm(delta))
        max_delta = self.max_accel * dt
        if delta_norm > max_delta:
            delta = delta * (max_delta / (delta_norm + 1e-6))

        self.v_filt = self._limit_velocity(self.v_filt + delta)
        return self.v_filt

    def _desired_yaw(self, v_ned: np.ndarray) -> float:
        if float(np.linalg.norm(v_ned[:2])) < 0.1:
            return float(self.yaw)
        return float(np.arctan2(v_ned[1], v_ned[0]))

    # ======================================================================= #
    # Main control loop (20 Hz)                                               #
    # ======================================================================= #

    def control_loop(self):
        self._publish_offboard_heartbeat()

        if self.offboard_counter == 10:
            self._set_offboard_mode()
            self._arm()
        if self.offboard_counter <= 10:
            self.offboard_counter += 1

        # Phase 1: Takeoff
        if not self.has_taken_off:
            if self.position[2] > self.takeoff_alt + 0.5:
                v_cmd = self._smooth_velocity(np.array([0.0, 0.0, -0.6]))
                self._publish_velocity_setpoint(v_cmd, float(self.yaw))
                return

            self.has_taken_off = True
            self.hover_position = self.position.copy()
            self.v_filt = np.zeros(3)
            self.get_logger().info(
                f'>>> Takeoff complete at NED z={self.position[2]:.2f}m. '
                f'Hovering - waiting for waypoint on /apf/goal.')

        # Phase 2: Hover until first waypoint
        if not self.goal_received:
            error = self.hover_position - self.position
            v_target = np.clip(error * 0.5, -0.5, 0.5)
            v_cmd = self._smooth_velocity(v_target)
            self._publish_velocity_setpoint(v_cmd, float(self.yaw))
            self._last_mode = 'HOVER_WAITING'
            return

        # Phase 3: APF flight to goal
        v_target, dist_to_goal, mode = self.compute_velocity()
        self._last_mode = mode
        in_tracking_mode = self._tracking_goal_is_fresh()

        if (not in_tracking_mode) and dist_to_goal < self.tol:
            error = self.goal - self.position
            # Dead-band: within 0.4 m just command zero and let PX4 hold position.
            # Outside that, use a gentle proportional nudge to avoid circling.
            if dist_to_goal < 0.4:
                v_target = np.zeros(3)
            else:
                v_target = np.clip(error * 0.25, -0.4, 0.4)
            self._last_mode = f'HOVER_AT_GOAL(d={dist_to_goal:.2f}m)'
            self.stuck_counter = 0
        else:
            if float(np.linalg.norm(v_target)) < 0.1:
                self.stuck_counter += 1
                if self.stuck_counter > 40:
                    self.get_logger().warn('>>> STUCK - applying sideways perturbation')
                    v_target = np.array([0.0, 0.5, 0.0])
                    self.stuck_counter = 0
            else:
                self.stuck_counter = 0

        v_cmd = self._smooth_velocity(v_target)
        self._publish_velocity_setpoint(v_cmd, self._desired_yaw(v_cmd))

    # ======================================================================= #
    # Periodic diagnostic print                                               #
    # ======================================================================= #

    def status_print(self):
        cloud_rx_rate = self.cloud_rx_count / self.verbose_period
        cloud_proc_rate = self.cloud_proc_count / self.verbose_period
        cloud_used_rate = self.cloud_used_count / self.verbose_period
        cloud_overwrite_rate = self.cloud_overwrite_count / self.verbose_period
        pos_rx_rate = self.position_rx_count / self.verbose_period
        stat_rx_rate = self.status_rx_count / self.verbose_period
        pos_rate = self.position_count / self.verbose_period
        stat_rate = self.status_count / self.verbose_period
        self.cloud_rx_count = 0
        self.cloud_proc_count = 0
        self.cloud_used_count = 0
        self.cloud_overwrite_count = 0
        self.position_rx_count = 0
        self.status_rx_count = 0
        self.position_count = 0
        self.status_count = 0

        if self.last_cloud_rx_time is not None:
            cloud_rx_age = (self.get_clock().now() - self.last_cloud_rx_time).nanoseconds / 1e9
            rx_age_str = f'rx_age={cloud_rx_age:.1f}s'
        else:
            rx_age_str = 'rx_age=never'

        if self.last_cloud_time is not None:
            cloud_age = (self.get_clock().now() - self.last_cloud_time).nanoseconds / 1e9
            proc_age_str = f'proc_age={cloud_age:.1f}s'
        else:
            proc_age_str = 'proc_age=never'

        if self.obstacles_body.shape[0] > 0:
            d = np.linalg.norm(self.obstacles_body, axis=1)
            min_d = float(np.min(d))
            n_pts = self.obstacles_body.shape[0]
            closest = self.obstacles_body[np.argmin(d)]
            obs_str = (
                f'min={min_d:.2f}m at body'
                f'[{closest[0]:+.1f},{closest[1]:+.1f},{closest[2]:+.1f}] | n_pts={n_pts}'
            )
        else:
            obs_str = 'NONE'

        if self.goal_received:
            dist_to_goal = float(np.linalg.norm(self.goal - self.position))
            goal_str = (
                f'NED[{self.goal[0]:+.2f}, {self.goal[1]:+.2f}, '
                f'{self.goal[2]:+.2f}] dist={dist_to_goal:.2f}m '
                f'tracking={self.tracking_enabled}'
            )
        else:
            goal_str = f'NONE - waiting for waypoint tracking={self.tracking_enabled}'

        mode = getattr(self, '_last_mode', 'INIT')
        arm_str = 'ARMED' if self.armed else f'DISARMED({self.arming_state})'

        self.get_logger().info(
            f'\n--- STATUS ---\n'
            f'  Topics:   cloud_rx={cloud_rx_rate:.1f}Hz cloud_proc={cloud_proc_rate:.1f}Hz '
            f'cloud_used={cloud_used_rate:.1f}Hz overwritten={cloud_overwrite_rate:.1f}Hz '
            f'({rx_age_str}, {proc_age_str}, proc={self.cloud_last_process_ms:.1f}ms) '
            f'pos_rx={pos_rx_rate:.1f}Hz pos_used={pos_rate:.1f}Hz '
            f'status_rx={stat_rx_rate:.1f}Hz status_used={stat_rate:.1f}Hz\n'
            f'  Vehicle:  {arm_str} nav={self.nav_state} taken_off={self.has_taken_off}\n'
            f'  Position: NED[{self.position[0]:+.2f}, {self.position[1]:+.2f}, '
            f'{self.position[2]:+.2f}] yaw={np.rad2deg(self.yaw):.1f}deg\n'
            f'  Goal:     {goal_str}\n'
            f'  Velocity: NED[{self.v_filt[0]:+.2f}, {self.v_filt[1]:+.2f}, '
            f'{self.v_filt[2]:+.2f}] |v|={np.linalg.norm(self.v_filt):.2f}m/s\n'
            f'  Obstacles:{obs_str}\n'
            f'  Mode:     {mode}'
        )

    # ======================================================================= #
    # PX4 helpers                                                             #
    # ======================================================================= #

    def _publish_offboard_heartbeat(self):
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = self._stamp()
        self.pub_offboard.publish(msg)

    def _publish_velocity_setpoint(self, v_ned: np.ndarray, yaw: float):
        msg = TrajectorySetpoint()
        msg.position = [float('nan'), float('nan'), float('nan')]
        msg.velocity = [float(v_ned[0]), float(v_ned[1]), float(v_ned[2])]
        msg.acceleration = [float('nan'), float('nan'), float('nan')]
        msg.yaw = float(yaw)
        msg.yawspeed = float('nan')
        msg.timestamp = self._stamp()
        self.pub_setpoint.publish(msg)

    def _set_offboard_mode(self):
        self._send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0, param2=6.0)
        self.get_logger().info('>>> Offboard mode requested')

    def _arm(self):
        self._send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0)
        self.get_logger().info('>>> Arm command sent')

    def _send_vehicle_command(self, command: int, **kwargs):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = float(kwargs.get('param1', 0.0))
        msg.param2 = float(kwargs.get('param2', 0.0))
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self._stamp()
        self.pub_command.publish(msg)

    def _stamp(self) -> int:
        return self.get_clock().now().nanoseconds // 1000

    def destroy_node(self):
        self.cloud_worker_stop.set()
        self.cloud_worker_wake.set()
        if hasattr(self, 'cloud_worker_thread'):
            self.cloud_worker_thread.join(timeout=1.0)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = APFPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
