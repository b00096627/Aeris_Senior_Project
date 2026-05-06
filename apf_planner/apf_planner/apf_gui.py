#!/usr/bin/env python3
"""
APF Planner GUI Control Panel with live telemetry plots.

Run with:
  ros2 run apf_planner apf_gui
"""

import shlex
import subprocess
import threading
import time
from collections import deque

import matplotlib
import numpy as np
import rclpy
import tkinter as tk
from geometry_msgs.msg import PointStamped
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleLocalPosition
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool
from tkinter import filedialog, messagebox, ttk

matplotlib.use('TkAgg')


# ---------------------------------------------------------------------- #
# Defaults                                                               #
# ---------------------------------------------------------------------- #
DEFAULT_PARAMS = {
    'v_max': 30.0,
    'vz_max': 3.0,
    'k_att': 2.0,
    'k_rep': 15.0,
    'd_max': 8.0,
    'danger_radius': 8.0,
    'danger_speed': 30.0,
    'max_accel': 8.0,
    'command_smoothing_alpha': 0.20,
    'repulsion_smoothing_alpha': 0.25,
    'cloud_stale_threshold': 2.0,
    'cloud_blind_threshold': 5.0,
    'blind_speed_factor': 1.0,
    'cloud_process_hz': 3.0,
    'max_raw_cloud_points': 4000,
    'max_cloud_points': 700,
    'cloud_sample_stride': 1,
    'voxel_size': 0.35,
    'use_gpu': True,
    'local_position_accept_hz': 5.0,
    'status_accept_hz': 5.0,
    'tracking_speed_match_gain': 1.0,
    'tracking_along_gain': 0.8,
    'tracking_lateral_gain': 0.8,
    'tracking_position_gain': 0.9,
    'tracking_max_correction_speed': 5.0,
    'tracking_speed_margin': 5.0,
    'tracking_slow_speed_threshold': 0.4,
    'tracking_hold_radius': 2.0,
    'tracking_deadband_radius': 0.75,
    'tracking_brake_gain': 0.2,
    'tracking_slow_max_speed': 4.0,
    'tracking_enabled': False,
    'auto_tracking_enabled': True,
    'tracking_start_radius': 10.0,
    'goal_odom_accept_hz': 5.0,
    'speed_sync_radius': 5.0,
}

INTEGER_PARAMS = {
    'max_raw_cloud_points',
    'max_cloud_points',
    'cloud_sample_stride',
}

BOOLEAN_PARAMS = {
    'use_gpu',
    'tracking_enabled',
    'auto_tracking_enabled',
}

DEFAULT_WAYPOINT = {
    'x': -490.0,
    'y': 950.0,
    'z': -10.0,
}

PLOT_WINDOW_SECONDS = 30.0
PLOT_UPDATE_HZ = 5
GUI_POSITION_ACCEPT_HZ = 5.0


# ---------------------------------------------------------------------- #
# ROS2 telemetry node                                                    #
# ---------------------------------------------------------------------- #
class TelemetryNode(Node):
    def __init__(self):
        super().__init__('apf_gui_node')

        goal_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub = self.create_publisher(PointStamped, '/apf/goal', goal_qos)
        self.pub_tracking_enabled = self.create_publisher(
            Bool, '/apf/tracking_enabled', goal_qos)

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1',
            self.position_cb, px4_qos)
        self.create_subscription(
            Odometry, '/apf/goal_odom',
            self.goal_odom_cb, goal_qos)

        self.lock = threading.Lock()
        self.t0 = time.time()
        self.last_position_accept_time = None
        self.latest_goal_odom = None
        self.history = {
            't': deque(maxlen=2000),
            'x': deque(maxlen=2000),
            'y': deque(maxlen=2000),
            'z': deque(maxlen=2000),
            'v': deque(maxlen=2000),
        }

    def publish_goal(self, x, y, z, repeats=5, interval_s=0.05):
        for _ in range(max(1, int(repeats))):
            msg = PointStamped()
            msg.header.frame_id = 'map'
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.point.x = float(x)
            msg.point.y = float(y)
            msg.point.z = float(z)
            self.pub.publish(msg)
            time.sleep(interval_s)
        self.get_logger().info(f'Published goal: [{x}, {y}, {z}]')

    def publish_tracking_enabled(self, enabled: bool):
        msg = Bool()
        msg.data = bool(enabled)
        self.pub_tracking_enabled.publish(msg)
        state = 'enabled' if enabled else 'disabled'
        self.get_logger().info(f'Published tracking {state}')

    def position_cb(self, msg: VehicleLocalPosition):
        now = time.monotonic()
        if self.last_position_accept_time is not None:
            if now - self.last_position_accept_time < (1.0 / GUI_POSITION_ACCEPT_HZ):
                return
        self.last_position_accept_time = now

        t = time.time() - self.t0
        speed = float(np.sqrt(msg.vx**2 + msg.vy**2 + msg.vz**2))
        with self.lock:
            self.history['t'].append(t)
            self.history['x'].append(float(msg.x))
            self.history['y'].append(float(msg.y))
            self.history['z'].append(float(msg.z))
            self.history['v'].append(speed)

    def goal_odom_cb(self, msg: Odometry):
        speed = float(np.sqrt(
            msg.twist.twist.linear.x**2 +
            msg.twist.twist.linear.y**2 +
            msg.twist.twist.linear.z**2))
        snap = {
            'x': float(msg.pose.pose.position.x),
            'y': float(msg.pose.pose.position.y),
            'z': float(msg.pose.pose.position.z),
            'vx': float(msg.twist.twist.linear.x),
            'vy': float(msg.twist.twist.linear.y),
            'vz': float(msg.twist.twist.linear.z),
            'speed': speed,
            't_wall': time.time(),
        }
        with self.lock:
            self.latest_goal_odom = snap

    def goal_odom_snapshot(self):
        with self.lock:
            if self.latest_goal_odom is None:
                return None
            return dict(self.latest_goal_odom)

    def snapshot(self):
        with self.lock:
            return {k: np.array(list(v)) for k, v in self.history.items()}


# ---------------------------------------------------------------------- #
# Main GUI                                                               #
# ---------------------------------------------------------------------- #
class APFGui:
    def __init__(self, root, ros_node):
        self.root = root
        self.ros_node = ros_node
        self.apf_process = None
        self.param_vars = {
            key: tk.StringVar(value=str(default))
            for key, default in DEFAULT_PARAMS.items()
        }
        self.wp_vars = {
            key: tk.StringVar(value=str(default))
            for key, default in DEFAULT_WAYPOINT.items()
        }
        self.amb_status_var = tk.StringVar(value='Ambulance: waiting for /apf/goal_odom')

        root.title('APF Planner Control Panel')
        root.geometry('1100x920')
        root.configure(bg='#f0f0f0')

        style = ttk.Style()
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'), background='#f0f0f0')
        style.configure('Section.TLabelframe.Label', font=('Arial', 11, 'bold'))
        style.configure('Action.TButton', font=('Arial', 11, 'bold'), padding=10)

        ttk.Label(root, text='APF Drone Control Panel',
                  style='Title.TLabel').pack(pady=10)

        self._build_top_launch_bar(root)

        main = ttk.Frame(root)
        main.pack(fill='both', expand=True, padx=12, pady=4)
        right = ttk.Frame(main)
        right.pack(side='right', fill='both', expand=True)

        self._build_status_panel(main)
        self._build_plot_panel(right)

        self.status_var = tk.StringVar(value='Ready')
        ttk.Label(root, textvariable=self.status_var,
                  relief='sunken', anchor='w',
                  font=('Arial', 9), padding=4).pack(side='bottom', fill='x')

        self._schedule_plot_update()
        self._schedule_ambulance_update()

    # =================================================================== #
    # Always-visible launch bar                                           #
    # =================================================================== #
    def _build_top_launch_bar(self, parent):
        frame = ttk.Frame(parent, padding=(12, 0, 12, 8))
        frame.pack(fill='x')

        ttk.Button(frame, text='Launch APF',
                   style='Action.TButton',
                   command=self._launch_apf).pack(side='left', padx=(0, 6))
        ttk.Button(frame, text='Publish Point',
                   style='Action.TButton',
                   command=self._publish_default_waypoint).pack(side='left', padx=6)
        ttk.Button(frame, text='Stop APF',
                   command=self._stop_apf).pack(side='left', padx=6)
        ttk.Button(frame, text='Reset',
                   command=self._reset_all).pack(side='left', padx=6)

    def _build_status_panel(self, parent):
        frame = ttk.LabelFrame(parent, text=' Mission ',
                               style='Section.TLabelframe', padding=12)
        frame.pack(side='left', fill='y', padx=(0, 8))

        waypoint = (
            f'Spawn point: NED[{DEFAULT_WAYPOINT["x"]:+.1f}, '
            f'{DEFAULT_WAYPOINT["y"]:+.1f}, {DEFAULT_WAYPOINT["z"]:+.1f}]'
        )
        ttk.Label(frame, text=waypoint, font=('Arial', 10, 'bold')).pack(anchor='w')
        ttk.Label(frame, text='Click Publish Point once before the pilot car spawns.',
                  font=('Arial', 9), foreground='#444').pack(anchor='w', pady=(6, 0))
        ttk.Label(frame, text='Tracking starts automatically when /apf/goal_odom appears.',
                  font=('Arial', 9), foreground='#444').pack(anchor='w', pady=(2, 10))
        ttk.Label(frame, textvariable=self.amb_status_var,
                  font=('Arial', 9), foreground='#444',
                  wraplength=300).pack(anchor='w', pady=(8, 0))

    def _reset_params(self):
        for key, default in DEFAULT_PARAMS.items():
            self.param_vars[key].set(str(default))
        self.status_var.set('Parameters reset')

    def _reset_all(self):
        self._reset_params()
        self._reset_waypoint()
        self._clear_history()
        self.status_var.set('Reset to defaults')

    def _launch_apf(self):
        if self.apf_process is not None and self.apf_process.poll() is None:
            messagebox.showwarning('Already running', 'APF node is already running.')
            return

        try:
            params = {}
            for key, var in self.param_vars.items():
                if key in BOOLEAN_PARAMS:
                    value = var.get().strip().lower()
                    if value in ('1', 'true', 'yes', 'y', 'on'):
                        params[key] = True
                    elif value in ('0', 'false', 'no', 'n', 'off'):
                        params[key] = False
                    else:
                        raise ValueError(f'{key} must be true or false')
                elif key in INTEGER_PARAMS:
                    params[key] = int(float(var.get()))
                else:
                    params[key] = float(var.get())
        except ValueError as e:
            messagebox.showerror('Invalid parameter', str(e))
            return

        cmd_parts = ['ros2', 'run', 'apf_planner', 'apf_node', '--ros-args']
        for key, val in params.items():
            if isinstance(val, bool):
                val_str = str(val).lower()
            else:
                val_str = str(val)
            cmd_parts.extend(['-p', f'{key}:={val_str}'])
        cmd_str = shlex.join(cmd_parts)

        bash_cmd = (
            'bash -c '
            + shlex.quote(
                'source ~/px4_ros_com_ros2/install/setup.bash && '
                f'{cmd_str}; '
                'echo; echo "[APF node exited - press Enter to close]"; read'
            )
        )
        terminal_cmd = ['terminator', '-T', 'APF Node', '-e', bash_cmd]

        try:
            self.apf_process = subprocess.Popen(terminal_cmd)
            self.status_var.set(
                f'Launched APF | v_max={params["v_max"]} k_rep={params["k_rep"]}')
        except FileNotFoundError:
            self.apf_process = subprocess.Popen(cmd_parts)
            self.status_var.set('Launched APF (no terminator)')

    def _stop_apf(self):
        if self.apf_process is None or self.apf_process.poll() is not None:
            self.status_var.set('No APF running')
            return
        try:
            subprocess.run(['pkill', '-f', 'apf_planner apf_node'], check=False)
            self.status_var.set('APF node stopped')
        except Exception as e:
            self.status_var.set(f'Stop failed: {e}')
        self.apf_process = None

    def _reset_waypoint(self):
        for key, default in DEFAULT_WAYPOINT.items():
            self.wp_vars[key].set(str(default))
        self.status_var.set('Waypoint reset')

    def _publish_default_waypoint(self):
        self._reset_waypoint()
        self._publish_waypoint()

    def _publish_waypoint(self):
        try:
            x = float(self.wp_vars['x'].get())
            y = float(self.wp_vars['y'].get())
            z = float(self.wp_vars['z'].get())
        except ValueError as e:
            messagebox.showerror('Invalid waypoint', str(e))
            return
        try:
            self.ros_node.publish_tracking_enabled(False)
            self.ros_node.publish_goal(x, y, z)
            self.status_var.set(f'Published: NED[{x:.1f}, {y:.1f}, {z:.1f}]')
        except Exception as e:
            messagebox.showerror('Publish failed', str(e))

    # =================================================================== #
    # Panel 3: Live Plots                                                 #
    # =================================================================== #
    def _build_plot_panel(self, parent):
        frame = ttk.LabelFrame(parent, text=' Live Telemetry ',
                               style='Section.TLabelframe', padding=8)
        frame.pack(fill='both', expand=True, pady=4)

        self.fig = Figure(figsize=(7, 6), dpi=90, tight_layout=True)
        self.ax_pos = self.fig.add_subplot(211)
        self.ax_vel = self.fig.add_subplot(212)

        self.line_x, = self.ax_pos.plot([], [], '-', color='#d62728', label='X (north)', linewidth=1.5)
        self.line_y, = self.ax_pos.plot([], [], '-', color='#2ca02c', label='Y (east)', linewidth=1.5)
        self.line_z, = self.ax_pos.plot([], [], '-', color='#1f77b4', label='Z (down)', linewidth=1.5)
        self.line_v, = self.ax_vel.plot([], [], '-', color='#9467bd', linewidth=1.5)
        self.vel_ref = None

        self.ax_pos.set_ylabel('Position (m)', fontsize=10)
        self.ax_pos.set_title('Drone position (NED)', fontsize=11)
        self.ax_pos.legend(loc='upper left', fontsize=8, ncol=3)
        self.ax_pos.grid(alpha=0.3)

        self.ax_vel.set_ylabel('Speed (m/s)', fontsize=10)
        self.ax_vel.set_xlabel('Time (s)', fontsize=10)
        self.ax_vel.set_title('Drone velocity magnitude', fontsize=11)
        self.ax_vel.grid(alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def _clear_history(self):
        with self.ros_node.lock:
            for key in self.ros_node.history:
                self.ros_node.history[key].clear()
            self.ros_node.t0 = time.time()
        self.status_var.set('Telemetry cleared')

    def _save_plot(self):
        path = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[('PNG image', '*.png'), ('PDF', '*.pdf')],
            initialfile='apf_telemetry.png',
            title='Save plot')
        if path:
            self.fig.savefig(path, dpi=150, bbox_inches='tight')
            self.status_var.set(f'Plot saved: {path}')

    def _schedule_plot_update(self):
        self._update_plots()
        self.root.after(int(1000 / PLOT_UPDATE_HZ), self._schedule_plot_update)

    def _schedule_ambulance_update(self):
        snap = self.ros_node.goal_odom_snapshot()
        if snap is None:
            self.amb_status_var.set('Ambulance: waiting for /apf/goal_odom')
        else:
            age = time.time() - snap['t_wall']
            self.amb_status_var.set(
                f'Ambulance: NED[{snap["x"]:+.1f}, {snap["y"]:+.1f}, {snap["z"]:+.1f}] '
                f'|v|={snap["speed"]:.1f}m/s age={age:.1f}s')
        self.root.after(500, self._schedule_ambulance_update)

    def _update_plots(self):
        snap = self.ros_node.snapshot()
        t = snap['t']
        if t.size < 2:
            return

        t_now = t[-1]
        mask = t > (t_now - PLOT_WINDOW_SECONDS)
        if not np.any(mask):
            return
        t_w = t[mask]

        self.line_x.set_data(t_w, snap['x'][mask])
        self.line_y.set_data(t_w, snap['y'][mask])
        self.line_z.set_data(t_w, snap['z'][mask])
        self.ax_pos.set_xlim(t_w[0], t_w[-1] + 0.5)
        self.ax_pos.relim()
        self.ax_pos.autoscale_view(scaley=True, scalex=False)

        self.line_v.set_data(t_w, snap['v'][mask])
        self.ax_vel.set_xlim(t_w[0], t_w[-1] + 0.5)
        try:
            v_max = float(self.param_vars['v_max'].get())
            if self.vel_ref is None:
                self.vel_ref = self.ax_vel.axhline(
                    v_max, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='v_max')
                self.ax_vel.legend(loc='upper right', fontsize=8)
            else:
                self.vel_ref.set_ydata([v_max, v_max])
        except ValueError:
            pass
        self.ax_vel.relim()
        self.ax_vel.autoscale_view(scaley=True, scalex=False)

        self.canvas.draw_idle()


# ---------------------------------------------------------------------- #
# Entry point                                                            #
# ---------------------------------------------------------------------- #
def main():
    rclpy.init()
    ros_node = TelemetryNode()
    threading.Thread(target=lambda: rclpy.spin(ros_node), daemon=True).start()
    root = tk.Tk()
    APFGui(root, ros_node)
    try:
        root.mainloop()
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

