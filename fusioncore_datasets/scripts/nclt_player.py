#!/usr/bin/env python3
"""
NCLT dataset player for FusionCore benchmarking.

Reads the University of Michigan NCLT CSV files and publishes them as
standard ROS2 sensor topics with a simulated clock. Both FusionCore and
robot_localization consume the same data stream.

NCLT download: http://robots.engin.umich.edu/nclt/
Expected files in <data_dir>/:
  ms25_euler.csv          utime, roll, pitch, yaw, wx, wy, wz, ax, ay, az
  gps.csv                 utime, lat, lon, alt, mode
  odometry_mu_100hz.csv   utime, x, y, heading   (integrated wheel odometry)
  groundtruth_*.csv       utime, lat, lon, alt    (RTK post-processed, optional)

Usage:
  ros2 run fusioncore_datasets nclt_player.py \
    --ros-args -p data_dir:=/path/to/nclt/2012-01-08 \
               -p playback_rate:=1.0
"""

import csv
import math
import os
import threading
import time

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion


# ─── helpers ──────────────────────────────────────────────────────────────────

def utime_to_ros(utime_us: int) -> Time:
    ns = utime_us * 1000
    t = Time()
    t.sec = int(ns // 1_000_000_000)
    t.nanosec = int(ns % 1_000_000_000)
    return t


def euler_to_quat(roll: float, pitch: float, yaw: float) -> Quaternion:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def angle_diff(a: float, b: float) -> float:
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


# ─── node ─────────────────────────────────────────────────────────────────────

class NCLTPlayer(Node):

    def __init__(self):
        super().__init__('nclt_player')

        self.declare_parameter('data_dir', '')
        self.declare_parameter('playback_rate', 1.0)
        # Optional: truncate playback. 0 = play all.
        self.declare_parameter('duration_s', 0.0)

        data_dir = self.get_parameter('data_dir').value
        if not data_dir:
            raise RuntimeError('nclt_player: data_dir parameter is required')

        self._rate = self.get_parameter('playback_rate').value
        self._duration = self.get_parameter('duration_s').value

        self._clock_pub = self.create_publisher(Clock,      '/clock',       10)
        self._imu_pub   = self.create_publisher(Imu,        '/imu/data',    50)
        self._gps_pub   = self.create_publisher(NavSatFix,  '/gnss/fix',    10)
        self._odom_pub  = self.create_publisher(Odometry,   '/odom/wheels', 50)

        self.get_logger().info(f'Loading NCLT data from: {data_dir}')
        self._events = []   # list of (utime_us, kind, payload)
        self._load_imu(os.path.join(data_dir, 'ms25_euler.csv'))
        self._load_gps(os.path.join(data_dir, 'gps.csv'))
        self._load_odom(self._find_odom(data_dir))

        self._events.sort(key=lambda e: e[0])
        total = len(self._events)
        self.get_logger().info(
            f'Loaded {total} events ({self._count("imu")} IMU, '
            f'{self._count("gps")} GPS, {self._count("odom")} odom). '
            f'Playback rate: {self._rate}x')

        t = threading.Thread(target=self._play, daemon=True)
        t.start()

    # ── loaders ───────────────────────────────────────────────────────────────

    def _find_odom(self, data_dir: str) -> str:
        for name in ('odometry_mu_100hz.csv', 'odometry_100hz.csv', 'wheels.csv'):
            p = os.path.join(data_dir, name)
            if os.path.exists(p):
                return p
        raise RuntimeError(
            f'nclt_player: no wheel odometry file found in {data_dir}. '
            'Expected odometry_mu_100hz.csv')

    def _load_imu(self, path: str):
        count = 0
        with open(path) as f:
            for row in csv.reader(f):
                if not row or row[0].startswith('#'):
                    continue
                try:
                    utime = int(row[0])
                    # roll, pitch, yaw (rad) | wx, wy, wz (rad/s) | ax, ay, az (m/s²)
                    vals = [float(v) for v in row[1:10]]
                    self._events.append((utime, 'imu', vals))
                    count += 1
                except (ValueError, IndexError):
                    continue
        self.get_logger().info(f'  IMU: {count} rows from {os.path.basename(path)}')

    def _load_gps(self, path: str):
        count = 0
        skipped = 0
        with open(path) as f:
            for row in csv.reader(f):
                if not row or row[0].startswith('#'):
                    continue
                try:
                    utime = int(row[0])
                    lat, lon, alt = float(row[1]), float(row[2]), float(row[3])
                    mode = int(float(row[4]))  # 1=no_fix, 2=2D, 3=3D
                    if mode < 2:
                        skipped += 1
                        continue
                    self._events.append((utime, 'gps', [lat, lon, alt, mode]))
                    count += 1
                except (ValueError, IndexError):
                    continue
        self.get_logger().info(
            f'  GPS: {count} fixes ({skipped} no-fix skipped) from {os.path.basename(path)}')

    def _load_odom(self, path: str):
        # NCLT provides integrated odometry (x, y, heading). Differentiate to
        # get body-frame velocity (vx, omega) for FusionCore's encoder update.
        rows = []
        with open(path) as f:
            for row in csv.reader(f):
                if not row or row[0].startswith('#'):
                    continue
                try:
                    utime = int(row[0])
                    x, y, h = float(row[1]), float(row[2]), float(row[3])
                    rows.append((utime, x, y, h))
                except (ValueError, IndexError):
                    continue

        count = 0
        for i in range(1, len(rows)):
            utime, x, y, h = rows[i]
            p_utime, px, py, ph = rows[i - 1]
            dt = (utime - p_utime) / 1e6
            if dt <= 0 or dt > 0.5:   # skip gaps and backwards timestamps
                continue
            dx, dy = x - px, y - py
            # Body-frame: project world displacement onto current heading
            vx = ( dx * math.cos(h) + dy * math.sin(h)) / dt
            vy = (-dx * math.sin(h) + dy * math.cos(h)) / dt  # ≈0 for diff drive
            omega = angle_diff(h, ph) / dt
            self._events.append((utime, 'odom', [vx, vy, omega]))
            count += 1
        self.get_logger().info(f'  Odom: {count} velocity estimates from {os.path.basename(path)}')

    def _count(self, kind: str) -> int:
        return sum(1 for e in self._events if e[1] == kind)

    # ── playback thread ───────────────────────────────────────────────────────

    def _play(self):
        if not self._events:
            self.get_logger().error('No events to play')
            return

        sim_start_us = self._events[0][0]
        wall_start = time.monotonic()

        for utime, kind, data in self._events:
            # Duration limit
            if self._duration > 0:
                sim_elapsed_s = (utime - sim_start_us) / 1e6
                if sim_elapsed_s > self._duration:
                    break

            # Sleep until this event's wall time
            sim_offset_s = (utime - sim_start_us) / 1e6
            target_wall = wall_start + sim_offset_s / self._rate
            sleep_s = target_wall - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

            ros_time = utime_to_ros(utime)

            # Advance the simulated clock
            clk = Clock()
            clk.clock = ros_time
            self._clock_pub.publish(clk)

            if kind == 'imu':
                self._pub_imu(ros_time, data)
            elif kind == 'gps':
                self._pub_gps(ros_time, data)
            elif kind == 'odom':
                self._pub_odom(ros_time, data)

        self.get_logger().info('Playback complete.')

    # ── publishers ────────────────────────────────────────────────────────────

    def _pub_imu(self, ros_time: Time, data: list):
        roll, pitch, yaw, wx, wy, wz, ax, ay, az = data
        msg = Imu()
        msg.header.stamp    = ros_time
        msg.header.frame_id = 'imu_link'

        # Orientation from the Microstrain's onboard AHRS — useful for
        # robot_localization which fuses it. FusionCore uses wx/wy/wz + ax/ay/az.
        msg.orientation = euler_to_quat(roll, pitch, yaw)
        msg.orientation_covariance = [
            1e-4, 0.0,  0.0,
            0.0,  1e-4, 0.0,
            0.0,  0.0,  1e-3,   # yaw less accurate without GPS heading
        ]

        msg.angular_velocity.x = wx
        msg.angular_velocity.y = wy
        msg.angular_velocity.z = wz
        # Microstrain 3DM-GX3-45: angular random walk ≈ 0.07°/√hr → σ ≈ 0.003 rad/s
        msg.angular_velocity_covariance = [
            9e-6, 0.0,  0.0,
            0.0,  9e-6, 0.0,
            0.0,  0.0,  9e-6,
        ]

        msg.linear_acceleration.x = ax
        msg.linear_acceleration.y = ay
        msg.linear_acceleration.z = az  # includes gravity (+9.81 when flat)
        # Velocity random walk ≈ 0.03 m/s/√hr → σ ≈ 0.001 m/s²; use 0.1 to be safe
        msg.linear_acceleration_covariance = [
            0.01, 0.0,  0.0,
            0.0,  0.01, 0.0,
            0.0,  0.0,  0.01,
        ]
        self._imu_pub.publish(msg)

    def _pub_gps(self, ros_time: Time, data: list):
        lat, lon, alt, mode = data
        msg = NavSatFix()
        msg.header.stamp    = ros_time
        msg.header.frame_id = 'gps_link'
        msg.status.service  = NavSatStatus.SERVICE_GPS
        msg.status.status   = (NavSatStatus.STATUS_FIX
                               if mode >= 2 else NavSatStatus.STATUS_NO_FIX)
        msg.latitude  = lat
        msg.longitude = lon
        msg.altitude  = alt
        # NCLT Novatel SPAN-CPT standard GPS: ~3m CEP → σ_xy ≈ 3m → var ≈ 9m²
        msg.position_covariance = [
            9.0, 0.0,  0.0,
            0.0, 9.0,  0.0,
            0.0, 0.0, 25.0,   # vertical less accurate
        ]
        msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        self._gps_pub.publish(msg)

    def _pub_odom(self, ros_time: Time, data: list):
        vx, vy, omega = data
        msg = Odometry()
        msg.header.stamp     = ros_time
        msg.header.frame_id  = 'odom'
        msg.child_frame_id   = 'base_link'
        msg.twist.twist.linear.x  = vx
        msg.twist.twist.linear.y  = vy    # ≈ 0 for differential drive
        msg.twist.twist.angular.z = omega
        # Segway RMP encoder noise: ~2% of velocity
        msg.twist.covariance[0]  = 0.04   # vx (m/s)²
        msg.twist.covariance[7]  = 0.01   # vy
        msg.twist.covariance[35] = 0.001  # omega (rad/s)²
        self._odom_pub.publish(msg)


def main():
    rclpy.init()
    node = NCLTPlayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
