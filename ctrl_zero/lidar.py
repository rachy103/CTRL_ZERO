from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ctrl_zero.common import clamp


@dataclass
class LidarConfig:
    port: str | None = None
    scan_type: str = "normal"
    max_buffer_size: int = 3000
    sample_rate: int = 10
    min_distance_mm: float = 0.0
    front_min_angle_deg: float = 87.5
    front_max_angle_deg: float = 92.5
    stop_distance_mm: float = 450.0
    slow_distance_mm: float = 900.0
    min_speed_scale: float = 0.35
    rpm: int | None = None


@dataclass
class ObstacleDecision:
    nearest_front_mm: float | None
    speed_scale: float
    should_stop: bool
    front_points: int

    @classmethod
    def clear(cls) -> "ObstacleDecision":
        return cls(nearest_front_mm=None, speed_scale=1.0, should_stop=False, front_points=0)


class LidarReader:
    def __init__(self, config: LidarConfig):
        self.config = config
        self.lidar = None
        self._scan_iter: Iterator[np.ndarray] | None = None

    def open(self) -> None:
        if not self.config.port:
            raise RuntimeError("LIDAR_PORT is empty. Set it in main.py before USE_LIDAR=True.")
        try:
            from rplidar import RPLidar
        except ModuleNotFoundError as exc:
            raise RuntimeError("rplidar-roboticia is required for LiDAR. Install requirements.txt first.") from exc

        self.lidar = RPLidar(self.config.port)
        if self.config.rpm is not None:
            self.lidar.motor_speed = self.config.rpm
        print(f"LiDAR info: {self.lidar.get_info()}")
        print(f"LiDAR health: {self.lidar.get_health()}")
        self._scan_iter = self._scan_generator()

    def read_scan(self) -> np.ndarray | None:
        if self._scan_iter is None:
            return None
        return next(self._scan_iter)

    def close(self) -> None:
        if self.lidar is not None:
            self.lidar.stop()
            self.lidar.stop_motor()
            self.lidar.disconnect()
            self.lidar = None

    def _scan_generator(self) -> Iterator[np.ndarray]:
        if self.lidar is None:
            raise RuntimeError("LiDAR is not open.")
        scan_list = []
        iterator = self.lidar.iter_measures(self.config.scan_type, self.config.max_buffer_size)
        for new_scan, quality, angle, distance in iterator:
            if new_scan:
                if len(scan_list) > self.config.sample_rate:
                    data = np.array(scan_list, dtype=np.float32)
                    yield data[:, 1:]
                scan_list = []
            if distance > self.config.min_distance_mm:
                scan_list.append((quality, angle, distance))


def angle_range(scan: np.ndarray, min_angle_deg: float, max_angle_deg: float) -> np.ndarray:
    if scan.size == 0:
        return scan
    angles = scan[:, 0] % 360.0
    return scan[_angle_mask(angles, min_angle_deg, max_angle_deg)]


def ros_angle_range(
    scan: np.ndarray,
    min_ros_angle_deg: float,
    max_ros_angle_deg: float,
    raw_angle_for_ros_zero_deg: float = 180.0,
) -> np.ndarray:
    """Select a ROS-style angular sector from raw RPLidar measurements.

    ``rplidar-roboticia`` returns the device's raw clockwise heading.  With the
    original contest launch settings (RPLidar ROS ``inverted=False`` and zero
    TF yaw), the official driver publishes ``ros_angle = 180 - raw_angle``.
    ``raw_angle_for_ros_zero_deg`` keeps the same physical sector configurable
    when the sensor is mounted with a different yaw on the Windows vehicle.
    """
    if scan.size == 0:
        return scan
    ros_angles = (raw_angle_for_ros_zero_deg - scan[:, 0]) % 360.0
    return scan[_angle_mask(ros_angles, min_ros_angle_deg, max_ros_angle_deg)]


def average_distance_mm_in_ros_sector(
    scan: np.ndarray | None,
    min_ros_angle_deg: float,
    max_ros_angle_deg: float,
    raw_angle_for_ros_zero_deg: float = 180.0,
) -> float | None:
    """Return the source-compatible mean range for one ROS angular sector."""
    if scan is None or len(scan) == 0:
        return None
    sector = ros_angle_range(
        scan,
        min_ros_angle_deg,
        max_ros_angle_deg,
        raw_angle_for_ros_zero_deg,
    )
    if len(sector) == 0:
        return None
    distances = sector[:, 1]
    valid = distances[np.isfinite(distances) & (distances > 0.0)]
    if len(valid) == 0:
        return None
    return float(np.mean(valid))


def _angle_mask(angles: np.ndarray, min_angle_deg: float, max_angle_deg: float) -> np.ndarray:
    min_angle = min_angle_deg % 360.0
    max_angle = max_angle_deg % 360.0
    if min_angle <= max_angle:
        return (angles >= min_angle) & (angles <= max_angle)
    return (angles >= min_angle) | (angles <= max_angle)


def analyze_obstacles(scan: np.ndarray | None, config: LidarConfig) -> ObstacleDecision:
    if scan is None or len(scan) == 0:
        return ObstacleDecision.clear()

    front = angle_range(scan, config.front_min_angle_deg, config.front_max_angle_deg)
    if len(front) == 0:
        return ObstacleDecision.clear()

    nearest = float(np.min(front[:, 1]))
    if nearest <= config.stop_distance_mm:
        return ObstacleDecision(nearest_front_mm=nearest, speed_scale=0.0, should_stop=True, front_points=len(front))
    if nearest >= config.slow_distance_mm:
        return ObstacleDecision(nearest_front_mm=nearest, speed_scale=1.0, should_stop=False, front_points=len(front))

    span = max(config.slow_distance_mm - config.stop_distance_mm, 1.0)
    ratio = (nearest - config.stop_distance_mm) / span
    speed_scale = config.min_speed_scale + ratio * (1.0 - config.min_speed_scale)
    return ObstacleDecision(
        nearest_front_mm=nearest,
        speed_scale=float(clamp(speed_scale, config.min_speed_scale, 1.0)),
        should_stop=False,
        front_points=len(front),
    )
