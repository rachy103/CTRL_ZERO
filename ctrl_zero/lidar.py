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
    front_min_angle_deg: float = 330.0
    front_max_angle_deg: float = 30.0
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
    min_angle = min_angle_deg % 360.0
    max_angle = max_angle_deg % 360.0
    if min_angle <= max_angle:
        mask = (angles >= min_angle) & (angles <= max_angle)
    else:
        mask = (angles >= min_angle) | (angles <= max_angle)
    return scan[mask]


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
