"""Simulated 2D LiDAR for the parking mission.

Produces scans in the CTRL_ZERO format: a numpy array of
``[raw_angle_deg, distance_mm]`` rows using the confirmed vehicle
convention (front = raw 180, ros = 180 - raw, so ros +90 is left).
Rays are cast in the world plane against oriented rectangles
(obstacle car footprints). Only returns rows for rays that hit,
matching a real scan where misses are dropped as invalid points.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObstacleRect:
    """Axis of ``yaw_rad`` oriented rectangle footprint in world meters."""

    center_x: float
    center_y: float
    length_m: float  # along the rectangle's yaw axis
    width_m: float
    yaw_rad: float


def _ray_rect_distance(
    origin_x: float,
    origin_y: float,
    direction_x: float,
    direction_y: float,
    rect: ObstacleRect,
) -> float | None:
    """Distance (m) from origin along the ray to the rectangle, or None."""
    cos_yaw = math.cos(rect.yaw_rad)
    sin_yaw = math.sin(rect.yaw_rad)
    # Transform the ray into the rectangle's local frame.
    rel_x = origin_x - rect.center_x
    rel_y = origin_y - rect.center_y
    local_ox = cos_yaw * rel_x + sin_yaw * rel_y
    local_oy = -sin_yaw * rel_x + cos_yaw * rel_y
    local_dx = cos_yaw * direction_x + sin_yaw * direction_y
    local_dy = -sin_yaw * direction_x + cos_yaw * direction_y

    half_l = rect.length_m / 2.0
    half_w = rect.width_m / 2.0

    # Slab method against the axis-aligned local box.
    t_min = 0.0
    t_max = math.inf
    for local_o, local_d, half in (
        (local_ox, local_dx, half_l),
        (local_oy, local_dy, half_w),
    ):
        if abs(local_d) < 1e-12:
            if abs(local_o) > half:
                return None
            continue
        t1 = (-half - local_o) / local_d
        t2 = (half - local_o) / local_d
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return None
    if t_max < 0.0:
        return None
    return t_min if t_min > 0.0 else t_max


def simulate_scan(
    lidar_x: float,
    lidar_y: float,
    vehicle_yaw_rad: float,
    obstacles: list[ObstacleRect],
    *,
    raw_angle_for_ros_zero_deg: float = 180.0,
    angular_step_deg: float = 1.0,
    range_max_mm: float = 12000.0,
) -> np.ndarray:
    """Cast one full revolution and return ``[raw_angle_deg, distance_mm]``."""
    rows: list[tuple[float, float]] = []
    raw = 0.0
    while raw < 360.0:
        ros_deg = raw_angle_for_ros_zero_deg - raw
        world_angle = vehicle_yaw_rad + math.radians(ros_deg)
        direction_x = math.cos(world_angle)
        direction_y = math.sin(world_angle)
        best: float | None = None
        for rect in obstacles:
            hit = _ray_rect_distance(
                lidar_x, lidar_y, direction_x, direction_y, rect
            )
            if hit is not None and (best is None or hit < best):
                best = hit
        if best is not None:
            distance_mm = best * 1000.0
            if 0.0 < distance_mm <= range_max_mm:
                rows.append((raw, distance_mm))
        raw += angular_step_deg
    if not rows:
        return np.empty((0, 2), dtype=np.float64)
    return np.array(rows, dtype=np.float64)
