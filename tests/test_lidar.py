from __future__ import annotations

import numpy as np

from ctrl_zero.lidar import LidarConfig, analyze_obstacles, angle_range


def test_angle_range_wraps_across_zero_degrees():
    scan = np.array([[350.0, 700.0], [5.0, 800.0], [90.0, 200.0]], dtype=np.float32)
    front = angle_range(scan, 330.0, 30.0)
    assert len(front) == 2


def test_obstacle_slow_zone_scales_speed():
    scan = np.array([[350.0, 700.0], [180.0, 200.0]], dtype=np.float32)
    decision = analyze_obstacles(scan, LidarConfig(stop_distance_mm=450.0, slow_distance_mm=900.0))
    assert not decision.should_stop
    assert 0.0 < decision.speed_scale < 1.0
