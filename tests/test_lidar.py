from __future__ import annotations

import numpy as np

from ctrl_zero.lidar import (
    FORWARD_RAW_ANGLE_DEG,
    LEFT_RAW_ANGLE_DEG,
    RIGHT_RAW_ANGLE_DEG,
    LidarConfig,
    analyze_obstacles,
    angle_range,
    average_distance_mm_in_ros_sector,
    direction_min_distance_mm,
    min_distance_mm_in_ros_sector,
    nearest_obstacle,
    raw_to_ros_angle_deg,
    ros_angle_range,
)


def test_angle_range_wraps_across_zero_degrees():
    scan = np.array([[350.0, 700.0], [5.0, 800.0], [90.0, 200.0]], dtype=np.float32)
    front = angle_range(scan, 330.0, 30.0)
    assert len(front) == 2


def test_obstacle_slow_zone_scales_speed():
    scan = np.array([[90.0, 700.0], [180.0, 200.0]], dtype=np.float32)
    decision = analyze_obstacles(scan, LidarConfig(stop_distance_mm=450.0, slow_distance_mm=900.0))
    assert not decision.should_stop
    assert 0.0 < decision.speed_scale < 1.0


def test_ros_sector_conversion_matches_original_rplidar_driver_orientation():
    scan = np.array(
        [
            [90.0, 800.0],     # raw 90 -> ROS +90 (source lane 2 sector)
            [271.5, 900.0],    # raw 271.5 -> ROS -91.5 (source lane 1 sector)
            [180.0, 2000.0],   # raw 180 -> ROS 0
        ],
        dtype=np.float32,
    )

    lane2 = ros_angle_range(scan, 87.5, 92.5, raw_angle_for_ros_zero_deg=180.0)
    lane1 = ros_angle_range(scan, -93.0, -90.0, raw_angle_for_ros_zero_deg=180.0)

    assert lane2[:, 0].tolist() == [90.0]
    assert lane1[:, 0].tolist() == [271.5]
    assert average_distance_mm_in_ros_sector(scan, 87.5, 92.5) == 800.0
    assert average_distance_mm_in_ros_sector(scan, -93.0, -90.0) == 900.0


def test_ros_sector_average_rejects_zero_and_infinite_ranges_like_source():
    scan = np.array([[90.0, 0.0], [91.0, np.inf], [92.0, 750.0]], dtype=np.float32)

    assert average_distance_mm_in_ros_sector(scan, 87.5, 92.5) == 750.0


def test_vehicle_direction_convention_maps_to_ros():
    # Confirmed on the car: forward=raw 180, left=raw 90, right=raw 270.
    assert raw_to_ros_angle_deg(FORWARD_RAW_ANGLE_DEG) == 0.0
    assert raw_to_ros_angle_deg(LEFT_RAW_ANGLE_DEG) == 90.0
    assert raw_to_ros_angle_deg(RIGHT_RAW_ANGLE_DEG) == -90.0


def test_nearest_obstacle_reports_closest_valid_point():
    scan = np.array([[180.0, 5000.0], [90.0, 500.0], [270.0, 0.0]], dtype=np.float32)
    nearest = nearest_obstacle(scan)
    assert nearest is not None
    assert nearest.raw_angle_deg == 90.0
    assert nearest.ros_angle_deg == 90.0  # left
    assert nearest.distance_mm == 500.0


def test_nearest_obstacle_honours_max_range_and_empty_scan():
    scan = np.array([[180.0, 5000.0]], dtype=np.float32)
    assert nearest_obstacle(scan, max_range_mm=1000.0) is None
    assert nearest_obstacle(None) is None


def test_direction_min_distance_selects_forward_sector():
    scan = np.array([[178.0, 600.0], [182.0, 400.0], [90.0, 100.0]], dtype=np.float32)
    # Forward sector ignores the near point at raw 90 (left).
    assert direction_min_distance_mm(scan, FORWARD_RAW_ANGLE_DEG, 15.0) == 400.0


def test_min_distance_in_ros_sector_uses_minimum_not_mean():
    scan = np.array([[180.0, 400.0], [180.5, 5000.0]], dtype=np.float32)  # ROS ~0 (forward)
    assert min_distance_mm_in_ros_sector(scan, -2.5, 2.5) == 400.0
