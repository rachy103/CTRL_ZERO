import math

import numpy as np
import pytest

from sim.lidar_sim import ObstacleRect, simulate_scan
from sim import parking_scenario


def _distance_at_raw(scan: np.ndarray, raw_deg: float) -> float | None:
    rows = scan[scan[:, 0] == raw_deg]
    return float(rows[0, 1]) if len(rows) else None


def test_front_obstacle_appears_at_raw_180() -> None:
    # Vehicle at origin heading +x; box 2 m ahead, near face at 1.5 m.
    scan = simulate_scan(
        0.0,
        0.0,
        0.0,
        [ObstacleRect(2.0, 0.0, 1.0, 1.0, 0.0)],
    )
    assert _distance_at_raw(scan, 180.0) == pytest.approx(1500.0)
    # Nothing behind.
    assert _distance_at_raw(scan, 0.0) is None


def test_left_obstacle_appears_at_raw_90() -> None:
    # ros = 180 - raw, so the left (+90 ros) target sits at raw 90.
    scan = simulate_scan(
        0.0,
        0.0,
        0.0,
        [ObstacleRect(0.0, 1.0, 1.0, 1.0, 0.0)],
    )
    assert _distance_at_raw(scan, 90.0) == pytest.approx(500.0)
    assert _distance_at_raw(scan, 270.0) is None


def test_scan_rotates_with_vehicle_yaw() -> None:
    # Vehicle heading +y; the same +y-side box is now dead ahead.
    scan = simulate_scan(
        0.0,
        0.0,
        math.pi / 2.0,
        [ObstacleRect(0.0, 1.0, 1.0, 1.0, 0.0)],
    )
    assert _distance_at_raw(scan, 180.0) == pytest.approx(500.0)


def test_range_limit_drops_far_hits() -> None:
    scan = simulate_scan(
        0.0,
        0.0,
        0.0,
        [ObstacleRect(20.0, 0.0, 1.0, 1.0, 0.0)],
        range_max_mm=12000.0,
    )
    assert len(scan) == 0


def test_obstacle_cases_occupy_the_specified_bays() -> None:
    case1 = parking_scenario.obstacle_rects(1)
    assert [round(r.center_x, 3) for r in case1] == [
        round(parking_scenario.bay_center_x_m(2), 3),
        round(parking_scenario.bay_center_x_m(4), 3),
    ]
    case2 = parking_scenario.obstacle_rects(2)
    assert [round(r.center_x, 3) for r in case2] == [
        round(parking_scenario.bay_center_x_m(1), 3),
        round(parking_scenario.bay_center_x_m(3), 3),
    ]


def test_target_bay_sits_between_the_obstacles() -> None:
    for case in (1, 2):
        x_min, x_max, y_min, y_max = parking_scenario.target_bay_rect(case)
        occupied = [r.center_x for r in parking_scenario.obstacle_rects(case)]
        assert occupied[0] < x_min < x_max < occupied[1]
        assert y_min < y_max


def test_start_pose_places_right_front_wheel_on_marker() -> None:
    wheelbase = 0.6
    front_track = 0.47
    for marker in (1, 2, 3, 4):
        pose = parking_scenario.start_pose(marker, wheelbase, front_track)
        marker_x, marker_y = parking_scenario.marker_position_m(marker)
        # Heading west: right front wheel = rear axle - (wheelbase, -track/2)
        # rotated by yaw=pi.
        wheel_x = pose.rear_axle_x_m - wheelbase
        wheel_y = pose.rear_axle_y_m + front_track / 2.0
        assert wheel_x == pytest.approx(marker_x)
        assert wheel_y == pytest.approx(marker_y)
        assert pose.yaw_rad == pytest.approx(math.pi)


def test_timed_plan_splits_ninety_degrees_and_stays_feasible() -> None:
    for marker in (1, 2, 3, 4):
        plan = parking_scenario.timed_plan(
            marker,
            wheelbase_m=0.603,
            front_track_m=0.47,
            max_steer_deg=18.0791788855,
            v_forward_mps=0.29,
            v_reverse_mps=0.195,
            lidar_forward_m=0.80,
        )
        assert plan["theta1_deg"] + plan["theta2_deg"] == pytest.approx(90.0)
        assert plan["approach_s"] > 0.0
        assert plan["swing_s"] > 0.0
        assert plan["arc_s"] > 0.0
    # The south column starts further from the bays, so it needs a
    # smaller swing than the north column.
    south = parking_scenario.timed_plan(
        1,
        wheelbase_m=0.603,
        front_track_m=0.47,
        max_steer_deg=18.0791788855,
        v_forward_mps=0.29,
        v_reverse_mps=0.195,
        lidar_forward_m=0.80,
    )
    north = parking_scenario.timed_plan(
        2,
        wheelbase_m=0.603,
        front_track_m=0.47,
        max_steer_deg=18.0791788855,
        v_forward_mps=0.29,
        v_reverse_mps=0.195,
        lidar_forward_m=0.80,
    )
    assert south["theta1_deg"] < north["theta1_deg"]


def test_marker_grid_spacing_matches_contest_sheet() -> None:
    # Markers are measured from the map artwork; the contest sheet says a
    # 30 cm grid, and the drawing reproduces it within its line accuracy.
    m1 = parking_scenario.marker_position_m(1)
    m2 = parking_scenario.marker_position_m(2)
    m3 = parking_scenario.marker_position_m(3)
    m4 = parking_scenario.marker_position_m(4)
    assert m3[0] - m1[0] == pytest.approx(0.30, abs=0.02)  # rows
    assert abs(m2[1] - m1[1]) == pytest.approx(0.30, abs=0.02)  # columns
    assert m1[0] == m2[0] and m3[0] == m4[0]
