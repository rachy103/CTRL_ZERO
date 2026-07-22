from __future__ import annotations

import numpy as np

from ctrl_zero.control import DriveCommand
from ctrl_zero.obstacles import (
    ContestObstacleMission,
    ContestObstaclePhase,
    detect_contest_car,
)
from ctrl_zero.perception import BoundingBox, DetectedObject
from ctrl_zero.safety import SafetyDecision
from main import apply_obstacle_mission_override


FRAME_WIDTH = 640


def car_with_area(area: float, center_x: float = 320.0) -> DetectedObject:
    width = 50.0
    height = area / width
    return DetectedObject(
        class_name="car",
        confidence=0.90,
        bbox=BoundingBox(
            center_x - width / 2.0,
            20.0,
            center_x + width / 2.0,
            20.0 + height,
        ),
    )


def lane2_scan(distance_mm: float) -> np.ndarray:
    # The obstacle is a car directly ahead: forward is raw 180 = ROS 0 on this
    # vehicle, with far background off to the side.
    return np.array([[180.0, distance_mm], [90.0, 3000.0]], dtype=np.float32)


def lane1_scan(distance_mm: float) -> np.ndarray:
    # Lane 1 monitoring also watches the forward cone (raw 180 = ROS 0).
    return np.array([[180.0, distance_mm], [90.0, 3000.0]], dtype=np.float32)


def step(
    mission: ContestObstacleMission,
    *,
    lane: int | None,
    scan: np.ndarray | None,
    objects=(),
    heading_deg: float = 0.0,
    vehicle_position_x: float = 0.0,
):
    return mission.step(
        objects=objects,
        current_lane=f"lane{lane}" if lane is not None else None,
        heading_deg=heading_deg,
        vehicle_position_x=vehicle_position_x,
        frame_width=FRAME_WIDTH,
        lidar_scan=scan,
    )


def enter_lane1_follow(mission: ContestObstacleMission) -> None:
    for _ in range(5):
        step(mission, lane=2, scan=lane2_scan(900.0), objects=[car_with_area(6000.0)])
    assert mission.phase == ContestObstaclePhase.CHANGE_TO_LANE1

    completion_command = step(mission, lane=1, scan=lane1_scan(900.0), vehicle_position_x=30.0)
    assert completion_command == DriveCommand(-80, 255, "obstacle_lane_change_2_to_1")
    assert mission.phase == ContestObstaclePhase.STABILIZE_LANE1

    for _ in range(3):
        stabilization_command = step(mission, lane=1, scan=lane1_scan(1400.0))
        assert stabilization_command.speed == 255
    assert mission.phase == ContestObstaclePhase.FOLLOW_LANE1


def test_car_detection_uses_center_region_and_bbox_area():
    centered = car_with_area(6000.0)
    observation = detect_contest_car([centered], FRAME_WIDTH)

    assert observation.detected
    assert observation.area == 6000.0

    outside_first = car_with_area(6000.0, center_x=100.0)
    observation = detect_contest_car([outside_first, centered], FRAME_WIDTH)

    assert observation.detected
    assert observation.obj is centered
    assert observation.area == 6000.0


def test_lane2_avoidance_starts_only_after_five_camera_and_lidar_detections():
    mission = ContestObstacleMission()
    base = DriveCommand(steer=13, speed=255, reason="contest")
    clear_safety = SafetyDecision.clear()
    forward_car = [car_with_area(6000.0)]

    for _ in range(4):
        mission_command = step(mission, lane=2, scan=lane2_scan(900.0), objects=forward_car)
        assert mission_command is None
        assert apply_obstacle_mission_override(base, mission_command, clear_safety) is base

    mission_command = step(mission, lane=2, scan=lane2_scan(900.0), objects=forward_car)

    assert mission.lidar_lane_change_executed
    assert mission.target_lane == 1
    assert mission.phase == ContestObstaclePhase.CHANGE_TO_LANE1
    assert mission_command == DriveCommand(-80, 255, "obstacle_lane_change_2_to_1")

    # Detection changes during the maneuver do not change its locked target.
    ignored = step(
        mission,
        lane=2,
        scan=lane2_scan(500.0),
        objects=[car_with_area(20000.0)],
    )
    assert ignored == DriveCommand(-80, 255, "obstacle_lane_change_2_to_1")
    assert mission.phase == ContestObstaclePhase.CHANGE_TO_LANE1


def test_lane2_detection_counter_requires_consecutive_sensor_agreement():
    mission = ContestObstacleMission()
    forward_car = [car_with_area(6000.0)]

    for _ in range(4):
        assert step(mission, lane=2, scan=lane2_scan(900.0), objects=forward_car) is None
    assert mission.lidar_lane_change_counter == 4

    # Losing either the camera or LiDAR condition resets the shared counter.
    assert step(mission, lane=2, scan=lane2_scan(900.0), objects=[]) is None
    assert mission.lidar_lane_change_counter == 0

    for _ in range(4):
        assert step(mission, lane=2, scan=lane2_scan(900.0), objects=forward_car) is None
    assert mission.phase == ContestObstaclePhase.MONITOR_LANE2

    assert step(mission, lane=2, scan=lane2_scan(900.0), objects=forward_car).speed == 255
    assert mission.phase == ContestObstaclePhase.CHANGE_TO_LANE1


def test_camera_area_never_slows_stops_or_reverses_during_avoidance():
    mission = ContestObstacleMission()
    enter_lane1_follow(mission)

    no_car = step(mission, lane=1, scan=None, heading_deg=20.0)
    small = step(mission, lane=1, scan=None, objects=[car_with_area(4999.0)], heading_deg=20.0)
    medium_lower = step(mission, lane=1, scan=None, objects=[car_with_area(5000.0)], heading_deg=20.0)
    medium_upper = step(mission, lane=1, scan=None, objects=[car_with_area(11999.0)], heading_deg=20.0)
    large_lower = step(mission, lane=1, scan=None, objects=[car_with_area(12000.0)], heading_deg=20.0)
    reverse = step(mission, lane=1, scan=None, objects=[car_with_area(15000.0)], heading_deg=20.0)

    commands = (no_car, small, medium_lower, medium_upper, large_lower, reverse)
    assert all(command.speed == 255 for command in commands)
    assert all(command.reason == "obstacle_lane1_follow" for command in commands)
    assert all(command.steer == medium_lower.steer for command in commands)


def test_obstacle_pass_return_to_lane2_rearms_for_next_obstacle():
    mission = ContestObstacleMission()
    enter_lane1_follow(mission)
    forward_car = [car_with_area(6000.0)]

    for _ in range(3):
        command = step(mission, lane=1, scan=lane1_scan(900.0), objects=forward_car)
        assert command.speed == 255
    assert mission.obstacle_flag

    for _ in range(3):
        command = step(mission, lane=1, scan=lane1_scan(1200.0), objects=forward_car)
        assert command.speed == 255

    assert mission.phase == ContestObstaclePhase.CHANGE_TO_LANE2
    assert mission.target_lane == 2

    return_command = step(mission, lane=1, scan=lane1_scan(1200.0), objects=[car_with_area(20000.0)])
    assert return_command == DriveCommand(56, 255, "obstacle_lane_change_1_to_2")

    completion_command = step(mission, lane=2, scan=lane2_scan(1200.0), vehicle_position_x=-30.0)
    assert completion_command == DriveCommand(56, 255, "obstacle_lane_change_1_to_2")
    assert mission.phase == ContestObstaclePhase.STABILIZE_LANE2

    for _ in range(3):
        stabilization_command = step(mission, lane=2, scan=lane2_scan(1200.0))
        assert stabilization_command.speed == 255
    assert mission.phase == ContestObstaclePhase.MONITOR_LANE2
    assert not mission.active
    assert not mission.lidar_lane_change_executed

    # While monitoring with no camera detection, main returns the base command.
    mission_command = step(mission, lane=2, scan=lane2_scan(500.0))
    base = DriveCommand(steer=-7, speed=255, reason="contest")
    restored = apply_obstacle_mission_override(base, mission_command, SafetyDecision.clear())

    assert mission_command is None
    assert restored is base

    # A second obstacle can trigger another complete avoidance cycle even if
    # its car detection briefly obscures the lane label.
    for _ in range(4):
        assert step(mission, lane=None, scan=lane2_scan(900.0), objects=forward_car) is None
    second_avoidance = step(mission, lane=None, scan=lane2_scan(900.0), objects=forward_car)

    assert second_avoidance == DriveCommand(-80, 255, "obstacle_lane_change_2_to_1")
    assert mission.phase == ContestObstaclePhase.CHANGE_TO_LANE1


def test_source_clear_counter_is_not_reset_by_near_sample_while_forward():
    mission = ContestObstacleMission()
    enter_lane1_follow(mission)
    forward_car = [car_with_area(6000.0)]

    for _ in range(3):
        step(mission, lane=1, scan=lane1_scan(900.0), objects=forward_car)
    assert mission.obstacle_flag

    step(mission, lane=1, scan=lane1_scan(1200.0), objects=forward_car)
    assert mission.no_obstacle_counter == 1

    step(mission, lane=1, scan=lane1_scan(900.0), objects=forward_car)
    assert mission.no_obstacle_counter == 1

    step(mission, lane=1, scan=lane1_scan(1200.0), objects=forward_car)
    step(mission, lane=1, scan=lane1_scan(1200.0), objects=forward_car)
    assert mission.phase == ContestObstaclePhase.CHANGE_TO_LANE2


def test_traffic_light_stop_is_preserved_during_mission_override():
    base_command = DriveCommand(12, 0, "traffic_light_stop")
    mission_command = DriveCommand(-80, 255, "obstacle_lane_change_2_to_1")
    traffic_stop = SafetyDecision(speed_scale=0.0, should_stop=True, reason="traffic_light_stop")

    command = apply_obstacle_mission_override(
        base_command,
        mission_command,
        traffic_stop,
    )

    assert command is base_command
