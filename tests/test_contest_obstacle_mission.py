from __future__ import annotations

import numpy as np

from ctrl_zero.control import DriveCommand
from ctrl_zero.obstacles import (
    ContestObstacleMission,
    ContestObstacleMissionConfig,
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


def forward_scan(distance_mm: float) -> np.ndarray:
    return np.array([[180.0, distance_mm], [90.0, 3000.0]], dtype=np.float32)


def step(
    mission: ContestObstacleMission,
    *,
    now_s: float,
    scan: np.ndarray | None,
    objects=(),
    cruise_speed: int | None = None,
    lane: int | None = 2,
):
    return mission.step(
        objects=objects,
        current_lane=f"lane{lane}" if lane is not None else None,
        frame_width=FRAME_WIDTH,
        lidar_scan=scan,
        cruise_speed=cruise_speed,
        now_s=now_s,
    )


def timed_config(**overrides) -> ContestObstacleMissionConfig:
    values = {
        "trigger_frames": 1,
        "lane2_shift_steer": -60,
        "lane1_shift_steer": 60,
        "shift_out_duration_s": 0.70,
        "pass_steer": 0,
        "pass_duration_s": 0.50,
        "retrigger_cooldown_s": 1.0,
    }
    values.update(overrides)
    return ContestObstacleMissionConfig(**values)


def test_car_detection_uses_center_region_and_bbox_area():
    centered = car_with_area(6000.0)
    observation = detect_contest_car([centered], FRAME_WIDTH)

    assert observation.detected
    assert observation.area == 6000.0

    outside_first = car_with_area(6000.0, center_x=100.0)
    observation = detect_contest_car([outside_first, centered], FRAME_WIDTH)

    assert observation.detected
    assert observation.obj is centered


def test_lane_follow_remains_separate_until_both_sensors_trigger():
    mission = ContestObstacleMission(ContestObstacleMissionConfig(trigger_frames=5))
    base = DriveCommand(steer=13, speed=255, reason="contest")
    car = [car_with_area(6000.0)]

    for frame in range(4):
        mission_command = step(mission, now_s=frame * 0.1, scan=forward_scan(900.0), objects=car)
        assert mission_command is None
        assert apply_obstacle_mission_override(base, mission_command, SafetyDecision.clear()) is base

    mission_command = step(mission, now_s=0.4, scan=forward_scan(900.0), objects=car)

    assert mission.phase == ContestObstaclePhase.AVOID_SHIFT_OUT
    assert mission_command == DriveCommand(-60, 255, "obstacle_avoid_shift_out")


def test_trigger_counter_requires_consecutive_camera_and_lidar_agreement():
    mission = ContestObstacleMission(ContestObstacleMissionConfig(trigger_frames=3))
    car = [car_with_area(6000.0)]

    assert step(mission, now_s=0.0, scan=forward_scan(900.0), objects=car) is None
    assert step(mission, now_s=0.1, scan=forward_scan(900.0), objects=car) is None
    assert mission.trigger_counter == 2

    assert step(mission, now_s=0.2, scan=forward_scan(900.0), objects=[]) is None
    assert mission.trigger_counter == 0

    for now_s in (0.3, 0.4):
        assert step(mission, now_s=now_s, scan=forward_scan(900.0), objects=car) is None
    command = step(mission, now_s=0.5, scan=forward_scan(900.0), objects=car)

    assert command is not None
    assert mission.phase == ContestObstaclePhase.AVOID_SHIFT_OUT


def test_lane2_avoidance_shifts_left_passes_and_has_no_return_steering():
    mission = ContestObstacleMission(timed_config(retrigger_cooldown_s=0.0))
    car = [car_with_area(6000.0)]

    commands = [step(mission, now_s=10.0, scan=forward_scan(900.0), objects=car)]
    commands.append(step(mission, now_s=10.69, scan=None, objects=[]))
    commands.append(step(mission, now_s=10.71, scan=None, objects=[]))
    commands.append(step(mission, now_s=11.19, scan=None, objects=[]))

    assert [command.reason for command in commands if command is not None] == [
        "obstacle_avoid_shift_out",
        "obstacle_avoid_shift_out",
        "obstacle_avoid_pass",
        "obstacle_avoid_pass",
    ]
    assert [command.steer for command in commands if command is not None] == [-60, -60, 0, 0]

    finished = step(mission, now_s=11.21, scan=None, objects=[])

    assert finished is None
    assert mission.phase == ContestObstaclePhase.LANE_FOLLOW
    assert not mission.active

    base = DriveCommand(steer=-12, speed=255, reason="contest")
    assert apply_obstacle_mission_override(base, finished, SafetyDecision.clear()) is base


def test_lane1_avoidance_shifts_right_and_locks_direction_until_pass():
    mission = ContestObstacleMission(timed_config())
    car = [car_with_area(6000.0)]

    start = step(mission, now_s=0.0, scan=forward_scan(900.0), objects=car, lane=1)
    # The detector may report lane 2 mid-maneuver, but the direction selected
    # from the source lane must remain locked until the pass phase.
    active = step(mission, now_s=0.3, scan=None, objects=[], lane=2)

    assert start == DriveCommand(60, 255, "obstacle_avoid_shift_out")
    assert active == DriveCommand(60, 255, "obstacle_avoid_shift_out")
    assert mission.avoidance_source_lane == 1


def test_avoidance_ignores_new_detections_and_keeps_runtime_max_speed():
    mission = ContestObstacleMission(timed_config())
    car = [car_with_area(6000.0)]

    start = step(mission, now_s=0.0, scan=forward_scan(900.0), objects=car, cruise_speed=235)
    large_car = [car_with_area(20000.0)]
    active = step(mission, now_s=0.3, scan=forward_scan(100.0), objects=large_car, cruise_speed=235)

    assert start == DriveCommand(-60, 235, "obstacle_avoid_shift_out")
    assert active == DriveCommand(-60, 235, "obstacle_avoid_shift_out")


def test_next_obstacle_can_trigger_after_timed_mode_and_cooldown():
    mission = ContestObstacleMission(
        timed_config(shift_out_duration_s=0.10, pass_duration_s=0.10, retrigger_cooldown_s=1.0)
    )
    car = [car_with_area(6000.0)]

    assert step(mission, now_s=0.0, scan=forward_scan(900.0), objects=car) is not None
    assert step(mission, now_s=0.1, scan=None, objects=[]) is not None
    assert step(mission, now_s=0.21, scan=None, objects=[]) is None
    assert mission.phase == ContestObstaclePhase.LANE_FOLLOW

    assert step(mission, now_s=1.20, scan=forward_scan(900.0), objects=car, lane=1) is None
    second = step(mission, now_s=1.21, scan=forward_scan(900.0), objects=car, lane=1)

    assert second == DriveCommand(60, 255, "obstacle_avoid_shift_out")
    assert mission.phase == ContestObstaclePhase.AVOID_SHIFT_OUT


def test_traffic_light_stop_is_preserved_during_avoidance_override():
    base_command = DriveCommand(12, 0, "traffic_light_stop")
    mission_command = DriveCommand(-60, 255, "obstacle_avoid_shift_out")
    traffic_stop = SafetyDecision(speed_scale=0.0, should_stop=True, reason="traffic_light_stop")

    command = apply_obstacle_mission_override(base_command, mission_command, traffic_stop)

    assert command is base_command
