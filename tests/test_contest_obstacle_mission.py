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
    heading_deg: float | None = None,
    lane_follow_steer: int = 0,
):
    return mission.step(
        objects=objects,
        current_lane=f"lane{lane}" if lane is not None else None,
        frame_width=FRAME_WIDTH,
        lidar_scan=scan,
        heading_deg=heading_deg,
        lane_follow_steer=lane_follow_steer,
        cruise_speed=cruise_speed,
        now_s=now_s,
    )


def timed_config(**overrides) -> ContestObstacleMissionConfig:
    values = {
        "trigger_frames": 1,
        "lane2_shift_steer": -60,
        "lane1_shift_steer": 60,
        "shift_out_duration_s": 0.70,
        "lane2_pass_steer": 60,
        "lane1_pass_steer": -60,
        "pass_align_heading_deg": 5.0,
        "pass_max_duration_s": 0.50,
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


def test_lane2_avoidance_shifts_left_then_counter_steers_right_on_pass():
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
    # Shift left (-60), then counter-steer right (+60) during the pass.
    assert [command.steer for command in commands if command is not None] == [-60, -60, 60, 60]

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


def test_lane1_avoidance_counter_steers_left_on_pass():
    mission = ContestObstacleMission(timed_config(retrigger_cooldown_s=0.0))
    car = [car_with_area(6000.0)]

    shift = step(mission, now_s=0.0, scan=forward_scan(900.0), objects=car, lane=1)
    # Enter the pass phase (after shift_out_duration_s = 0.70).
    passing = step(mission, now_s=0.71, scan=None, objects=[], lane=1)

    assert shift == DriveCommand(60, 255, "obstacle_avoid_shift_out")
    # Shift right (+60), then counter-steer left (-60) during the pass.
    assert passing == DriveCommand(-60, 255, "obstacle_avoid_pass")


def test_pass_ends_when_heading_aligns_with_new_lane_before_timeout():
    # Long safety cap so the closed-loop heading condition, not the timeout, ends
    # the pass.  Source lane 2 -> target lane 1.
    mission = ContestObstacleMission(
        timed_config(pass_max_duration_s=5.0, pass_align_heading_deg=5.0, retrigger_cooldown_s=0.0)
    )
    car = [car_with_area(6000.0)]

    step(mission, now_s=0.0, scan=forward_scan(900.0), objects=car, lane=2)

    # In the new lane (1) but still angled -> keep counter-steering.
    still_angled = step(mission, now_s=0.71, scan=None, objects=[], lane=1, heading_deg=20.0)
    assert still_angled == DriveCommand(60, 255, "obstacle_avoid_pass")
    assert mission.phase == ContestObstaclePhase.AVOID_PASS

    # Heading now matches the new lane centerline -> finish well before the 5s cap.
    aligned = step(mission, now_s=0.9, scan=None, objects=[], lane=1, heading_deg=2.0)
    assert aligned is None
    assert mission.phase == ContestObstaclePhase.LANE_FOLLOW


def test_pass_does_not_end_on_alignment_while_still_in_source_lane():
    # Heading is 0 but we are still detected in the source lane -> not aligned
    # with the *new* lane yet, so counter-steering continues.
    mission = ContestObstacleMission(
        timed_config(pass_max_duration_s=5.0, pass_align_heading_deg=5.0, retrigger_cooldown_s=0.0)
    )
    car = [car_with_area(6000.0)]

    step(mission, now_s=0.0, scan=forward_scan(900.0), objects=car, lane=2)
    still_passing = step(mission, now_s=0.71, scan=None, objects=[], lane=2, heading_deg=0.0)

    assert still_passing == DriveCommand(60, 255, "obstacle_avoid_pass")
    assert mission.phase == ContestObstaclePhase.AVOID_PASS


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
        timed_config(shift_out_duration_s=0.10, pass_max_duration_s=0.10, retrigger_cooldown_s=1.0)
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


def test_shift_out_duration_grows_with_steering_at_detection():
    # base 0.10s + |steer| * 0.01 = 0.10 + 80*0.01 = 0.90s of shift-out.
    mission = ContestObstacleMission(
        timed_config(shift_out_duration_s=0.10, shift_out_steer_weight=0.01, retrigger_cooldown_s=0.0)
    )
    car = [car_with_area(6000.0)]

    start = step(mission, now_s=0.0, scan=forward_scan(900.0), objects=car, lane=2, lane_follow_steer=-80)
    assert start.reason == "obstacle_avoid_shift_out"

    # Still shifting at 0.89s (would already be in the pass phase without the weight).
    mid = step(mission, now_s=0.89, scan=None, objects=[], lane=2)
    assert mid.reason == "obstacle_avoid_shift_out"

    # After 0.90s the shift-out ends and the pass (counter-steer) begins.
    after = step(mission, now_s=0.91, scan=None, objects=[], lane=2)
    assert after.reason == "obstacle_avoid_pass"


def test_avoidance_steer_ignores_max_steer_and_follows_tuned_value():
    # Tuned pass steer beyond the usual MAX_STEER (80) is emitted verbatim.
    mission = ContestObstacleMission(
        timed_config(lane2_pass_steer=100, pass_max_duration_s=0.50, retrigger_cooldown_s=0.0)
    )
    car = [car_with_area(6000.0)]

    step(mission, now_s=0.0, scan=forward_scan(900.0), objects=car, lane=2)
    passing = step(mission, now_s=0.71, scan=None, objects=[], lane=2)

    assert passing == DriveCommand(100, 255, "obstacle_avoid_pass")


def test_pass_completion_rechains_shift_out_when_obstacle_still_ahead():
    # No lane-follow gap between consecutive obstacles: when the pass completes
    # with a car still ahead, avoidance chains straight into a new shift-out.
    mission = ContestObstacleMission(
        timed_config(pass_max_duration_s=0.50, retrigger_cooldown_s=0.0)
    )
    car = [car_with_area(6000.0)]

    step(mission, now_s=0.0, scan=forward_scan(900.0), objects=car, lane=2)
    # Shift-out (0.70) + pass cap (0.50) elapsed, and a car is still ahead in the
    # new lane (1) -> immediately re-shift from lane 1 (steer +60), no "contest".
    rechained = step(mission, now_s=1.21, scan=forward_scan(900.0), objects=car, lane=1)

    assert rechained == DriveCommand(60, 255, "obstacle_avoid_shift_out")
    assert mission.phase == ContestObstaclePhase.AVOID_SHIFT_OUT
    assert mission.avoidance_source_lane == 1


def test_pass_completion_returns_to_lane_follow_when_clear():
    mission = ContestObstacleMission(
        timed_config(pass_max_duration_s=0.50, retrigger_cooldown_s=0.0)
    )
    car = [car_with_area(6000.0)]

    step(mission, now_s=0.0, scan=forward_scan(900.0), objects=car, lane=2)
    # Pass completes with nothing ahead -> hand back to the lane follower.
    cleared = step(mission, now_s=1.21, scan=None, objects=[], lane=1)

    assert cleared is None
    assert mission.phase == ContestObstaclePhase.LANE_FOLLOW


def test_traffic_light_stop_is_preserved_during_avoidance_override():
    base_command = DriveCommand(12, 0, "traffic_light_stop")
    mission_command = DriveCommand(-60, 255, "obstacle_avoid_shift_out")
    traffic_stop = SafetyDecision(speed_scale=0.0, should_stop=True, reason="traffic_light_stop")

    command = apply_obstacle_mission_override(base_command, mission_command, traffic_stop)

    assert command is base_command
