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
FRAME_HEIGHT = 480


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
    offset_norm: float | None = None,
    lane_follow_steer: int = 0,
):
    return mission.step(
        objects=objects,
        current_lane=f"lane{lane}" if lane is not None else None,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        lidar_scan=scan,
        heading_deg=heading_deg,
        offset_norm=offset_norm,
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


def test_lane2_avoidance_shifts_left_then_returns_to_lane_follow():
    mission = ContestObstacleMission(timed_config(retrigger_cooldown_s=0.0))
    car = [car_with_area(6000.0)]

    start = step(mission, now_s=10.0, scan=forward_scan(900.0), objects=car)
    mid = step(mission, now_s=10.69, scan=None, objects=[])
    finished = step(mission, now_s=10.71, scan=None, objects=[])

    # Shift left (-60) for the whole shift-out window, then hand straight back to
    # the lane follower (no counter-steer pass phase).
    assert start == DriveCommand(-60, 255, "obstacle_avoid_shift_out")
    assert mid == DriveCommand(-60, 255, "obstacle_avoid_shift_out")
    assert finished is None
    assert mission.phase == ContestObstaclePhase.LANE_FOLLOW
    assert not mission.active

    base = DriveCommand(steer=-12, speed=255, reason="contest")
    assert apply_obstacle_mission_override(base, finished, SafetyDecision.clear()) is base


def test_lane1_avoidance_shifts_right_and_locks_direction():
    mission = ContestObstacleMission(timed_config())
    car = [car_with_area(6000.0)]

    start = step(mission, now_s=0.0, scan=forward_scan(900.0), objects=car, lane=1)
    # The detector may report lane 2 mid-maneuver, but the direction selected
    # from the source lane must remain locked for the whole shift-out.
    active = step(mission, now_s=0.3, scan=None, objects=[], lane=2)

    assert start == DriveCommand(60, 255, "obstacle_avoid_shift_out")
    assert active == DriveCommand(60, 255, "obstacle_avoid_shift_out")
    assert mission.avoidance_source_lane == 1


def test_shift_out_steer_ignores_max_steer_and_follows_tuned_value():
    # Tuned shift steer beyond the usual MAX_STEER (80) is emitted verbatim.
    mission = ContestObstacleMission(timed_config(lane2_shift_steer=-100, retrigger_cooldown_s=0.0))
    car = [car_with_area(6000.0)]

    start = step(mission, now_s=0.0, scan=forward_scan(900.0), objects=car, lane=2)

    assert start == DriveCommand(-100, 255, "obstacle_avoid_shift_out")


def test_avoidance_ignores_new_detections_and_keeps_runtime_max_speed():
    mission = ContestObstacleMission(timed_config())
    car = [car_with_area(6000.0)]

    start = step(mission, now_s=0.0, scan=forward_scan(900.0), objects=car, cruise_speed=235)
    large_car = [car_with_area(20000.0)]
    active = step(mission, now_s=0.3, scan=forward_scan(100.0), objects=large_car, cruise_speed=235)

    assert start == DriveCommand(-60, 235, "obstacle_avoid_shift_out")
    assert active == DriveCommand(-60, 235, "obstacle_avoid_shift_out")


def test_next_obstacle_can_trigger_after_shift_out_and_cooldown():
    mission = ContestObstacleMission(
        timed_config(shift_out_duration_s=0.10, retrigger_cooldown_s=1.0)
    )
    car = [car_with_area(6000.0)]

    assert step(mission, now_s=0.0, scan=forward_scan(900.0), objects=car) is not None
    # Shift-out ends at 0.10; clear ahead -> finish, cooldown until 0.11 + 1.0.
    assert step(mission, now_s=0.11, scan=None, objects=[]) is None
    assert mission.phase == ContestObstaclePhase.LANE_FOLLOW

    # Still inside the cooldown window -> blocked.
    assert step(mission, now_s=1.10, scan=forward_scan(900.0), objects=car, lane=1) is None
    second = step(mission, now_s=1.20, scan=forward_scan(900.0), objects=car, lane=1)

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

    # Still shifting at 0.89s (would already have finished without the weight).
    mid = step(mission, now_s=0.89, scan=None, objects=[], lane=2)
    assert mid.reason == "obstacle_avoid_shift_out"

    # After 0.90s the shift-out ends and, with nothing ahead, control returns.
    after = step(mission, now_s=0.91, scan=None, objects=[], lane=2)
    assert after is None
    assert mission.phase == ContestObstaclePhase.LANE_FOLLOW


def test_shift_out_completion_rechains_when_obstacle_still_ahead():
    # No lane-follow gap between consecutive obstacles: when the shift-out
    # completes with a car still ahead, avoidance chains straight into a new
    # shift-out from the lane just entered.
    mission = ContestObstacleMission(timed_config(retrigger_cooldown_s=0.0))
    car = [car_with_area(6000.0)]

    step(mission, now_s=0.0, scan=forward_scan(900.0), objects=car, lane=2)
    # Shift-out (0.70) elapsed and a car is still ahead in the new lane (1) ->
    # immediately re-shift from lane 1 (steer +60), no "contest" frame.
    rechained = step(mission, now_s=0.71, scan=forward_scan(900.0), objects=car, lane=1)

    assert rechained == DriveCommand(60, 255, "obstacle_avoid_shift_out")
    assert mission.phase == ContestObstaclePhase.AVOID_SHIFT_OUT
    assert mission.avoidance_source_lane == 1


def test_shift_out_completion_returns_to_lane_follow_when_clear():
    mission = ContestObstacleMission(timed_config(retrigger_cooldown_s=0.0))
    car = [car_with_area(6000.0)]

    step(mission, now_s=0.0, scan=forward_scan(900.0), objects=car, lane=2)
    # Shift-out completes with nothing ahead -> hand back to the lane follower.
    cleared = step(mission, now_s=0.71, scan=None, objects=[], lane=1)

    assert cleared is None
    assert mission.phase == ContestObstaclePhase.LANE_FOLLOW


def test_camera_only_trigger_uses_bbox_far_y_without_lidar():
    # use_lidar=False: no scan, proximity comes from the car bbox bottom edge.
    # car_with_area(area) has bbox y from 20 to 20+area/50, so bottom_y = 20+h.
    # Frame height 480.  far_y ratio = bottom_y / 480.
    mission = ContestObstacleMission(
        ContestObstacleMissionConfig(
            use_lidar=False, trigger_frames=1, camera_trigger_far_y_ratio=0.55
        )
    )

    # bottom_y = 20 + 6000/50 = 140 -> ratio 0.29 < 0.55 -> too far, no trigger.
    assert step(mission, now_s=0.0, scan=None, objects=[car_with_area(6000.0)]) is None
    assert mission.present_reason().startswith("car_too_far")

    # bottom_y = 20 + 20000/50 = 420 -> ratio 0.875 >= 0.55 -> near, triggers.
    command = step(mission, now_s=0.1, scan=None, objects=[car_with_area(20000.0)])
    assert command == DriveCommand(-60, 255, "obstacle_avoid_shift_out")
    assert mission.phase == ContestObstaclePhase.AVOID_SHIFT_OUT


def test_camera_only_zero_far_y_ratio_triggers_on_any_central_car():
    mission = ContestObstacleMission(
        ContestObstacleMissionConfig(
            use_lidar=False, trigger_frames=1, camera_trigger_far_y_ratio=0.0
        )
    )
    # A car high in the frame (bottom_y = 40) still triggers when the gate is 0.
    command = step(mission, now_s=0.0, scan=None, objects=[car_with_area(1000.0)])
    assert command == DriveCommand(-60, 255, "obstacle_avoid_shift_out")


def test_obstacle_alert_reflects_sensing_and_active_maneuver():
    mission = ContestObstacleMission(timed_config(trigger_frames=3, retrigger_cooldown_s=0.0))
    car = [car_with_area(6000.0)]

    # Sensing an obstacle before the trigger fires already raises the alert.
    step(mission, now_s=0.0, scan=forward_scan(900.0), objects=car)
    assert mission.obstacle_alert
    assert mission.camera_sees_car
    assert mission.lidar_is_near

    # Nothing sensed and not maneuvering -> no alert.
    step(mission, now_s=0.1, scan=None, objects=[])
    assert not mission.obstacle_alert


def test_traffic_light_stop_is_preserved_during_avoidance_override():
    base_command = DriveCommand(12, 0, "traffic_light_stop")
    mission_command = DriveCommand(-60, 255, "obstacle_avoid_shift_out")
    traffic_stop = SafetyDecision(speed_scale=0.0, should_stop=True, reason="traffic_light_stop")

    command = apply_obstacle_mission_override(base_command, mission_command, traffic_stop)

    assert command is base_command
