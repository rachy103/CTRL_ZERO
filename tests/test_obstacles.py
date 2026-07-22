from __future__ import annotations

import numpy as np
import pytest

from ctrl_zero.lidar import ObstacleDecision
from ctrl_zero.obstacles import (
    ContestObstacleMission,
    ContestObstacleMissionConfig,
    ContestObstaclePhase,
    LaneChangeState,
    VisionObstacleConfig,
    analyze_vision_obstacles,
    apply_lane_change_for_obstacle,
)
from ctrl_zero.perception import BoundingBox, DetectedObject
from ctrl_zero.safety import SafetyDecision, build_safety_decision
from ctrl_zero.vision.base import LaneDetection, LaneReference


def lane_with_objects(objects, lane_label="", lane_references=None, offset_norm=0.0) -> LaneDetection:
    return LaneDetection(
        lanes=[],
        left_fit=None,
        right_fit=None,
        lane_center_near_x=100.0,
        lane_center_far_x=100.0,
        frame_center_x=100.0,
        offset_px=offset_norm * 100.0,
        offset_norm=offset_norm,
        heading_deg=0.0,
        lane_width_px=80.0,
        confidence=0.8,
        mask=None,
        annotated=np.zeros((200, 200, 3), dtype=np.uint8),
        objects=objects,
        lane_label=lane_label,
        lane_references=lane_references or {},
    )


def obj(class_name: str, bbox: BoundingBox, confidence: float = 0.90, lane_label: str = "") -> DetectedObject:
    return DetectedObject(class_name=class_name, confidence=confidence, bbox=bbox, lane_label=lane_label)


def two_lane_references():
    return {
        "lane1": LaneReference(name="lane1", near_x=60.0, far_x=60.0, near_y=190, far_y=136, width_px=50.0),
        "lane2": LaneReference(name="lane2", near_x=140.0, far_x=140.0, near_y=190, far_y=136, width_px=50.0),
    }


def test_center_obstacle_near_bottom_is_detected_without_stop_or_slow():
    lane = lane_with_objects([obj("car", BoundingBox(80, 100, 120, 170))])

    decision = analyze_vision_obstacles(lane, VisionObstacleConfig())

    assert not decision.should_stop
    assert decision.speed_scale == 1.0
    assert decision.reason == "vision_obstacle_detected"
    assert decision.vision_obstacle is not None
    assert decision.vision_obstacle.class_name == "car"


def test_center_obstacle_midfield_is_detected_without_speed_scale():
    lane = lane_with_objects([obj("car", BoundingBox(90, 105, 110, 130))])

    decision = analyze_vision_obstacles(lane, VisionObstacleConfig())

    assert not decision.should_stop
    assert decision.reason == "vision_obstacle_detected"
    assert decision.speed_scale == 1.0


def test_outside_corridor_obstacle_is_ignored():
    lane = lane_with_objects([obj("car", BoundingBox(5, 100, 30, 180))])

    decision = analyze_vision_obstacles(lane, VisionObstacleConfig())

    assert not decision.should_stop
    assert decision.speed_scale == 1.0
    assert decision.reason == "clear"


def test_lidar_slow_still_applies_when_vision_obstacle_is_detected():
    lane = lane_with_objects([obj("car", BoundingBox(80, 100, 120, 170))])
    vision = analyze_vision_obstacles(lane, VisionObstacleConfig())
    lidar = ObstacleDecision(nearest_front_mm=700.0, speed_scale=0.65, should_stop=False, front_points=4)

    fused = build_safety_decision(lidar=lidar, traffic_light_state="green", vision_obstacle_decision=vision)

    assert not fused.should_stop
    assert fused.speed_scale == 0.65
    assert fused.reason == "lidar_slow"
    assert fused.nearest_front_mm == 700.0
    assert fused.vision_obstacle is not None


def _forward_scan(distance_mm: float) -> np.ndarray:
    # A single return straight ahead (raw 180 = ROS 0 on this vehicle).
    return np.array([[180.0, distance_mm]], dtype=np.float32)


def test_mission_triggers_lane_change_on_forward_lidar_obstacle():
    config = ContestObstacleMissionConfig()
    mission = ContestObstacleMission(config)

    # A car ahead within the obstacle distance should trip after the required
    # number of consecutive frames, then change toward lane 1.
    command = None
    for _ in range(config.lane2_obstacle_frames):
        assert mission.phase == ContestObstaclePhase.MONITOR_LANE2
        command = mission.step(
            objects=[],
            current_lane="lane2",
            heading_deg=0.0,
            vehicle_position_x=0.0,
            frame_width=640,
            lidar_scan=_forward_scan(500.0),
        )

    assert mission.phase == ContestObstaclePhase.CHANGE_TO_LANE1
    assert command is not None
    assert command.reason == "obstacle_lane_change_2_to_1"


def test_mission_does_not_trigger_when_forward_is_clear():
    config = ContestObstacleMissionConfig()
    mission = ContestObstacleMission(config)

    for _ in range(config.lane2_obstacle_frames + 3):
        mission.step(
            objects=[],
            current_lane="lane2",
            heading_deg=0.0,
            vehicle_position_x=0.0,
            frame_width=640,
            lidar_scan=_forward_scan(config.lidar_obstacle_distance_mm + 500.0),
        )

    assert mission.phase == ContestObstaclePhase.MONITOR_LANE2


def test_mission_side_obstacle_no_longer_triggers():
    # A return only to the left (raw 90) must NOT trigger now that the mission
    # watches the forward cone.
    config = ContestObstacleMissionConfig()
    mission = ContestObstacleMission(config)
    left_scan = np.array([[90.0, 400.0]], dtype=np.float32)

    for _ in range(config.lane2_obstacle_frames + 3):
        mission.step(
            objects=[],
            current_lane="lane2",
            heading_deg=0.0,
            vehicle_position_x=0.0,
            frame_width=640,
            lidar_scan=left_scan,
        )

    assert mission.phase == ContestObstaclePhase.MONITOR_LANE2


def test_obstacle_in_other_lane_is_ignored_when_lane_labels_exist():
    lane = lane_with_objects(
        [obj("car", BoundingBox(120, 100, 160, 170), lane_label="lane2")],
        lane_label="lane1",
        lane_references=two_lane_references(),
    )

    decision = analyze_vision_obstacles(lane, VisionObstacleConfig())

    assert not decision.should_stop
    assert decision.reason == "clear"


def test_current_lane_obstacle_changes_target_to_other_lane():
    lane = lane_with_objects(
        [obj("car", BoundingBox(40, 100, 80, 170), lane_label="lane1")],
        lane_label="lane1",
        lane_references=two_lane_references(),
    )
    decision = analyze_vision_obstacles(lane, VisionObstacleConfig())

    changed_lane, changed_decision = apply_lane_change_for_obstacle(lane, decision, VisionObstacleConfig())

    assert changed_lane.lane_label == "lane1"
    assert changed_decision.speed_scale == 1.0
    assert changed_decision.avoidance_steer == 0.0
    assert changed_decision.target_lane_label == "lane2"
    assert changed_lane.lane_center_near_x == pytest.approx(60.0)
    assert 60.0 < changed_lane.lane_center_far_x < 140.0
    assert changed_lane.heading_deg > 0.0
    assert "avoidance_path=lane1->lane2" in changed_lane.lane_pair_label
    assert not changed_decision.should_stop
    assert changed_decision.reason == "vision_obstacle_lane_change_lane1_to_lane2"


def test_current_lane_small_obstacle_does_not_change_lane_until_bbox_is_large_enough():
    lane = lane_with_objects(
        [obj("car", BoundingBox(50, 100, 70, 130), lane_label="lane1")],
        lane_label="lane1",
        lane_references=two_lane_references(),
    )
    decision = analyze_vision_obstacles(lane, VisionObstacleConfig())

    changed_lane, changed_decision = apply_lane_change_for_obstacle(
        lane,
        decision,
        VisionObstacleConfig(lane_change_area_ratio=0.060),
    )

    assert changed_lane.lane_label == "lane1"
    assert changed_decision.reason == "vision_obstacle_detected"
    assert changed_decision.avoidance_steer == 0.0


def test_lane2_obstacle_changes_target_to_lane1():
    lane = lane_with_objects(
        [obj("car", BoundingBox(120, 100, 160, 170), lane_label="lane2")],
        lane_label="lane2",
        lane_references=two_lane_references(),
    )
    decision = analyze_vision_obstacles(lane, VisionObstacleConfig())

    changed_lane, changed_decision = apply_lane_change_for_obstacle(lane, decision, VisionObstacleConfig())

    assert changed_lane.lane_label == "lane2"
    assert changed_decision.avoidance_steer == 0.0
    assert changed_decision.target_lane_label == "lane1"
    assert changed_lane.lane_center_near_x == pytest.approx(140.0)
    assert 60.0 < changed_lane.lane_center_far_x < 140.0
    assert changed_lane.heading_deg < 0.0
    assert "avoidance_path=lane2->lane1" in changed_lane.lane_pair_label
    assert changed_decision.reason == "vision_obstacle_lane_change_lane2_to_lane1"


def test_active_lane_change_locks_target_until_completion():
    state = LaneChangeState()
    config = VisionObstacleConfig()
    lane_start = lane_with_objects(
        [obj("car", BoundingBox(40, 100, 80, 170), lane_label="lane1")],
        lane_label="lane1",
        lane_references=two_lane_references(),
    )
    first_decision = analyze_vision_obstacles(lane_start, config)
    _, first_change = apply_lane_change_for_obstacle(lane_start, first_decision, config, state)

    lane_mid_change = lane_with_objects(
        [obj("car", BoundingBox(120, 100, 160, 170), lane_label="lane2")],
        lane_label="lane2",
        lane_references=two_lane_references(),
        offset_norm=0.30,
    )
    mid_decision = analyze_vision_obstacles(lane_mid_change, config)
    _, mid_change = apply_lane_change_for_obstacle(lane_mid_change, mid_decision, config, state)

    assert state.active
    assert first_change.target_lane_label == "lane2"
    assert mid_change.target_lane_label == "lane2"
    assert mid_change.reason == "vision_obstacle_lane_change_lane1_to_lane2"


def test_active_lane_change_advances_virtual_path_gradually():
    state = LaneChangeState()
    config = VisionObstacleConfig(lane_change_path_progress_step=0.25)
    lane_start = lane_with_objects(
        [obj("car", BoundingBox(40, 100, 80, 170), lane_label="lane1")],
        lane_label="lane1",
        lane_references=two_lane_references(),
    )
    first_decision = analyze_vision_obstacles(lane_start, config)
    first_lane, _ = apply_lane_change_for_obstacle(lane_start, first_decision, config, state)

    next_decision = analyze_vision_obstacles(lane_start, config)
    second_lane, _ = apply_lane_change_for_obstacle(lane_start, next_decision, config, state)

    assert first_lane.lane_center_near_x == pytest.approx(60.0)
    assert first_lane.lane_center_far_x > first_lane.lane_center_near_x
    assert first_lane.lane_center_far_x < second_lane.lane_center_far_x
    assert first_lane.lane_center_near_x < second_lane.lane_center_near_x < 140.0


def test_new_lane_change_can_start_after_previous_change_completes():
    state = LaneChangeState()
    config = VisionObstacleConfig(lane_change_complete_frames=1)
    lane_start = lane_with_objects(
        [obj("car", BoundingBox(40, 100, 80, 170), lane_label="lane1")],
        lane_label="lane1",
        lane_references=two_lane_references(),
    )
    first_decision = analyze_vision_obstacles(lane_start, config)
    apply_lane_change_for_obstacle(lane_start, first_decision, config, state)

    lane_complete = lane_with_objects(
        [],
        lane_label="lane2",
        lane_references=two_lane_references(),
        offset_norm=0.02,
    )
    apply_lane_change_for_obstacle(lane_complete, SafetyDecision.clear(), config, state)

    lane_after_completion = lane_with_objects(
        [obj("car", BoundingBox(120, 100, 160, 170), lane_label="lane2")],
        lane_label="lane2",
        lane_references=two_lane_references(),
    )
    next_decision = analyze_vision_obstacles(lane_after_completion, config)
    _, next_change = apply_lane_change_for_obstacle(lane_after_completion, next_decision, config, state)

    assert state.active
    assert next_change.target_lane_label == "lane1"
    assert next_change.reason == "vision_obstacle_lane_change_lane2_to_lane1"


def test_disabled_config_clears_active_lane_change_state():
    state = LaneChangeState(active=True, source_lane_label="lane1", target_lane_label="lane2")
    lane = lane_with_objects(
        [obj("car", BoundingBox(40, 100, 80, 170), lane_label="lane1")],
        lane_label="lane1",
        lane_references=two_lane_references(),
    )
    decision = SafetyDecision(
        reason="vision_obstacle_detected",
        vision_obstacle=lane.objects[0],
    )

    _, changed_decision = apply_lane_change_for_obstacle(lane, decision, VisionObstacleConfig(enabled=False), state)

    assert not state.active
    assert changed_decision.avoidance_steer == 0.0
    assert changed_decision.reason == "vision_obstacle_detected"
