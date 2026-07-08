from __future__ import annotations

import numpy as np

from ctrl_zero.lidar import ObstacleDecision
from ctrl_zero.obstacles import LaneChangeState, VisionObstacleConfig, analyze_vision_obstacles, apply_lane_change_for_obstacle
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
    assert changed_decision.avoidance_steer == 80.0
    assert changed_decision.target_lane_label == "lane2"
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
    assert changed_decision.avoidance_steer == -80.0
    assert changed_decision.target_lane_label == "lane1"
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
