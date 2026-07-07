from __future__ import annotations

import numpy as np

from ctrl_zero.lidar import ObstacleDecision
from ctrl_zero.obstacles import VisionObstacleConfig, analyze_vision_obstacles
from ctrl_zero.perception import BoundingBox, DetectedObject
from ctrl_zero.safety import build_safety_decision
from ctrl_zero.vision.base import LaneDetection


def lane_with_objects(objects) -> LaneDetection:
    return LaneDetection(
        lanes=[],
        left_fit=None,
        right_fit=None,
        lane_center_near_x=100.0,
        lane_center_far_x=100.0,
        frame_center_x=100.0,
        offset_px=0.0,
        offset_norm=0.0,
        heading_deg=0.0,
        lane_width_px=80.0,
        confidence=0.8,
        mask=None,
        annotated=np.zeros((200, 200, 3), dtype=np.uint8),
        objects=objects,
    )


def obj(class_name: str, bbox: BoundingBox, confidence: float = 0.90) -> DetectedObject:
    return DetectedObject(class_name=class_name, confidence=confidence, bbox=bbox)


def test_center_obstacle_near_bottom_stops():
    lane = lane_with_objects([obj("car", BoundingBox(80, 100, 120, 170))])

    decision = analyze_vision_obstacles(lane, VisionObstacleConfig())

    assert decision.should_stop
    assert decision.reason == "vision_obstacle_stop"
    assert decision.vision_obstacle is not None
    assert decision.vision_obstacle.class_name == "car"


def test_center_obstacle_midfield_slows():
    lane = lane_with_objects([obj("obstacle", BoundingBox(90, 105, 110, 130))])

    decision = analyze_vision_obstacles(lane, VisionObstacleConfig(slow_speed_scale=0.40))

    assert not decision.should_stop
    assert decision.reason == "vision_obstacle_slow"
    assert decision.speed_scale == 0.40


def test_outside_corridor_obstacle_is_ignored():
    lane = lane_with_objects([obj("car", BoundingBox(5, 100, 30, 180))])

    decision = analyze_vision_obstacles(lane, VisionObstacleConfig())

    assert not decision.should_stop
    assert decision.speed_scale == 1.0
    assert decision.reason == "clear"


def test_lidar_and_vision_obstacle_fusion_uses_most_restrictive_decision():
    lane = lane_with_objects([obj("car", BoundingBox(80, 100, 120, 170))])
    vision = analyze_vision_obstacles(lane, VisionObstacleConfig())
    lidar = ObstacleDecision(nearest_front_mm=700.0, speed_scale=0.65, should_stop=False, front_points=4)

    fused = build_safety_decision(lidar=lidar, traffic_light_state="green", vision_obstacle_decision=vision)

    assert fused.should_stop
    assert fused.speed_scale == 0.0
    assert fused.reason == "vision_obstacle_stop"
    assert fused.nearest_front_mm == 700.0
