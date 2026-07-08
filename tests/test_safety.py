from __future__ import annotations

import numpy as np

from ctrl_zero.control import DriveConfig, DriveController
from ctrl_zero.lidar import ObstacleDecision
from ctrl_zero.perception import BoundingBox, DetectedObject
from ctrl_zero.safety import (
    SafetyDecision,
    build_safety_decision,
    fuse_safety_decisions,
    safety_from_lidar,
    safety_from_traffic_light,
)
from ctrl_zero.vision.base import LaneDetection


def lane() -> LaneDetection:
    return LaneDetection(
        lanes=[],
        left_fit=None,
        right_fit=None,
        lane_center_near_x=320.0,
        lane_center_far_x=320.0,
        frame_center_x=320.0,
        offset_px=0.0,
        offset_norm=0.0,
        heading_deg=0.0,
        lane_width_px=250.0,
        confidence=0.8,
        mask=None,
        annotated=np.zeros((10, 10, 3), dtype=np.uint8),
    )


def test_red_traffic_light_stops_controller():
    controller = DriveController(DriveConfig(min_confidence=0.1))
    safety = build_safety_decision(traffic_light_state="red")

    command = controller.compute(lane(), safety, "auto")

    assert command.speed == 0
    assert command.reason == "traffic_light_stop"


def test_green_traffic_light_keeps_lane_command():
    controller = DriveController(DriveConfig(min_confidence=0.1))
    safety = build_safety_decision(traffic_light_state="green")

    command = controller.compute(lane(), safety, "auto")

    assert command.speed > 0
    assert command.reason == "contest"


def test_red_traffic_light_below_bbox_threshold_does_not_stop():
    controller = DriveController(DriveConfig(min_confidence=0.1))
    traffic_obj = DetectedObject("traffic_light", 0.95, BoundingBox(0, 0, 10, 10))
    safety = build_safety_decision(
        traffic_light_state="red",
        traffic_light_object=traffic_obj,
        traffic_light_frame_area=10000.0,
        traffic_light_min_stop_area_ratio=0.020,
    )

    command = controller.compute(lane(), safety, "auto")

    assert command.speed > 0
    assert command.reason == "contest"
    assert safety.traffic_light_area_ratio == 0.010


def test_red_traffic_light_at_bbox_threshold_stops():
    controller = DriveController(DriveConfig(min_confidence=0.1))
    traffic_obj = DetectedObject("traffic_light", 0.95, BoundingBox(0, 0, 20, 20))
    safety = build_safety_decision(
        traffic_light_state="red",
        traffic_light_object=traffic_obj,
        traffic_light_frame_area=10000.0,
        traffic_light_min_stop_area_ratio=0.020,
    )

    command = controller.compute(lane(), safety, "auto")

    assert command.speed == 0
    assert command.reason == "traffic_light_stop"
    assert safety.traffic_light_area_ratio == 0.040


def test_fuse_safety_decisions_uses_most_restrictive_speed_scale():
    lidar = safety_from_lidar(
        ObstacleDecision(nearest_front_mm=700.0, speed_scale=0.65, should_stop=False, front_points=4)
    )
    traffic = safety_from_traffic_light("green")

    fused = fuse_safety_decisions(traffic, lidar)

    assert not fused.should_stop
    assert fused.speed_scale == 0.65
    assert fused.reason == "lidar_slow"


def test_fuse_safety_decisions_stop_overrides_slowdown():
    lidar = safety_from_lidar(
        ObstacleDecision(nearest_front_mm=700.0, speed_scale=0.65, should_stop=False, front_points=4)
    )
    traffic = safety_from_traffic_light("yellow")

    fused = fuse_safety_decisions(traffic, lidar)

    assert fused.should_stop
    assert fused.speed_scale == 0.0
    assert fused.reason == "traffic_light_stop"
    assert fused.nearest_front_mm == 700.0


def test_fuse_safety_decisions_preserves_lane_change_reason_without_speed_scale():
    vision = SafetyDecision(
        speed_scale=1.0,
        should_stop=False,
        reason="vision_obstacle_lane_change_lane1_to_lane2",
        avoidance_steer=40.0,
        current_lane_label="lane1",
        target_lane_label="lane2",
    )

    fused = fuse_safety_decisions(safety_from_traffic_light("green"), vision)

    assert not fused.should_stop
    assert fused.speed_scale == 1.0
    assert fused.avoidance_steer == 40.0
    assert fused.reason == "vision_obstacle_lane_change_lane1_to_lane2"
