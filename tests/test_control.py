from __future__ import annotations

import numpy as np

from ctrl_zero.control import DriveConfig, DriveController
from ctrl_zero.lidar import ObstacleDecision
from ctrl_zero.vision.base import LaneDetection


def lane(offset_norm=0.0, confidence=0.8, heading_deg=0.0):
    return LaneDetection(
        lanes=[],
        left_fit=None,
        right_fit=None,
        lane_center_near_x=320.0,
        lane_center_far_x=320.0,
        frame_center_x=320.0,
        offset_px=offset_norm * 320.0,
        offset_norm=offset_norm,
        heading_deg=heading_deg,
        lane_width_px=250.0,
        confidence=confidence,
        mask=None,
        annotated=np.zeros((10, 10, 3), dtype=np.uint8),
    )


def test_low_confidence_stops_auto_mode():
    controller = DriveController(DriveConfig(min_confidence=0.5))
    command = controller.compute(lane(confidence=0.2), None, "auto")
    assert command.speed == 0
    assert command.reason == "low_lane_confidence"


def test_positive_offset_steers_right():
    controller = DriveController(DriveConfig(base_speed=50, min_confidence=0.1, kp_offset=80.0, kp_heading=0.0, kd_offset=0.0))
    command = controller.compute(lane(offset_norm=0.25), None, "auto")
    assert command.steer > 0
    assert command.speed > 0


def test_lidar_stop_overrides_speed():
    controller = DriveController(DriveConfig(base_speed=50, min_confidence=0.1))
    obstacle = ObstacleDecision(nearest_front_mm=300.0, speed_scale=0.0, should_stop=True, front_points=5)
    command = controller.compute(lane(), obstacle, "auto")
    assert command.speed == 0
    assert command.reason == "lidar_stop"
