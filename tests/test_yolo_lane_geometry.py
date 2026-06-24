from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ctrl_zero.vision.yolo_lane import YOLOLaneConfig, YOLOLaneDetector


def make_detector() -> YOLOLaneDetector:
    detector = YOLOLaneDetector.__new__(YOLOLaneDetector)
    detector.config = YOLOLaneConfig(
        model_path=Path("unused.pt"),
        min_points_per_lane=3,
        min_valid_y_span_ratio=0.08,
        skeleton_enabled=True,
        skeleton_bridge_gap_px=1,
        dashed_merge_max_x_gap_ratio=0.12,
        curve_fit_degree=2,
        solidify_step_px=5,
        fit_outlier_rejection_px=25.0,
    )
    return detector


def curve_x(y: int) -> int:
    return int(58 + 0.32 * y + 0.0012 * (y - 80) ** 2)


def dashed_curve_mask(height: int, width: int, y_start: int, y_end: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.array([(curve_x(y), y) for y in range(y_start, y_end + 1, 2)], dtype=np.int32)
    cv2.polylines(mask, [points], isClosed=False, color=255, thickness=7)
    return mask


def test_dashed_skeleton_fragments_merge_into_solid_curve():
    detector = make_detector()
    height, width = 160, 240
    masks = [
        dashed_curve_mask(height, width, 18, 42),
        dashed_curve_mask(height, width, 66, 92),
        dashed_curve_mask(height, width, 116, 144),
    ]

    fragments = [detector._centerline_from_mask(mask > 0) for mask in masks]
    fits = detector._fit_lane_groups(fragments, height, width, near_y=150, frame_center_x=180.0)
    solid_lanes = detector._solidified_lanes_from_fits(fits, height, width, near_y=150, far_y=18)

    assert len(fits) == 1
    assert abs(detector._x_at_y(fits[0], 140) - curve_x(140)) < 12
    assert len(solid_lanes) == 1
    assert len(solid_lanes[0]) > 20
    assert solid_lanes[0][0][1] == 18
    assert solid_lanes[0][-1][1] == 150
