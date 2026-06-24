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
    detector.class_names = tuple(name.lower() for name in detector.config.class_names)
    detector.lane_width_px = None
    detector.center_near_x = None
    detector.center_far_x = None
    detector.missed_frames = 0
    return detector


def curve_x(y: int) -> int:
    return int(58 + 0.32 * y + 0.0012 * (y - 80) ** 2)


def constant_fit(x: float) -> np.ndarray:
    return np.array([float(x)])


def dashed_curve_mask(height: int, width: int, y_start: int, y_end: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.array([(curve_x(y), y) for y in range(y_start, y_end + 1, 2)], dtype=np.int32)
    cv2.polylines(mask, [points], isClosed=False, color=255, thickness=7)
    return mask


def drivable_area_mask(height: int, width: int, y_start: int, y_end: int, half_width: int = 28) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for y in range(y_start, y_end + 1):
        center = curve_x(y)
        x1 = max(0, center - half_width)
        x2 = min(width - 1, center + half_width)
        mask[y, x1 : x2 + 1] = 255
    return mask


def straight_area_mask(height: int, width: int, center_x: int, y_start: int, y_end: int, half_width: int = 16) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    x1 = max(0, center_x - half_width)
    x2 = min(width - 1, center_x + half_width)
    mask[y_start : y_end + 1, x1 : x2 + 1] = 255
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


def test_drivable_area_mask_centerline_follows_area_midpoint_curve():
    detector = make_detector()
    detector.config.segmentation_mode = "drivable_area"
    height, width = 160, 240
    mask = drivable_area_mask(height, width, 18, 144)

    points, widths = detector._centerline_from_drivable_area_mask(mask)
    fit = detector._fit_lane_x_of_y(points, height)

    assert len(points) > 20
    assert widths and abs(np.median(widths) - 55) < 4
    assert fit is not None
    assert abs(detector._x_at_y(fit, 140) - curve_x(140)) < 8


def test_lane_area_selects_nearest_lane_center_curve():
    detector = make_detector()
    detector.config.segmentation_mode = "lane_area"
    height, width = 180, 320
    masks_by_name = {
        "lane1": straight_area_mask(height, width, center_x=75, y_start=25, y_end=166),
        "lane2": straight_area_mask(height, width, center_x=172, y_start=25, y_end=166),
    }

    candidates = detector._lane_area_candidates_from_masks(
        masks_by_name,
        image_height=height,
        image_width=width,
        near_y=160,
        far_y=60,
        frame_center_x=160.0,
    )

    assert len(candidates) == 2
    assert candidates[0].name == "lane2"
    assert abs(candidates[0].near_x - 172) < 2
    assert candidates[0].width_px is not None
    assert abs(candidates[0].width_px - 32) < 3


def test_lane_geometry_filter_excludes_car_and_traffic_light():
    detector = make_detector()
    detector.class_names = ("car", "lane1", "lane2", "traffic_light")
    names = {0: "car", 1: "lane1", 2: "traffic light"}

    assert not detector._class_allowed_for_lane_geometry(0, names)
    assert detector._class_allowed_for_lane_geometry(1, names)
    assert not detector._class_allowed_for_lane_geometry(2, names)


def test_right_target_selects_right_adjacent_lane_pair():
    detector = make_detector()
    detector.config.target_lane_pair = "right"
    detector.lane_width_px = 95.0

    selection = detector._select_left_right_fits(
        [constant_fit(60), constant_fit(155), constant_fit(255)],
        frame_center_x=170.0,
        near_y=150,
        selection_y=100,
        image_width=320,
    )

    assert selection.left_fit is not None
    assert selection.right_fit is not None
    assert detector._x_at_y(selection.left_fit, 100) == 155
    assert detector._x_at_y(selection.right_fit, 100) == 255
    assert selection.label.startswith("pair=right 1-2")


def test_right_target_falls_back_to_rightmost_line_as_left_boundary():
    detector = make_detector()
    detector.config.target_lane_pair = "right"
    detector.lane_width_px = 95.0

    selection = detector._select_left_right_fits(
        [constant_fit(60), constant_fit(155)],
        frame_center_x=220.0,
        near_y=150,
        selection_y=100,
        image_width=320,
    )

    assert selection.left_fit is not None
    assert selection.right_fit is None
    assert detector._x_at_y(selection.left_fit, 100) == 155
    assert selection.label.startswith("pair=right single-left")


def test_closest_line_target_uses_nearest_detected_line_not_lane_center():
    detector = make_detector()
    detector.config.target_lane_pair = "right"
    detector.config.target_path_mode = "closest_line"
    detector.lane_width_px = 95.0

    fits = [constant_fit(60), constant_fit(155), constant_fit(255)]
    selection = detector._select_left_right_fits(
        fits,
        frame_center_x=170.0,
        near_y=150,
        selection_y=100,
        image_width=320,
    )
    target = detector._select_control_target(
        selection,
        fits,
        frame_center_x=170.0,
        near_y=150,
        far_y=90,
        image_width=320,
    )

    assert target.fit is not None
    assert target.near_x == 155
    assert target.label.startswith("target=closest_line 1")


def test_lane_center_target_keeps_pair_center_control_mode():
    detector = make_detector()
    detector.config.target_path_mode = "lane_center"
    detector.lane_width_px = 95.0

    target = detector._select_control_target(
        selection=detector._select_left_right_fits([constant_fit(60), constant_fit(155)], 110.0, 150, 100, 240),
        fits=[constant_fit(60), constant_fit(155)],
        frame_center_x=110.0,
        near_y=150,
        far_y=90,
        image_width=240,
    )

    assert target.fit is None
    assert target.label == "target=lane_center"
