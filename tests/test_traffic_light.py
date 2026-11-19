from __future__ import annotations

import cv2
import numpy as np

from ctrl_zero.perception import BoundingBox, DetectedObject
from ctrl_zero.traffic_light import (
    classify_traffic_light_crop,
    parse_traffic_light_state,
    traffic_light_state_from_objects,
)


def solid_light(color_bgr: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((40, 30, 3), dtype=np.uint8)
    cv2.circle(frame, (15, 20), 8, color_bgr, -1)
    return frame


def test_parse_explicit_traffic_light_state_from_class_name():
    assert parse_traffic_light_state("red_light") == "red"
    assert parse_traffic_light_state("traffic_light_yellow") == "yellow"
    assert parse_traffic_light_state("green_signal") == "green"
    assert parse_traffic_light_state("traffic_light") == "unknown"


def test_classify_traffic_light_crop_by_color():
    bbox = BoundingBox(0, 0, 30, 40)

    assert classify_traffic_light_crop(solid_light((0, 0, 255)), bbox) == "red"
    assert classify_traffic_light_crop(solid_light((0, 255, 255)), bbox) == "yellow"
    assert classify_traffic_light_crop(solid_light((0, 255, 0)), bbox) == "green"
    assert classify_traffic_light_crop(np.zeros((40, 30, 3), dtype=np.uint8), bbox) == "unknown"


def test_explicit_state_class_overrides_crop_color():
    frame = solid_light((0, 255, 0))
    obj = DetectedObject("red_light", 0.95, BoundingBox(0, 0, 30, 40))

    assert traffic_light_state_from_objects(frame, [obj]) == "red"


def test_stop_width_gate_requires_bbox_to_span_ratio_of_frame():
    from ctrl_zero.traffic_light import traffic_light_meets_stop_width

    narrow = DetectedObject("traffic_light", 0.9, BoundingBox(0, 0, 400, 100))   # 62.5% of 640
    wide = DetectedObject("traffic_light", 0.9, BoundingBox(0, 0, 450, 100))     # 70.3% of 640

    assert not traffic_light_meets_stop_width(narrow, 640, 0.70)
    assert traffic_light_meets_stop_width(wide, 640, 0.70)
    assert not traffic_light_meets_stop_width(None, 640, 0.70)
    assert traffic_light_meets_stop_width(None, 640, 0.0)
