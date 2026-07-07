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
