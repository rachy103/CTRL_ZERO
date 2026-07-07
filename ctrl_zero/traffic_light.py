from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from ctrl_zero.perception import BoundingBox, DetectedObject, compact_class_name

TRAFFIC_LIGHT_UNKNOWN = "unknown"
TRAFFIC_LIGHT_RED = "red"
TRAFFIC_LIGHT_YELLOW = "yellow"
TRAFFIC_LIGHT_GREEN = "green"
TRAFFIC_LIGHT_STOP_STATES = (TRAFFIC_LIGHT_RED, TRAFFIC_LIGHT_YELLOW)


@dataclass(frozen=True)
class TrafficLightConfig:
    min_confidence: float = 0.20
    min_color_ratio: float = 0.015
    min_color_pixels: int = 6


def is_traffic_light_name(class_name: str) -> bool:
    compact = compact_class_name(class_name)
    if parse_traffic_light_state(class_name) != TRAFFIC_LIGHT_UNKNOWN:
        return True
    return compact in {"trafficlight", "signallight", "signal", "stoplight"} or compact.startswith(
        ("trafficlight", "signallight", "stoplight")
    )


def parse_traffic_light_state(class_name: str) -> str:
    compact = compact_class_name(class_name)
    if not compact:
        return TRAFFIC_LIGHT_UNKNOWN
    if "red" in compact:
        return TRAFFIC_LIGHT_RED
    if "yellow" in compact or "amber" in compact:
        return TRAFFIC_LIGHT_YELLOW
    if "green" in compact:
        return TRAFFIC_LIGHT_GREEN
    return TRAFFIC_LIGHT_UNKNOWN


def traffic_light_object(objects: Iterable[DetectedObject]) -> DetectedObject | None:
    candidates = [obj for obj in objects if is_traffic_light_name(obj.class_name)]
    if not candidates:
        return None
    return max(candidates, key=lambda obj: obj.confidence)


def traffic_light_state_from_objects(
    frame: np.ndarray,
    objects: Iterable[DetectedObject],
    config: TrafficLightConfig | None = None,
) -> str:
    config = config or TrafficLightConfig()
    candidates = [
        obj
        for obj in objects
        if obj.confidence >= config.min_confidence and is_traffic_light_name(obj.class_name)
    ]
    if not candidates:
        return TRAFFIC_LIGHT_UNKNOWN

    states: list[str] = []
    for obj in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        parsed = parse_traffic_light_state(obj.class_name)
        states.append(parsed if parsed != TRAFFIC_LIGHT_UNKNOWN else classify_traffic_light_crop(frame, obj.bbox, config))

    for state in (TRAFFIC_LIGHT_RED, TRAFFIC_LIGHT_YELLOW, TRAFFIC_LIGHT_GREEN):
        if state in states:
            return state
    return TRAFFIC_LIGHT_UNKNOWN


def classify_traffic_light_crop(
    frame: np.ndarray,
    bbox: BoundingBox,
    config: TrafficLightConfig | None = None,
) -> str:
    config = config or TrafficLightConfig()
    if frame.size == 0:
        return TRAFFIC_LIGHT_UNKNOWN

    h, w = frame.shape[:2]
    box = bbox.clipped(w, h)
    x1 = int(round(box.x1))
    y1 = int(round(box.y1))
    x2 = int(round(box.x2))
    y2 = int(round(box.y2))
    if x2 <= x1 or y2 <= y1:
        return TRAFFIC_LIGHT_UNKNOWN

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return TRAFFIC_LIGHT_UNKNOWN

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 70, 60), (10, 255, 255)) | cv2.inRange(hsv, (160, 70, 60), (179, 255, 255))
    yellow = cv2.inRange(hsv, (15, 70, 70), (40, 255, 255))
    green = cv2.inRange(hsv, (45, 60, 50), (95, 255, 255))

    scores = {
        TRAFFIC_LIGHT_RED: int(cv2.countNonZero(red)),
        TRAFFIC_LIGHT_YELLOW: int(cv2.countNonZero(yellow)),
        TRAFFIC_LIGHT_GREEN: int(cv2.countNonZero(green)),
    }
    state, pixels = max(scores.items(), key=lambda item: item[1])
    threshold = max(config.min_color_pixels, int(crop.shape[0] * crop.shape[1] * config.min_color_ratio))
    return state if pixels >= threshold else TRAFFIC_LIGHT_UNKNOWN
