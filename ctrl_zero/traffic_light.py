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
# Only a dominant RED stops.  Green drives, and so does a traffic light that is
# detected but shows no lit lamp (state UNKNOWN) — an unlit signal must not
# strand the car.
TRAFFIC_LIGHT_STOP_STATES = (TRAFFIC_LIGHT_RED,)
DEFAULT_TRAFFIC_LIGHT_STOP_AREA_RATIO = 0.010


@dataclass(frozen=True)
class TrafficLightConfig:
    min_confidence: float = 0.20
    min_color_ratio: float = 0.015
    min_color_pixels: int = 6
    # RGB dominance test.  A pixel votes red when R beats both G and B by
    # min_channel_margin (green likewise), and only counts when it is bright
    # enough to be a lit lamp rather than dark housing.
    min_channel_margin: int = 30
    min_lit_value: int = 90


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


def traffic_light_area_ratio(obj: DetectedObject | None, frame_area: float) -> float | None:
    if obj is None:
        return None
    return obj.bbox.area / max(frame_area, 1.0)


def traffic_light_meets_stop_size(
    obj: DetectedObject | None,
    frame_area: float | None,
    min_area_ratio: float,
) -> bool:
    if min_area_ratio <= 0.0:
        return True
    if obj is None or frame_area is None:
        return False
    ratio = traffic_light_area_ratio(obj, frame_area)
    return ratio is not None and ratio >= min_area_ratio


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

    # Red wins over green when several lights are visible: stopping is the safe
    # tie-break.  UNKNOWN (no lamp lit) falls through and drives.
    for state in (TRAFFIC_LIGHT_RED, TRAFFIC_LIGHT_GREEN):
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

    # RGB dominance: count pixels where one channel clearly beats the others and
    # the pixel is bright enough to be a lit lamp.  Whichever of red / green wins
    # is the state; if neither clears the threshold the lamp is off (UNKNOWN),
    # which drives.
    bgr = crop.astype(np.int16)
    blue, green_c, red_c = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    value = bgr.max(axis=2)
    lit = value >= config.min_lit_value

    red_mask = lit & (red_c - np.maximum(green_c, blue) >= config.min_channel_margin)
    green_mask = lit & (green_c - np.maximum(red_c, blue) >= config.min_channel_margin)

    red_px = int(red_mask.sum())
    green_px = int(green_mask.sum())
    threshold = max(config.min_color_pixels, int(crop.shape[0] * crop.shape[1] * config.min_color_ratio))

    if max(red_px, green_px) < threshold:
        return TRAFFIC_LIGHT_UNKNOWN  # detected, but no lamp lit -> go
    return TRAFFIC_LIGHT_RED if red_px >= green_px else TRAFFIC_LIGHT_GREEN
