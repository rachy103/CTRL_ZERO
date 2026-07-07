from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ctrl_zero.perception import DetectedObject, compact_class_name
from ctrl_zero.safety import SafetyDecision
from ctrl_zero.vision.base import LaneDetection


@dataclass(frozen=True)
class VisionObstacleConfig:
    enabled: bool = True
    object_classes: tuple[str, ...] = (
        "car",
        "obstacle",
        "person",
        "pedestrian",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "cone",
        "barrier",
    )
    min_confidence: float = 0.30
    corridor_width_ratio: float = 0.45
    corridor_margin_ratio: float = 0.08
    far_y_ratio: float = 0.58
    slow_bottom_y_ratio: float = 0.58
    stop_bottom_y_ratio: float = 0.78
    slow_area_ratio: float = 0.035
    stop_area_ratio: float = 0.080
    slow_speed_scale: float = 0.45


def analyze_vision_obstacles(
    lane: LaneDetection,
    config: VisionObstacleConfig | None = None,
) -> SafetyDecision:
    config = config or VisionObstacleConfig()
    if not config.enabled or not lane.objects:
        return SafetyDecision.clear()

    h, w = lane.annotated.shape[:2]
    frame_area = max(float(h * w), 1.0)
    candidates = [
        obj
        for obj in lane.objects
        if obj.confidence >= config.min_confidence
        and _is_obstacle_class(obj.class_name, config.object_classes)
        and _in_forward_corridor(obj, lane, w, h, config)
    ]
    if not candidates:
        return SafetyDecision.clear()

    stop_candidates = [
        obj
        for obj in candidates
        if _bottom_ratio(obj, h) >= config.stop_bottom_y_ratio or obj.bbox.area / frame_area >= config.stop_area_ratio
    ]
    if stop_candidates:
        obj = _most_urgent(stop_candidates, h, frame_area)
        return SafetyDecision(
            speed_scale=0.0,
            should_stop=True,
            reason="vision_obstacle_stop",
            vision_obstacle=obj,
        )

    slow_candidates = [
        obj
        for obj in candidates
        if _bottom_ratio(obj, h) >= config.slow_bottom_y_ratio or obj.bbox.area / frame_area >= config.slow_area_ratio
    ]
    if slow_candidates:
        obj = _most_urgent(slow_candidates, h, frame_area)
        return SafetyDecision(
            speed_scale=config.slow_speed_scale,
            should_stop=False,
            reason="vision_obstacle_slow",
            vision_obstacle=obj,
        )

    return SafetyDecision.clear()


def _is_obstacle_class(class_name: str, allowed: Iterable[str]) -> bool:
    compact = compact_class_name(class_name)
    return any(compact == compact_class_name(item) or compact.startswith(compact_class_name(item)) for item in allowed)


def _in_forward_corridor(
    obj: DetectedObject,
    lane: LaneDetection,
    width: int,
    height: int,
    config: VisionObstacleConfig,
) -> bool:
    if _bottom_ratio(obj, height) < config.far_y_ratio:
        return False

    center_x = _lane_center_x_at_y(lane, obj.bbox.bottom_y, height)
    fallback_half_width = width * config.corridor_width_ratio / 2.0
    if lane.lane_width_px is None:
        half_width = fallback_half_width
    else:
        half_width = max(fallback_half_width, lane.lane_width_px / 2.0)
    half_width += width * config.corridor_margin_ratio
    return abs(obj.bbox.center_x - center_x) <= half_width


def _lane_center_x_at_y(lane: LaneDetection, y: float, height: int) -> float:
    if lane.lane_center_near_x is None:
        return lane.frame_center_x
    if lane.lane_center_far_x is None:
        return lane.lane_center_near_x

    near_y = height * 0.95
    far_y = height * 0.68
    span = max(near_y - far_y, 1.0)
    ratio = min(max((y - far_y) / span, 0.0), 1.0)
    return lane.lane_center_far_x + ratio * (lane.lane_center_near_x - lane.lane_center_far_x)


def _bottom_ratio(obj: DetectedObject, height: int) -> float:
    return obj.bbox.bottom_y / max(float(height), 1.0)


def _most_urgent(objects: Iterable[DetectedObject], height: int, frame_area: float) -> DetectedObject:
    return max(objects, key=lambda obj: (_bottom_ratio(obj, height), obj.bbox.area / frame_area, obj.confidence))
