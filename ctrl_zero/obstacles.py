from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable

import cv2
import numpy as np

from ctrl_zero.common import clamp
from ctrl_zero.perception import DetectedObject, compact_class_name
from ctrl_zero.safety import SafetyDecision
from ctrl_zero.vision.base import LaneDetection


@dataclass(frozen=True)
class VisionObstacleConfig:
    enabled: bool = True
    object_classes: tuple[str, ...] = (
        "car",
    )
    min_confidence: float = 0.30
    require_current_lane_match: bool = True
    lane_change_enabled: bool = True
    lane_change_area_ratio: float = 0.060
    avoidance_steer_weight: float = 1.0
    avoidance_steer_limit: float = 80.0
    lane_change_path_progress_step: float = 0.12
    lane_change_path_lookahead_progress: float = 0.35
    lane_change_complete_offset_norm: float = 0.12
    lane_change_complete_frames: int = 2
    corridor_width_ratio: float = 0.45
    corridor_margin_ratio: float = 0.08
    far_y_ratio: float = 0.10


@dataclass
class LaneChangeState:
    active: bool = False
    source_lane_label: str = ""
    target_lane_label: str = ""
    completed_frames: int = 0
    progress: float = 0.0
    vision_obstacle: DetectedObject | None = None

    def start(self, source_lane_label: str, target_lane_label: str, vision_obstacle: DetectedObject | None) -> None:
        self.active = True
        self.source_lane_label = source_lane_label
        self.target_lane_label = target_lane_label
        self.completed_frames = 0
        self.progress = 0.0
        self.vision_obstacle = vision_obstacle

    def clear(self) -> None:
        self.active = False
        self.source_lane_label = ""
        self.target_lane_label = ""
        self.completed_frames = 0
        self.progress = 0.0
        self.vision_obstacle = None


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
        and _matches_current_lane(obj, lane, config)
        and _in_forward_corridor(obj, lane, w, h, config)
    ]
    if not candidates:
        return SafetyDecision.clear()

    return SafetyDecision(
        speed_scale=1.0,
        should_stop=False,
        reason="vision_obstacle_detected",
        vision_obstacle=_most_urgent(candidates, h, frame_area),
    )


def apply_lane_change_for_obstacle(
    lane: LaneDetection,
    decision: SafetyDecision,
    config: VisionObstacleConfig | None = None,
    state: LaneChangeState | None = None,
) -> tuple[LaneDetection, SafetyDecision]:
    config = config or VisionObstacleConfig()
    if not config.enabled or not config.lane_change_enabled:
        if state is not None:
            state.clear()
        return lane, decision

    if state is not None and state.active:
        if _lane_change_completed(lane, state, config):
            state.completed_frames += 1
            if state.completed_frames >= max(1, config.lane_change_complete_frames):
                state.clear()
                return lane, decision
        else:
            state.completed_frames = 0
        path_lane = _lane_detection_for_avoidance_path(lane, config, state)
        active_decision = _active_lane_change_decision(path_lane, decision, config, state)
        _advance_lane_change_progress(state, config)
        return path_lane, active_decision

    if decision.vision_obstacle is None:
        return lane, decision

    h, w = lane.annotated.shape[:2]
    frame_area = max(float(h * w), 1.0)
    if _area_ratio(decision.vision_obstacle, frame_area) < config.lane_change_area_ratio:
        return lane, decision

    current_label = _current_lane_label(lane)
    if not current_label:
        return lane, decision

    target_label = _alternate_lane_label(current_label, lane.lane_references.keys())
    if target_label is None:
        return lane, decision

    target = lane.lane_references.get(target_label)
    if target is None:
        return lane, decision

    active_state = state or LaneChangeState()
    active_state.start(current_label, target_label, decision.vision_obstacle)
    path_lane = _lane_detection_for_avoidance_path(lane, config, active_state)
    if state is not None:
        _advance_lane_change_progress(state, config)

    lane_change_decision = replace(
        decision,
        speed_scale=1.0,
        should_stop=False,
        avoidance_steer=0.0,
        current_lane_label=current_label,
        target_lane_label=target_label,
        reason=f"vision_obstacle_lane_change_{current_label}_to_{target_label}",
    )
    return path_lane, lane_change_decision


def _active_lane_change_decision(
    lane: LaneDetection,
    decision: SafetyDecision,
    config: VisionObstacleConfig,
    state: LaneChangeState,
) -> SafetyDecision:
    target = lane.lane_references.get(state.target_lane_label)
    if target is None:
        return replace(
            decision,
            speed_scale=1.0,
            should_stop=False,
            vision_obstacle=decision.vision_obstacle or state.vision_obstacle,
            current_lane_label=state.source_lane_label,
            target_lane_label=state.target_lane_label,
            reason=f"vision_obstacle_lane_change_{state.source_lane_label}_to_{state.target_lane_label}",
        )

    return replace(
        decision,
        speed_scale=1.0,
        should_stop=False,
        vision_obstacle=decision.vision_obstacle or state.vision_obstacle,
        avoidance_steer=0.0,
        current_lane_label=state.source_lane_label,
        target_lane_label=state.target_lane_label,
        reason=f"vision_obstacle_lane_change_{state.source_lane_label}_to_{state.target_lane_label}",
    )


def _is_obstacle_class(class_name: str, allowed: Iterable[str]) -> bool:
    compact = compact_class_name(class_name)
    return any(compact == compact_class_name(item) or compact.startswith(compact_class_name(item)) for item in allowed)


def _matches_current_lane(
    obj: DetectedObject,
    lane: LaneDetection,
    config: VisionObstacleConfig,
) -> bool:
    if not config.require_current_lane_match:
        return True

    current_label = _current_lane_label(lane)
    if not current_label:
        return not lane.lane_references
    if obj.lane_label:
        return obj.lane_label == current_label
    return not lane.lane_references


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
    current_reference = lane.lane_references.get(_current_lane_label(lane))
    reference_width = current_reference.width_px if current_reference is not None else lane.lane_width_px
    if reference_width is None:
        half_width = fallback_half_width
    else:
        half_width = max(fallback_half_width, reference_width / 2.0)
    half_width += width * config.corridor_margin_ratio
    return abs(obj.bbox.center_x - center_x) <= half_width


def _lane_center_x_at_y(lane: LaneDetection, y: float, height: int) -> float:
    current_reference = lane.lane_references.get(_current_lane_label(lane))
    if current_reference is not None:
        return _reference_x_at_y(current_reference.near_x, current_reference.far_x, current_reference.near_y, current_reference.far_y, y)

    if lane.lane_center_near_x is None:
        return lane.frame_center_x
    if lane.lane_center_far_x is None:
        return lane.lane_center_near_x

    near_y = height * 0.95
    far_y = height * 0.68
    span = max(near_y - far_y, 1.0)
    ratio = min(max((y - far_y) / span, 0.0), 1.0)
    return lane.lane_center_far_x + ratio * (lane.lane_center_near_x - lane.lane_center_far_x)


def _reference_x_at_y(near_x: float, far_x: float | None, near_y: int, far_y: int, y: float) -> float:
    if far_x is None:
        return near_x
    span = max(float(near_y - far_y), 1.0)
    ratio = min(max((y - far_y) / span, 0.0), 1.0)
    return far_x + ratio * (near_x - far_x)


def _current_lane_label(lane: LaneDetection) -> str:
    if lane.lane_label:
        return lane.lane_label
    for token in lane.lane_pair_label.split():
        if token.startswith("target=lane"):
            return token.split("=", 1)[1]
    return ""


def _alternate_lane_label(current_label: str, available_labels: Iterable[str]) -> str | None:
    available = set(available_labels)
    if current_label == "lane1" and "lane2" in available:
        return "lane2"
    if current_label == "lane2" and "lane1" in available:
        return "lane1"
    for label in sorted(available):
        if label != current_label:
            return label
    return None


def _lane_detection_for_reference(lane: LaneDetection, target_label: str, target) -> LaneDetection:
    center_near = target.near_x
    center_far = target.far_x
    offset_px = center_near - lane.frame_center_x
    offset_norm = offset_px / max(lane.frame_center_x, 1.0)
    heading_deg = None
    if center_far is not None:
        heading_deg = math.degrees(math.atan2(center_far - center_near, max(target.near_y - target.far_y, 1)))

    return replace(
        lane,
        left_fit=target.fit,
        right_fit=None,
        lane_center_near_x=center_near,
        lane_center_far_x=center_far,
        offset_px=offset_px,
        offset_norm=offset_norm,
        heading_deg=heading_deg,
        lane_width_px=target.width_px or lane.lane_width_px,
        lane_label=target_label,
        lane_pair_label=f"{lane.lane_pair_label} lane_change={_current_lane_label(lane)}->{target_label}",
    )


def _lane_detection_for_avoidance_path(
    lane: LaneDetection,
    config: VisionObstacleConfig,
    state: LaneChangeState,
) -> LaneDetection:
    target = lane.lane_references.get(state.target_lane_label)
    if target is None:
        return lane

    source = lane.lane_references.get(state.source_lane_label)
    near_y = target.near_y
    far_y = target.far_y
    progress_near = _smoothstep(state.progress)
    progress_far = _smoothstep(min(1.0, state.progress + config.lane_change_path_lookahead_progress))

    source_near = _source_x_at_y(lane, source, near_y)
    source_far = _source_x_at_y(lane, source, far_y)
    target_near = _reference_x_at_y(target.near_x, target.far_x, target.near_y, target.far_y, near_y)
    target_far = _reference_x_at_y(target.near_x, target.far_x, target.near_y, target.far_y, far_y)

    center_near = _lerp(source_near, target_near, progress_near)
    center_far = _lerp(source_far, target_far, progress_far)
    offset_px = center_near - lane.frame_center_x
    offset_norm = offset_px / max(lane.frame_center_x, 1.0)
    heading_deg = math.degrees(math.atan2(center_far - center_near, max(near_y - far_y, 1)))
    path_points = _avoidance_path_points(lane, source, target, progress_near, progress_far)
    annotated = _draw_avoidance_path(lane.annotated, path_points, state)

    return replace(
        lane,
        left_fit=None,
        right_fit=None,
        lanes=[path_points] if len(path_points) >= 2 else lane.lanes,
        lane_center_near_x=center_near,
        lane_center_far_x=center_far,
        offset_px=offset_px,
        offset_norm=offset_norm,
        heading_deg=heading_deg,
        lane_width_px=target.width_px or lane.lane_width_px,
        lane_label=state.source_lane_label,
        lane_pair_label=f"{lane.lane_pair_label} avoidance_path={state.source_lane_label}->{state.target_lane_label} progress={state.progress:.2f}",
        annotated=annotated,
    )


def _source_x_at_y(lane: LaneDetection, source, y: float) -> float:
    if source is not None:
        return _reference_x_at_y(source.near_x, source.far_x, source.near_y, source.far_y, y)
    return _lane_center_x_at_y(lane, y, lane.annotated.shape[0])


def _avoidance_path_points(
    lane: LaneDetection,
    source,
    target,
    progress_near: float,
    progress_far: float,
) -> list[tuple[int, int]]:
    step = 6
    near_y = target.near_y
    far_y = target.far_y
    points: list[tuple[int, int]] = []
    for y in range(int(far_y), int(near_y) + 1, step):
        vertical = min(max((y - far_y) / max(float(near_y - far_y), 1.0), 0.0), 1.0)
        progress = _lerp(progress_far, progress_near, vertical)
        source_x = _source_x_at_y(lane, source, y)
        target_x = _reference_x_at_y(target.near_x, target.far_x, target.near_y, target.far_y, y)
        x = _lerp(source_x, target_x, progress)
        points.append((int(round(clamp(x, 0.0, lane.annotated.shape[1] - 1.0))), int(y)))
    if not points or points[-1][1] != int(near_y):
        source_x = _source_x_at_y(lane, source, near_y)
        target_x = _reference_x_at_y(target.near_x, target.far_x, target.near_y, target.far_y, near_y)
        x = _lerp(source_x, target_x, progress_near)
        points.append((int(round(clamp(x, 0.0, lane.annotated.shape[1] - 1.0))), int(near_y)))
    return points


def _draw_avoidance_path(image: np.ndarray, points: list[tuple[int, int]], state: LaneChangeState) -> np.ndarray:
    annotated = image.copy()
    if len(points) >= 2:
        cv2.polylines(annotated, [np.array(points, dtype=np.int32)], isClosed=False, color=(255, 0, 255), thickness=3)
        cv2.circle(annotated, points[-1], 6, (255, 0, 255), -1)
    cv2.putText(
        annotated,
        f"avoid_path {state.source_lane_label}->{state.target_lane_label} p={state.progress:.2f}",
        (10, max(annotated.shape[0] - 18, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"avoid_path {state.source_lane_label}->{state.target_lane_label} p={state.progress:.2f}",
        (10, max(annotated.shape[0] - 18, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 255),
        1,
        cv2.LINE_AA,
    )
    return annotated


def _advance_lane_change_progress(state: LaneChangeState, config: VisionObstacleConfig) -> None:
    state.progress = min(1.0, state.progress + max(config.lane_change_path_progress_step, 0.0))


def _smoothstep(value: float) -> float:
    clipped = min(max(value, 0.0), 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _lerp(start: float, end: float, amount: float) -> float:
    return start + (end - start) * amount


def _avoidance_steer_for_reference(
    lane: LaneDetection,
    target,
    config: VisionObstacleConfig,
    source_label: str = "",
) -> float:
    source_reference = lane.lane_references.get(source_label) if source_label else None
    if source_reference is None:
        current_x = _lane_center_x_at_y(lane, target.near_y, lane.annotated.shape[0])
    else:
        current_x = _reference_x_at_y(
            source_reference.near_x,
            source_reference.far_x,
            source_reference.near_y,
            source_reference.far_y,
            target.near_y,
        )
    delta_x = target.near_x - current_x
    return max(
        -abs(config.avoidance_steer_limit),
        min(abs(config.avoidance_steer_limit), delta_x * config.avoidance_steer_weight),
    )


def _lane_change_completed(lane: LaneDetection, state: LaneChangeState, config: VisionObstacleConfig) -> bool:
    if _current_lane_label(lane) != state.target_lane_label:
        return False
    if lane.offset_norm is None:
        return False
    return abs(lane.offset_norm) <= config.lane_change_complete_offset_norm


def _bottom_ratio(obj: DetectedObject, height: int) -> float:
    return obj.bbox.bottom_y / max(float(height), 1.0)


def _area_ratio(obj: DetectedObject, frame_area: float) -> float:
    return obj.bbox.area / max(frame_area, 1.0)


def _most_urgent(objects: Iterable[DetectedObject], height: int, frame_area: float) -> DetectedObject:
    return max(objects, key=lambda obj: (_bottom_ratio(obj, height), _area_ratio(obj, frame_area), obj.confidence))
