from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Sequence

import cv2
import numpy as np

from ctrl_zero.common import clamp
from ctrl_zero.control import DriveCommand
from ctrl_zero.lidar import min_distance_mm_in_ros_sector
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


class ContestObstaclePhase(str, Enum):
    LANE_FOLLOW = "lane_follow"
    AVOID_SHIFT_OUT = "avoid_shift_out"
    AVOID_PASS = "avoid_pass"


@dataclass(frozen=True)
class ContestObstacleMissionConfig:
    enabled: bool = True
    camera_min_confidence: float = 0.50
    lidar_obstacle_distance_mm: float = 1000.0
    lidar_min_ros_deg: float = -10.0
    lidar_max_ros_deg: float = 10.0
    raw_angle_for_ros_zero_deg: float = 180.0
    trigger_frames: int = 5
    cruise_speed: int = 255
    lane2_shift_steer: int = -60
    lane1_shift_steer: int = 60
    shift_out_duration_s: float = 0.70
    # The shift-out phase runs longer when the car is already steered hard at
    # detection: duration = shift_out_duration_s + |steer_at_detection| * weight.
    shift_out_steer_weight: float = 0.0
    # Pass phase counter-steers opposite the shift (per lane) until the car's
    # heading matches the new lane's centerline, instead of driving straight or
    # counter-steering for a fixed time.  Defaults are the negatives of the
    # shift steers.
    lane2_pass_steer: int = 60
    lane1_pass_steer: int = -60
    # Counter-steer ends when |heading of the new lane| <= this many degrees
    # (car aligned with the new lane centerline).  pass_max_duration_s is a
    # safety cap so a lost lane never leaves the car counter-steering forever.
    pass_align_heading_deg: float = 5.0
    pass_max_duration_s: float = 2.0
    retrigger_cooldown_s: float = 1.0

@dataclass(frozen=True)
class ContestCarObservation:
    detected: bool
    area: float
    obj: DetectedObject | None = None


def detect_contest_car(
    objects: Sequence[DetectedObject],
    frame_width: int,
    min_confidence: float = 0.50,
) -> ContestCarObservation:
    """Find a car in the central 50% and report its bbox area."""
    x_min = int(float(frame_width) * 0.25)
    x_max = int(float(frame_width) * 0.75)
    for obj in objects:
        # The source obstacle node used an exact class-name comparison.  Its
        # upstream YOLO node filtered detections at 0.5 confidence.
        if obj.class_name != "car" or obj.confidence < min_confidence:
            continue
        center_x = int(obj.bbox.center_x)
        if x_min <= center_x <= x_max:
            area = float(int(obj.bbox.width) * int(obj.bbox.height))
            return ContestCarObservation(True, area, obj)
    return ContestCarObservation(False, 0.0, None)


class ContestObstacleMission:
    """Timed obstacle mode that is independent from the lane-follow controller."""

    def __init__(self, config: ContestObstacleMissionConfig | None = None):
        self.config = config or ContestObstacleMissionConfig()
        self.phase = ContestObstaclePhase.LANE_FOLLOW
        self.current_lane: int | None = None
        self.avoidance_source_lane: int | None = None
        self.active_shift_steer = 0
        self.active_shift_out_duration_s = float(self.config.shift_out_duration_s)
        self.latest_lidar_range_mm: float | None = None
        self.latest_heading_deg: float | None = None
        self.latest_lane_follow_steer = 0
        self.last_car = ContestCarObservation(False, 0.0, None)
        self.trigger_counter = 0
        self.phase_started_s = 0.0
        self.cooldown_until_s = 0.0
        self.cruise_speed = int(self.config.cruise_speed)

    @property
    def active(self) -> bool:
        return self.phase != ContestObstaclePhase.LANE_FOLLOW

    @property
    def is_changing_lane(self) -> bool:
        return self.phase == ContestObstaclePhase.AVOID_SHIFT_OUT

    @property
    def ignoring_obstacles(self) -> bool:
        return self.active

    def step(
        self,
        *,
        objects: Sequence[DetectedObject],
        current_lane: int | str | None,
        frame_width: int,
        lidar_scan: np.ndarray | None,
        heading_deg: float | None = None,
        lane_follow_steer: int = 0,
        cruise_speed: int | None = None,
        now_s: float | None = None,
    ) -> DriveCommand | None:
        now = time.monotonic() if now_s is None else float(now_s)
        observed_lane = _contest_lane_number(current_lane)
        if observed_lane is not None:
            self.current_lane = observed_lane
        self.latest_heading_deg = heading_deg
        self.latest_lane_follow_steer = int(lane_follow_steer)
        self.latest_lidar_range_mm = self._lidar_range(lidar_scan)
        self.last_car = detect_contest_car(objects, frame_width, self.config.camera_min_confidence)
        if cruise_speed is not None:
            self.cruise_speed = int(cruise_speed)

        if not self.config.enabled:
            return None
        if self.active:
            return self._avoidance_command(now)
        return self._lane_follow_step(now)

    def _lidar_range(self, scan: np.ndarray | None) -> float | None:
        return min_distance_mm_in_ros_sector(
            scan,
            self.config.lidar_min_ros_deg,
            self.config.lidar_max_ros_deg,
            self.config.raw_angle_for_ros_zero_deg,
        )

    def _lane_follow_step(self, now_s: float) -> DriveCommand | None:
        if now_s < self.cooldown_until_s:
            self.trigger_counter = 0
            return None

        if self._obstacle_present():
            self.trigger_counter += 1
        else:
            self.trigger_counter = 0
        if self.trigger_counter < max(1, self.config.trigger_frames):
            return None
        return self._begin_shift_out(now_s)

    def _obstacle_present(self) -> bool:
        # A car in the central camera region with a near LiDAR return straight
        # ahead, in a known lane.  This is the trigger condition without the
        # consecutive-frame counter.
        return (
            self.current_lane in (1, 2)
            and self.last_car.detected
            and self._lidar_is_near()
        )

    def _begin_shift_out(self, now_s: float) -> DriveCommand:
        self.trigger_counter = 0
        self.phase = ContestObstaclePhase.AVOID_SHIFT_OUT
        self.phase_started_s = now_s
        self.avoidance_source_lane = self.current_lane
        self.active_shift_steer = (
            self.config.lane2_shift_steer
            if self.avoidance_source_lane == 2
            else self.config.lane1_shift_steer
        )
        # Extend the shift-out time when the car is already steered hard at
        # detection (e.g. mid-curve), so the maneuver still clears the lane.
        self.active_shift_out_duration_s = max(
            0.0,
            self.config.shift_out_duration_s
            + abs(self.latest_lane_follow_steer) * self.config.shift_out_steer_weight,
        )
        return self._command(self.active_shift_steer, "obstacle_avoid_shift_out")

    def _avoidance_command(self, now_s: float) -> DriveCommand | None:
        if self.phase == ContestObstaclePhase.AVOID_SHIFT_OUT:
            duration = max(0.0, self.active_shift_out_duration_s)
            if now_s - self.phase_started_s < duration:
                return self._command(self.active_shift_steer, "obstacle_avoid_shift_out")
            self.phase = ContestObstaclePhase.AVOID_PASS
            self.phase_started_s += duration

        if self.phase == ContestObstaclePhase.AVOID_PASS:
            timed_out = now_s - self.phase_started_s >= max(0.0, self.config.pass_max_duration_s)
            if not (self._heading_aligned_with_target_lane() or timed_out):
                return self._command(self._pass_steer(), "obstacle_avoid_pass")
            return self._finish_or_rechain(now_s)
        return None

    def _finish_or_rechain(self, now_s: float) -> DriveCommand | None:
        # Obstacle avoidance stays top priority.  If another obstacle is still
        # ahead when the pass completes, chain straight into a fresh shift-out
        # from the lane we just entered instead of handing even one frame back
        # to the lane follower, which would jerk the wheel toward the lane
        # center before the next avoidance re-triggers.
        if self._obstacle_present():
            return self._begin_shift_out(now_s)
        self._finish_avoidance(now_s)
        return None

    def _pass_steer(self) -> int:
        # Counter-steer opposite the shift, selected by the lane the maneuver
        # started from (locked in avoidance_source_lane, not the live detection).
        return (
            self.config.lane2_pass_steer
            if self.avoidance_source_lane == 2
            else self.config.lane1_pass_steer
        )

    def _target_lane(self) -> int | None:
        # The lane we are shifting into: the opposite of the source lane.
        if self.avoidance_source_lane == 2:
            return 1
        if self.avoidance_source_lane == 1:
            return 2
        return None

    def _heading_aligned_with_target_lane(self) -> bool:
        # Counter-steering ends once vision reports we are in the target lane and
        # the followed lane's heading is within the threshold of straight ahead,
        # i.e. the car's heading matches the new lane's centerline.
        if self.current_lane != self._target_lane():
            return False
        if self.latest_heading_deg is None:
            return False
        return abs(self.latest_heading_deg) <= abs(self.config.pass_align_heading_deg)

    def _finish_avoidance(self, now_s: float) -> None:
        # After the counter-steer pass phase there is no further steering back to
        # the source lane; normal control resumes on the closest lane area.
        self.phase = ContestObstaclePhase.LANE_FOLLOW
        self.trigger_counter = 0
        self.avoidance_source_lane = None
        self.active_shift_steer = 0
        self.cooldown_until_s = now_s + max(0.0, self.config.retrigger_cooldown_s)

    def _lidar_is_near(self) -> bool:
        return (
            self.latest_lidar_range_mm is not None
            and self.latest_lidar_range_mm <= self.config.lidar_obstacle_distance_mm
        )

    def _command(self, steering: int, reason: str) -> DriveCommand:
        # Avoidance steering intentionally bypasses the lane-follow MAX_STEER
        # limit and follows the tuned shift/pass values exactly.  The only cap
        # applied downstream is the Arduino protocol limit (+/-100) in
        # ArduinoMotorController.send.
        return DriveCommand(
            steer=int(steering),
            speed=self.cruise_speed,
            reason=reason,
        )


def _contest_lane_number(value: int | str | None) -> int | None:
    if isinstance(value, int):
        return value if value in (1, 2) else None
    normalized = str(value or "").lower().strip()
    if normalized in {"1", "lane1"}:
        return 1
    if normalized in {"2", "lane2"}:
        return 2
    # Composite labels such as "lane2 target=lane1" or
    # "mode=lane_area target=lane2 points=…" carry the driving lane in a
    # token; a plain equality check on the whole string misses it and the
    # mission stays inert.  Prefer an explicit target=laneN token, then a
    # bare laneN token.
    tokens = normalized.split()
    for token in tokens:
        if token.startswith("target="):
            parsed = _contest_lane_number(token.split("=", 1)[1])
            if parsed is not None:
                return parsed
    # Bare digits inside a composite label (candidate indexes, counts) are
    # not lane numbers, so only accept explicit laneN tokens here.
    for token in tokens:
        if token == "lane1":
            return 1
        if token == "lane2":
            return 2
    return None


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
