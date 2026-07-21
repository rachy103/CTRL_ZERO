from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Sequence

import cv2
import numpy as np

from ctrl_zero.common import clamp
from ctrl_zero.control import DriveCommand
from ctrl_zero.lidar import average_distance_mm_in_ros_sector
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
    MONITOR_LANE2 = "monitor_lane2"
    CHANGE_TO_LANE1 = "change_to_lane1"
    STABILIZE_LANE1 = "stabilize_lane1"
    FOLLOW_LANE1 = "follow_lane1"
    CHANGE_TO_LANE2 = "change_to_lane2"
    STABILIZE_LANE2 = "stabilize_lane2"
    COMPLETE = "complete"


@dataclass(frozen=True)
class ContestObstacleMissionConfig:
    enabled: bool = True
    camera_min_confidence: float = 0.50
    lidar_obstacle_distance_mm: float = 1000.0
    lane2_lidar_min_ros_deg: float = 87.5
    lane2_lidar_max_ros_deg: float = 92.5
    lane1_lidar_min_ros_deg: float = -93.0
    lane1_lidar_max_ros_deg: float = -90.0
    raw_angle_for_ros_zero_deg: float = 180.0
    lane2_obstacle_frames: int = 5
    lane1_obstacle_frames: int = 3
    lane1_clear_frames: int = 3
    lane_change_complete_position_px: float = 30.0
    stabilization_frames: int = 3
    small_area_threshold: float = 5000.0
    medium_area_threshold: float = 12000.0
    large_area_threshold: float = 15000.0
    lane1_no_car_speed: int = 50
    lane1_forward_speed: int = 100
    lane1_reverse_speed: int = -100
    normal_speed: int = 170
    change_to_lane1_speed: int = 150
    change_to_lane2_speed: int = 170
    change_to_lane1_steering: float = -10.0
    change_to_lane2_steering: float = 7.0
    source_steering_limit: float = 10.0
    drive_steering_limit: int = 80
    heading_norm_deg: float = 50.0
    angle_weight: float = 0.7
    position_weight: float = 0.05


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
    """Reproduce obstacle.py: first car, central 50%, bbox width * height."""
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
        return ContestCarObservation(False, 0.0, obj)
    return ContestCarObservation(False, 0.0, None)


def source_steering_to_drive(
    source_steering: float,
    drive_steering_limit: int,
    source_steering_limit: float = 10.0,
) -> int:
    """Map the source MotionCommand [-10, 10] range to DriveCommand."""
    source_limit = max(abs(source_steering_limit), 1e-9)
    source_value = clamp(source_steering, -source_limit, source_limit)
    mapped = source_value / source_limit * abs(drive_steering_limit)
    return int(round(clamp(mapped, -abs(drive_steering_limit), abs(drive_steering_limit))))


class ContestObstacleMission:
    """Frame-driven port of motion_mission.py's obstacle state transitions."""

    def __init__(self, config: ContestObstacleMissionConfig | None = None):
        self.config = config or ContestObstacleMissionConfig()
        self.phase = ContestObstaclePhase.MONITOR_LANE2
        self.current_lane: int | None = None
        self.target_lane = 2
        self.latest_lidar_avg_mm: float | None = None
        self.last_car = ContestCarObservation(False, 0.0, None)
        self.lidar_lane_change_counter = 0
        self.lidar_lane_change_executed = False
        self.lidar_obstacle_counter = 0
        self.obstacle_flag = False
        self.no_obstacle_counter = 0
        self.steering_stable_count = 0

    @property
    def active(self) -> bool:
        return self.phase not in (ContestObstaclePhase.MONITOR_LANE2, ContestObstaclePhase.COMPLETE)

    @property
    def is_changing_lane(self) -> bool:
        return self.phase in (ContestObstaclePhase.CHANGE_TO_LANE1, ContestObstaclePhase.CHANGE_TO_LANE2)

    @property
    def ignoring_obstacles(self) -> bool:
        return self.is_changing_lane or self.phase in (
            ContestObstaclePhase.STABILIZE_LANE1,
            ContestObstaclePhase.STABILIZE_LANE2,
        )

    def step(
        self,
        *,
        objects: Sequence[DetectedObject],
        current_lane: int | str | None,
        heading_deg: float | None,
        vehicle_position_x: float | None,
        frame_width: int,
        lidar_scan: np.ndarray | None,
    ) -> DriveCommand | None:
        self.current_lane = _contest_lane_number(current_lane)
        self.latest_lidar_avg_mm = self._lidar_average(lidar_scan)
        self.last_car = detect_contest_car(objects, frame_width, self.config.camera_min_confidence)

        if not self.config.enabled or self.phase == ContestObstaclePhase.COMPLETE:
            return None
        if self.phase == ContestObstaclePhase.MONITOR_LANE2:
            return self._monitor_lane2()
        if self.phase == ContestObstaclePhase.CHANGE_TO_LANE1:
            return self._change_lane(1, vehicle_position_x)
        if self.phase == ContestObstaclePhase.STABILIZE_LANE1:
            return self._stabilize(1, heading_deg, vehicle_position_x)
        if self.phase == ContestObstaclePhase.FOLLOW_LANE1:
            return self._follow_lane1(heading_deg, vehicle_position_x)
        if self.phase == ContestObstaclePhase.CHANGE_TO_LANE2:
            return self._change_lane(2, vehicle_position_x)
        if self.phase == ContestObstaclePhase.STABILIZE_LANE2:
            return self._stabilize(2, heading_deg, vehicle_position_x)
        return None

    def _lidar_average(self, scan: np.ndarray | None) -> float | None:
        if self.current_lane == 2:
            start = self.config.lane2_lidar_min_ros_deg
            end = self.config.lane2_lidar_max_ros_deg
        else:
            # This is also the source fallback when lane information is absent.
            start = self.config.lane1_lidar_min_ros_deg
            end = self.config.lane1_lidar_max_ros_deg
        return average_distance_mm_in_ros_sector(
            scan,
            start,
            end,
            self.config.raw_angle_for_ros_zero_deg,
        )

    def _monitor_lane2(self) -> DriveCommand | None:
        if self.current_lane != 2 or self.lidar_lane_change_executed:
            return None
        if self._lidar_is_near():
            self.lidar_lane_change_counter += 1
        else:
            self.lidar_lane_change_counter = 0
        if self.lidar_lane_change_counter < max(1, self.config.lane2_obstacle_frames):
            return None

        self.target_lane = 1
        self.phase = ContestObstaclePhase.CHANGE_TO_LANE1
        self.lidar_lane_change_executed = True
        self.lidar_lane_change_counter = 0
        return self._lane_change_command(1)

    def _change_lane(self, target_lane: int, vehicle_position_x: float | None) -> DriveCommand:
        command = self._lane_change_command(target_lane)
        if (
            self.current_lane == target_lane
            and vehicle_position_x is not None
            and abs(vehicle_position_x) <= self.config.lane_change_complete_position_px
        ):
            self.steering_stable_count = 0
            self.phase = (
                ContestObstaclePhase.STABILIZE_LANE1
                if target_lane == 1
                else ContestObstaclePhase.STABILIZE_LANE2
            )
        return command

    def _stabilize(
        self,
        lane_number: int,
        heading_deg: float | None,
        vehicle_position_x: float | None,
    ) -> DriveCommand:
        source_steering = self._lane_follow_source_steering(heading_deg, vehicle_position_x)
        command = self._command(source_steering, self.config.normal_speed, f"obstacle_stabilize_lane{lane_number}")

        # motion_mission.py clamps steering to [-10, 10] immediately before
        # this <= 10 test, so its actual behavior is an unconditional 3 frames.
        if abs(source_steering) <= self.config.source_steering_limit:
            self.steering_stable_count += 1
        else:
            self.steering_stable_count = 0
        if self.steering_stable_count >= max(1, self.config.stabilization_frames):
            self.steering_stable_count = 0
            self.phase = ContestObstaclePhase.FOLLOW_LANE1 if lane_number == 1 else ContestObstaclePhase.COMPLETE
        return command

    def _follow_lane1(
        self,
        heading_deg: float | None,
        vehicle_position_x: float | None,
    ) -> DriveCommand:
        if self.current_lane != 1:
            return self._command(
                self._lane_follow_source_steering(heading_deg, vehicle_position_x),
                self.config.normal_speed,
                "obstacle_lane1_wait_for_lane",
            )

        source_steering = self._lane_follow_source_steering(heading_deg, vehicle_position_x)
        forward_motion = False
        backward_motion = False
        if not self.last_car.detected:
            command = self._command(source_steering, self.config.lane1_no_car_speed, "obstacle_lane1_no_car")
        elif self.last_car.area < self.config.small_area_threshold:
            command = self._command(source_steering, 0, "obstacle_lane1_small_stop")
        elif self.last_car.area < self.config.medium_area_threshold:
            forward_motion = True
            command = self._command(source_steering, self.config.lane1_forward_speed, "obstacle_lane1_forward")
        elif self.last_car.area < self.config.large_area_threshold:
            command = self._command(source_steering, 0, "obstacle_lane1_large_stop")
        else:
            backward_motion = True
            command = self._command(-source_steering, self.config.lane1_reverse_speed, "obstacle_lane1_reverse")

        self._update_lane1_lidar(forward_motion, backward_motion)
        return command

    def _update_lane1_lidar(self, forward_motion: bool, backward_motion: bool) -> None:
        if backward_motion:
            self.lidar_obstacle_counter = 0
            return

        if self._lidar_is_near():
            self.lidar_obstacle_counter += 1
            if self.lidar_obstacle_counter >= max(1, self.config.lane1_obstacle_frames) and not self.obstacle_flag:
                self.obstacle_flag = True
        else:
            self.lidar_obstacle_counter = 0

        lidar_is_clear = (
            self.latest_lidar_avg_mm is not None
            and self.latest_lidar_avg_mm > self.config.lidar_obstacle_distance_mm
        )
        if self.obstacle_flag and forward_motion and lidar_is_clear:
            self.no_obstacle_counter += 1
            if self.no_obstacle_counter >= max(1, self.config.lane1_clear_frames):
                self.target_lane = 2
                self.phase = ContestObstaclePhase.CHANGE_TO_LANE2
                self.obstacle_flag = False
                self.no_obstacle_counter = 0
        elif not (self.obstacle_flag and forward_motion):
            # The source does not reset this counter for a near/invalid LiDAR
            # sample while obstacle_flag and forward_motion remain true.
            self.no_obstacle_counter = 0

    def _lidar_is_near(self) -> bool:
        return (
            self.latest_lidar_avg_mm is not None
            and self.latest_lidar_avg_mm < self.config.lidar_obstacle_distance_mm
        )

    def _lane_change_command(self, target_lane: int) -> DriveCommand:
        if target_lane == 1:
            return self._command(
                self.config.change_to_lane1_steering,
                self.config.change_to_lane1_speed,
                "obstacle_lane_change_2_to_1",
            )
        return self._command(
            self.config.change_to_lane2_steering,
            self.config.change_to_lane2_speed,
            "obstacle_lane_change_1_to_2",
        )

    def _lane_follow_source_steering(
        self,
        heading_deg: float | None,
        vehicle_position_x: float | None,
    ) -> int:
        heading = heading_deg or 0.0
        position = vehicle_position_x or 0.0
        angle_norm = max(abs(self.config.heading_norm_deg), 1e-9)
        mapped = (heading / angle_norm) * self.config.source_steering_limit * self.config.angle_weight
        adjust = -position * self.config.position_weight
        return int(clamp(mapped + adjust, -self.config.source_steering_limit, self.config.source_steering_limit))

    def _command(self, source_steering: float, speed: int, reason: str) -> DriveCommand:
        return DriveCommand(
            steer=source_steering_to_drive(
                source_steering,
                self.config.drive_steering_limit,
                self.config.source_steering_limit,
            ),
            speed=int(speed),
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
