from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ctrl_zero.lidar import ObstacleDecision
from ctrl_zero.perception import DetectedObject
from ctrl_zero.traffic_light import (
    TRAFFIC_LIGHT_STOP_STATES,
    TRAFFIC_LIGHT_UNKNOWN,
    traffic_light_area_ratio,
    traffic_light_meets_stop_size,
)


@dataclass(frozen=True)
class SafetyDecision:
    nearest_front_mm: float | None = None
    speed_scale: float = 1.0
    should_stop: bool = False
    front_points: int = 0
    reason: str = "clear"
    lidar: ObstacleDecision | None = None
    traffic_light_state: str = TRAFFIC_LIGHT_UNKNOWN
    traffic_light_object: DetectedObject | None = None
    traffic_light_area_ratio: float | None = None
    vision_obstacle: DetectedObject | None = None
    avoidance_steer: float = 0.0
    current_lane_label: str = ""
    target_lane_label: str = ""

    @classmethod
    def clear(cls) -> "SafetyDecision":
        return cls()


def safety_from_lidar(obstacle: ObstacleDecision | None) -> SafetyDecision:
    if obstacle is None:
        return SafetyDecision.clear()
    if obstacle.should_stop:
        reason = "lidar_stop"
    elif obstacle.speed_scale < 1.0:
        reason = "lidar_slow"
    else:
        reason = "clear"
    return SafetyDecision(
        nearest_front_mm=obstacle.nearest_front_mm,
        speed_scale=obstacle.speed_scale,
        should_stop=obstacle.should_stop,
        front_points=obstacle.front_points,
        reason=reason,
        lidar=obstacle,
    )


def safety_from_traffic_light(
    state: str,
    obj: DetectedObject | None = None,
    stop_states: Iterable[str] = TRAFFIC_LIGHT_STOP_STATES,
    frame_area: float | None = None,
    min_stop_area_ratio: float = 0.0,
) -> SafetyDecision:
    normalized = state.lower().strip() if state else TRAFFIC_LIGHT_UNKNOWN
    stop_state = normalized in {item.lower().strip() for item in stop_states}
    should_stop = stop_state and traffic_light_meets_stop_size(obj, frame_area, min_stop_area_ratio)
    return SafetyDecision(
        speed_scale=0.0 if should_stop else 1.0,
        should_stop=should_stop,
        reason="traffic_light_stop" if should_stop else "clear",
        traffic_light_state=normalized,
        traffic_light_object=obj,
        traffic_light_area_ratio=traffic_light_area_ratio(obj, frame_area) if frame_area is not None else None,
    )


def fuse_safety_decisions(*decisions: SafetyDecision | None) -> SafetyDecision:
    active = [decision for decision in decisions if decision is not None]
    if not active:
        return SafetyDecision.clear()

    lidar_decision = next((decision for decision in active if decision.lidar is not None), None)
    traffic_decision = next(
        (decision for decision in active if decision.traffic_light_state != TRAFFIC_LIGHT_UNKNOWN),
        None,
    )
    vision_obstacle = next((decision.vision_obstacle for decision in active if decision.vision_obstacle is not None), None)
    avoidance_decision = next(
        (decision for decision in active if decision.avoidance_steer != 0.0 or decision.target_lane_label),
        None,
    )

    stop_decisions = [decision for decision in active if decision.should_stop]
    if stop_decisions:
        selected = stop_decisions[0]
        return SafetyDecision(
            nearest_front_mm=lidar_decision.nearest_front_mm if lidar_decision is not None else None,
            speed_scale=0.0,
            should_stop=True,
            front_points=lidar_decision.front_points if lidar_decision is not None else 0,
            reason=selected.reason,
            lidar=lidar_decision.lidar if lidar_decision is not None else None,
            traffic_light_state=traffic_decision.traffic_light_state if traffic_decision is not None else TRAFFIC_LIGHT_UNKNOWN,
            traffic_light_object=traffic_decision.traffic_light_object if traffic_decision is not None else None,
            traffic_light_area_ratio=traffic_decision.traffic_light_area_ratio if traffic_decision is not None else None,
            vision_obstacle=vision_obstacle,
            avoidance_steer=avoidance_decision.avoidance_steer if avoidance_decision is not None else 0.0,
            current_lane_label=avoidance_decision.current_lane_label if avoidance_decision is not None else "",
            target_lane_label=avoidance_decision.target_lane_label if avoidance_decision is not None else "",
        )

    selected = min(active, key=lambda decision: decision.speed_scale)
    if selected.speed_scale >= 1.0:
        selected = avoidance_decision or next((decision for decision in active if decision.reason != "clear"), selected)
    reason = selected.reason if selected.speed_scale < 1.0 or selected.avoidance_steer != 0.0 or selected.target_lane_label else "clear"
    return SafetyDecision(
        nearest_front_mm=lidar_decision.nearest_front_mm if lidar_decision is not None else None,
        speed_scale=selected.speed_scale,
        should_stop=False,
        front_points=lidar_decision.front_points if lidar_decision is not None else 0,
        reason=reason,
        lidar=lidar_decision.lidar if lidar_decision is not None else None,
        traffic_light_state=traffic_decision.traffic_light_state if traffic_decision is not None else TRAFFIC_LIGHT_UNKNOWN,
        traffic_light_object=traffic_decision.traffic_light_object if traffic_decision is not None else None,
        traffic_light_area_ratio=traffic_decision.traffic_light_area_ratio if traffic_decision is not None else None,
        vision_obstacle=vision_obstacle,
        avoidance_steer=avoidance_decision.avoidance_steer if avoidance_decision is not None else 0.0,
        current_lane_label=avoidance_decision.current_lane_label if avoidance_decision is not None else "",
        target_lane_label=avoidance_decision.target_lane_label if avoidance_decision is not None else "",
    )


def build_safety_decision(
    lidar: ObstacleDecision | None = None,
    traffic_light_state: str = TRAFFIC_LIGHT_UNKNOWN,
    traffic_light_object: DetectedObject | None = None,
    traffic_light_frame_area: float | None = None,
    traffic_light_min_stop_area_ratio: float = 0.0,
    vision_obstacle_decision: SafetyDecision | None = None,
) -> SafetyDecision:
    return fuse_safety_decisions(
        safety_from_traffic_light(
            traffic_light_state,
            traffic_light_object,
            frame_area=traffic_light_frame_area,
            min_stop_area_ratio=traffic_light_min_stop_area_ratio,
        ),
        vision_obstacle_decision,
        safety_from_lidar(lidar),
    )
