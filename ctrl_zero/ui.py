from __future__ import annotations

import cv2

from ctrl_zero.common import format_optional
from ctrl_zero.control import DriveCommand
from ctrl_zero.lidar import ObstacleDecision
from ctrl_zero.safety import SafetyDecision
from ctrl_zero.vision.base import LaneDetection


def draw_status(
    frame,
    lane: LaneDetection,
    obstacle: ObstacleDecision | SafetyDecision | None,
    command: DriveCommand,
    mode: str,
    backend: str,
    fps: float,
    motor_enabled: bool,
) -> None:
    lidar_text = "off"
    if obstacle is not None:
        nearest = "NA" if obstacle.nearest_front_mm is None else f"{obstacle.nearest_front_mm:.0f}mm"
        lidar_text = f"{nearest} scale={obstacle.speed_scale:.2f}"

    vision_text = "none"
    if isinstance(obstacle, SafetyDecision) and obstacle.vision_obstacle is not None:
        object_lane = f" {obstacle.vision_obstacle.lane_label}" if obstacle.vision_obstacle.lane_label else ""
        vision_text = f"{obstacle.vision_obstacle.class_name}{object_lane} {obstacle.vision_obstacle.confidence:.2f}"
    avoid_text = "0"
    if isinstance(obstacle, SafetyDecision) and obstacle.avoidance_steer != 0.0:
        avoid_text = f"{obstacle.avoidance_steer:+.0f}->{obstacle.target_lane_label}"
    elif isinstance(obstacle, SafetyDecision) and obstacle.target_lane_label:
        avoid_text = f"path->{obstacle.target_lane_label}"
    traffic_ratio = "NA"
    if isinstance(obstacle, SafetyDecision) and obstacle.traffic_light_area_ratio is not None:
        traffic_ratio = f"{obstacle.traffic_light_area_ratio:.3f}"

    lines = [
        f"mode={mode} backend={backend} fps={fps:.1f} motor={'on' if motor_enabled else 'dry'}",
        f"steer={command.steer:+d} speed={command.speed:+d} reason={command.reason}",
        f"conf={lane.confidence:.2f} offset={format_optional(lane.offset_norm, 3)} heading={format_optional(lane.heading_deg, 1)} kappa={lane.curvature:+.6f}",
        f"lanes={len(lane.lanes)} current_lane={lane.lane_label or 'NA'} width_px={format_optional(lane.lane_width_px, 1)} lidar={lidar_text}",
        f"traffic={lane.traffic_light_state} traffic_area={traffic_ratio} objects={len(lane.objects)} obstacle={vision_text} avoid={avoid_text} safety={getattr(obstacle, 'reason', 'clear') if obstacle is not None else 'off'}",
    ]
    y = 24
    for text in lines:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
        y += 24
