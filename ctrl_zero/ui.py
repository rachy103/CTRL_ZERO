from __future__ import annotations

import cv2

from ctrl_zero.common import format_optional
from ctrl_zero.control import DriveCommand
from ctrl_zero.lidar import ObstacleDecision
from ctrl_zero.vision.base import LaneDetection


def draw_status(
    frame,
    lane: LaneDetection,
    obstacle: ObstacleDecision | None,
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

    lines = [
        f"mode={mode} backend={backend} fps={fps:.1f} motor={'on' if motor_enabled else 'dry'}",
        f"steer={command.steer:+d} speed={command.speed:+d} reason={command.reason}",
        f"conf={lane.confidence:.2f} offset={format_optional(lane.offset_norm, 3)} heading={format_optional(lane.heading_deg, 1)}",
        f"lanes={len(lane.lanes)} width_px={format_optional(lane.lane_width_px, 1)} lidar={lidar_text}",
    ]
    y = 24
    for text in lines:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
