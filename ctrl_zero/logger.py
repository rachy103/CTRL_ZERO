from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from ctrl_zero.control import DriveCommand
from ctrl_zero.lidar import ObstacleDecision
from ctrl_zero.safety import SafetyDecision
from ctrl_zero.vision.base import LaneDetection


@dataclass
class LogConfig:
    enabled: bool = False
    directory: Path = Path("lane_logs")
    save_every_n_frames: int = 5


class DriveLogger:
    def __init__(self, config: LogConfig):
        self.config = config
        self.index = 0
        self.csv_file = None
        self.writer = None
        self.frame_dir = config.directory / "frames"
        self.annotated_dir = config.directory / "annotated"
        self.mask_dir = config.directory / "masks"

    def open(self) -> None:
        if not self.config.enabled:
            return
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        self.csv_file = (self.config.directory / "drive_log.csv").open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "timestamp",
                "frame",
                "mode",
                "backend",
                "steer",
                "speed",
                "reason",
                "confidence",
                "offset_px",
                "offset_norm",
                "heading_deg",
                "curvature_1_per_px",
                "lane_width_px",
                "lane_pair",
                "traffic_light_state",
                "object_count",
                "safety_reason",
                "vision_obstacle_class",
                "vision_obstacle_confidence",
                "nearest_front_mm",
                "lidar_speed_scale",
                "lidar_front_points",
                "image_file",
                "annotated_file",
                "mask_file",
            ],
        )
        self.writer.writeheader()
        print(f"Logging enabled: {self.config.directory / 'drive_log.csv'}")

    def log(
        self,
        frame,
        mode: str,
        backend: str,
        lane: LaneDetection,
        obstacle: ObstacleDecision | SafetyDecision | None,
        command: DriveCommand,
    ) -> None:
        if not self.config.enabled or self.writer is None:
            return
        self.index += 1
        image_file = ""
        annotated_file = ""
        mask_file = ""
        if self.index % max(self.config.save_every_n_frames, 1) == 0:
            image_file = f"frame_{self.index:06d}.jpg"
            annotated_file = f"annotated_{self.index:06d}.jpg"
            cv2.imwrite(str(self.frame_dir / image_file), frame)
            cv2.imwrite(str(self.annotated_dir / annotated_file), lane.annotated)
            if lane.mask is not None:
                mask_file = f"mask_{self.index:06d}.png"
                cv2.imwrite(str(self.mask_dir / mask_file), lane.mask)

        vision_obstacle = obstacle.vision_obstacle if isinstance(obstacle, SafetyDecision) else None
        self.writer.writerow(
            {
                "timestamp": f"{time.time():.6f}",
                "frame": self.index,
                "mode": mode,
                "backend": backend,
                "steer": command.steer,
                "speed": command.speed,
                "reason": command.reason,
                "confidence": f"{lane.confidence:.3f}",
                "offset_px": "" if lane.offset_px is None else f"{lane.offset_px:.3f}",
                "offset_norm": "" if lane.offset_norm is None else f"{lane.offset_norm:.5f}",
                "heading_deg": "" if lane.heading_deg is None else f"{lane.heading_deg:.3f}",
                "curvature_1_per_px": f"{lane.curvature:.8f}",
                "lane_width_px": "" if lane.lane_width_px is None else f"{lane.lane_width_px:.3f}",
                "lane_pair": lane.lane_pair_label,
                "traffic_light_state": lane.traffic_light_state,
                "object_count": len(lane.objects),
                "safety_reason": "" if obstacle is None else getattr(obstacle, "reason", ""),
                "vision_obstacle_class": "" if vision_obstacle is None else vision_obstacle.class_name,
                "vision_obstacle_confidence": "" if vision_obstacle is None else f"{vision_obstacle.confidence:.3f}",
                "nearest_front_mm": "" if obstacle is None or obstacle.nearest_front_mm is None else f"{obstacle.nearest_front_mm:.1f}",
                "lidar_speed_scale": "" if obstacle is None else f"{obstacle.speed_scale:.3f}",
                "lidar_front_points": "" if obstacle is None else obstacle.front_points,
                "image_file": image_file,
                "annotated_file": annotated_file,
                "mask_file": mask_file,
            }
        )
        self.csv_file.flush()

    def close(self) -> None:
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.writer = None

    def set_enabled(self, enabled: bool) -> None:
        self.config.enabled = enabled
        if enabled and self.writer is None:
            self.open()
