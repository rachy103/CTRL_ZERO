from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from ctrl_zero.control import DriveCommand
from ctrl_zero.lidar import ObstacleDecision
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

    def open(self) -> None:
        if not self.config.enabled:
            return
        self.frame_dir.mkdir(parents=True, exist_ok=True)
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
                "lane_width_px",
                "nearest_front_mm",
                "lidar_speed_scale",
                "lidar_front_points",
                "image_file",
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
        obstacle: ObstacleDecision | None,
        command: DriveCommand,
    ) -> None:
        if not self.config.enabled or self.writer is None:
            return
        self.index += 1
        image_file = ""
        if self.index % max(self.config.save_every_n_frames, 1) == 0:
            image_file = f"frame_{self.index:06d}.jpg"
            cv2.imwrite(str(self.frame_dir / image_file), frame)

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
                "lane_width_px": "" if lane.lane_width_px is None else f"{lane.lane_width_px:.3f}",
                "nearest_front_mm": "" if obstacle is None or obstacle.nearest_front_mm is None else f"{obstacle.nearest_front_mm:.1f}",
                "lidar_speed_scale": "" if obstacle is None else f"{obstacle.speed_scale:.3f}",
                "lidar_front_points": "" if obstacle is None else obstacle.front_points,
                "image_file": image_file,
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
