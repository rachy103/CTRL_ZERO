from __future__ import annotations

from dataclasses import dataclass

from ctrl_zero.common import clamp
from ctrl_zero.lidar import ObstacleDecision
from ctrl_zero.vision.base import LaneDetection


@dataclass
class DriveConfig:
    base_speed: int = 45
    max_speed: int = 80
    min_confidence: float = 0.45
    kp_offset: float = 78.0
    kp_heading: float = 32.0
    kd_offset: float = 18.0
    heading_norm_deg: float = 28.0
    max_steer: int = 100
    reverse_steer: bool = False
    curve_speed_reduction: float = 0.55


@dataclass
class DriveCommand:
    steer: int
    speed: int
    reason: str


class DriveController:
    def __init__(self, config: DriveConfig):
        self.config = config
        self.previous_offset_norm = 0.0

    def compute(
        self,
        lane: LaneDetection,
        obstacle: ObstacleDecision | None,
        mode: str,
        manual_steer: int = 0,
        manual_speed: int = 0,
    ) -> DriveCommand:
        if mode == "manual":
            return self._manual_command(manual_steer, manual_speed, obstacle)
        if mode == "vision":
            return DriveCommand(steer=0, speed=0, reason="vision_only")

        if lane.offset_norm is None:
            return DriveCommand(steer=0, speed=0, reason="lane_center_missing")
        if lane.confidence < self.config.min_confidence:
            return DriveCommand(steer=0, speed=0, reason="low_lane_confidence")

        heading_deg = lane.heading_deg if lane.heading_deg is not None else 0.0
        offset_delta = lane.offset_norm - self.previous_offset_norm
        self.previous_offset_norm = lane.offset_norm

        steer_float = (
            self.config.kp_offset * lane.offset_norm
            + self.config.kp_heading * (heading_deg / max(self.config.heading_norm_deg, 1.0))
            + self.config.kd_offset * offset_delta
        )
        steer = int(round(clamp(steer_float, -self.config.max_steer, self.config.max_steer)))
        if self.config.reverse_steer:
            steer = -steer

        speed = self._curve_speed(steer)
        speed, reason = self._apply_obstacle(speed, obstacle, "auto")
        return DriveCommand(steer=steer, speed=speed, reason=reason)

    def _manual_command(self, manual_steer: int, manual_speed: int, obstacle: ObstacleDecision | None) -> DriveCommand:
        steer = int(clamp(manual_steer, -self.config.max_steer, self.config.max_steer))
        speed = int(clamp(manual_speed, -self.config.max_speed, self.config.max_speed))
        speed, reason = self._apply_obstacle(speed, obstacle, "manual")
        return DriveCommand(steer=steer, speed=speed, reason=reason)

    def _curve_speed(self, steer: int) -> int:
        reduction = min(abs(steer) / max(self.config.max_steer, 1) * self.config.curve_speed_reduction, 0.90)
        return int(round(clamp(self.config.base_speed * (1.0 - reduction), 0, self.config.max_speed)))

    @staticmethod
    def _apply_obstacle(speed: int, obstacle: ObstacleDecision | None, reason: str) -> tuple[int, str]:
        if obstacle is None:
            return speed, reason
        if obstacle.should_stop:
            return 0, "lidar_stop"
        return int(round(speed * obstacle.speed_scale)), reason if obstacle.speed_scale >= 1.0 else "lidar_slow"
