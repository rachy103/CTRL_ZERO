from __future__ import annotations

import math
from dataclasses import dataclass

from ctrl_zero.common import clamp
from ctrl_zero.lidar import ObstacleDecision
from ctrl_zero.safety import SafetyDecision
from ctrl_zero.vision.base import LaneDetection


@dataclass
class DriveConfig:
    control_mode: str = "contest"
    min_speed: int = 150
    max_speed: int = 255
    contest_angle_weight: float = 0.7
    contest_position_weight: float = 0.05
    contest_angle_norm_deg: float = 50.0
    contest_steer_limit: float = 10.0
    curve_speed_gain: float = 3.0
    kappa_ref: float = 0.0015
    wheelbase_px: float = 220.0
    k_stanley: float = 1.6
    heading_gain: float = 1.0
    ff_gain: float = 0.0
    steer_scale: float = 100.0
    max_steer: int = 100
    reverse_steer: bool = False
    steer_slew_base: float = 18.0
    steer_slew_min: float = 8.0
    min_confidence: float = 0.45
    max_hold_frames: int = 6
    hold_decel_step: int = 6


@dataclass
class DriveCommand:
    steer: int
    speed: int
    reason: str


class DriveController:
    """Contest lane follower by default, with Stanley available as a fallback."""

    def __init__(self, config: DriveConfig):
        self.config = config
        self.prev_steer = 0.0
        self.prev_speed = 0.0
        self.lost_frames = 0

    def compute(
        self,
        lane: LaneDetection,
        obstacle: ObstacleDecision | SafetyDecision | None,
        mode: str,
        manual_steer: int = 0,
        manual_speed: int = 0,
    ) -> DriveCommand:
        if mode == "manual":
            return self._manual_command(manual_steer, manual_speed, obstacle)
        if mode == "vision":
            return DriveCommand(steer=0, speed=0, reason="vision_only")

        lane_ok = lane.offset_norm is not None and lane.confidence >= self.config.min_confidence
        if not lane_ok:
            return self._handle_lost(obstacle)

        self.lost_frames = 0
        if self.config.control_mode.lower().strip() == "stanley":
            return self._compute_stanley(lane, obstacle)
        return self._compute_contest(lane, obstacle)

    def _compute_contest(self, lane: LaneDetection, obstacle: ObstacleDecision | SafetyDecision | None) -> DriveCommand:
        steering_angle = lane.heading_deg or 0.0
        vehicle_position_x = -(lane.offset_px or 0.0)

        angle_norm = max(abs(self.config.contest_angle_norm_deg), 1e-9)
        steer_limit = max(abs(self.config.contest_steer_limit), 1e-9)
        mapped = (steering_angle / angle_norm) * steer_limit * self.config.contest_angle_weight
        adjust = -vehicle_position_x * self.config.contest_position_weight
        contest_steer = clamp(mapped + adjust, -steer_limit, steer_limit)

        steer_raw = (contest_steer / steer_limit) * self.config.max_steer
        steer_raw = self._apply_avoidance_steer(steer_raw, obstacle)
        if self.config.reverse_steer:
            steer_raw = -steer_raw
        steer = int(round(clamp(steer_raw, -self.config.max_steer, self.config.max_steer)))

        speed_target = int(clamp(self.config.max_speed, self.config.min_speed, self.config.max_speed))
        self.prev_steer = steer
        self.prev_speed = speed_target
        speed, reason = self._apply_obstacle(speed_target, obstacle, "contest")
        return DriveCommand(steer=steer, speed=speed, reason=reason)

    def _compute_stanley(self, lane: LaneDetection, obstacle: ObstacleDecision | SafetyDecision | None) -> DriveCommand:
        e_y = lane.offset_norm
        e_psi = math.radians(lane.heading_deg or 0.0)
        kappa = lane.curvature or 0.0

        kappa_norm = abs(kappa) / max(self.config.kappa_ref, 1e-9)
        speed_target = self.config.max_speed / (1.0 + self.config.curve_speed_gain * kappa_norm)
        speed_target = clamp(speed_target, self.config.min_speed, self.config.max_speed)

        v_norm = max(self.prev_speed / max(self.config.max_speed, 1), 0.15)
        delta_ff = self.config.ff_gain * math.atan(self.config.wheelbase_px * kappa)
        delta_fb = self.config.heading_gain * e_psi + math.atan(self.config.k_stanley * e_y / v_norm)
        steer_raw = clamp((delta_ff + delta_fb) * self.config.steer_scale, -self.config.max_steer, self.config.max_steer)
        if self.config.reverse_steer:
            steer_raw = -steer_raw

        slew = self._slew_limit(speed_target)
        steer = clamp(steer_raw, self.prev_steer - slew, self.prev_steer + slew)
        self.prev_steer = steer

        speed = self._ramp_speed(speed_target)
        self.prev_speed = speed

        speed_i, reason = self._apply_obstacle(int(round(speed)), obstacle, "auto")
        return DriveCommand(steer=int(round(steer)), speed=speed_i, reason=reason)

    def _slew_limit(self, speed_target: float) -> float:
        ratio = clamp(speed_target / max(self.config.max_speed, 1), 0.0, 1.0)
        return self.config.steer_slew_base - (self.config.steer_slew_base - self.config.steer_slew_min) * ratio

    def _ramp_speed(self, target: float) -> float:
        delta = clamp(target - self.prev_speed, -8.0, 4.0)
        return self.prev_speed + delta

    def _apply_avoidance_steer(
        self,
        steer_raw: float,
        obstacle: ObstacleDecision | SafetyDecision | None,
    ) -> float:
        if not isinstance(obstacle, SafetyDecision) or obstacle.avoidance_steer == 0.0:
            return steer_raw
        return clamp(steer_raw + obstacle.avoidance_steer, -self.config.max_steer, self.config.max_steer)

    def _handle_lost(self, obstacle: ObstacleDecision | SafetyDecision | None) -> DriveCommand:
        self.lost_frames += 1
        if self.lost_frames <= self.config.max_hold_frames:
            speed = max(0.0, self.prev_speed - self.config.hold_decel_step)
            self.prev_speed = speed
            speed_i, reason = self._apply_obstacle(int(round(speed)), obstacle, "lane_hold")
            return DriveCommand(steer=int(round(self.prev_steer)), speed=speed_i, reason=reason)

        self.prev_speed = 0.0
        self.prev_steer = 0.0
        return DriveCommand(steer=0, speed=0, reason="lane_lost")

    def _manual_command(
        self,
        manual_steer: int,
        manual_speed: int,
        obstacle: ObstacleDecision | SafetyDecision | None,
    ) -> DriveCommand:
        steer = int(clamp(manual_steer, -self.config.max_steer, self.config.max_steer))
        speed = int(clamp(manual_speed, -self.config.max_speed, self.config.max_speed))
        self.prev_steer = steer
        self.prev_speed = speed
        speed, reason = self._apply_obstacle(speed, obstacle, "manual")
        return DriveCommand(steer=steer, speed=speed, reason=reason)

    @staticmethod
    def _apply_obstacle(speed: int, obstacle: ObstacleDecision | SafetyDecision | None, reason: str) -> tuple[int, str]:
        if obstacle is None:
            return speed, reason
        if obstacle.should_stop:
            return 0, obstacle.reason if isinstance(obstacle, SafetyDecision) else "lidar_stop"
        scaled = int(round(speed * obstacle.speed_scale))
        if isinstance(obstacle, SafetyDecision) and obstacle.reason != "clear":
            return scaled, obstacle.reason
        if obstacle.speed_scale >= 1.0:
            return scaled, reason
        return scaled, obstacle.reason if isinstance(obstacle, SafetyDecision) else "lidar_slow"
