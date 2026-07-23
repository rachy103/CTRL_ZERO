"""Frame-driven parallel-parking state machine (port of the ROS2 parking node).

The original was a ``rclpy`` node that reacted to ``/scan`` callbacks and
published ``MotionCommand(left_speed, right_speed, steering)``.  This port keeps
the exact state machine and transition logic but adapts it to CTRL_ZERO:

* No ROS.  ``step(scan, now_s)`` is called once per frame and returns a
  :class:`~ctrl_zero.control.DriveCommand` (``steer``, ``speed``, ``reason``).
* LiDAR scan is the CTRL_ZERO numpy array of ``[raw_angle_deg, distance_mm]``
  rows, and sectors are expressed in ROS angles (forward=0, left=+90, right=-90)
  via ``raw_angle_for_ros_zero_deg`` — the vehicle's confirmed convention
  (front=raw 180, left=raw 90, right=raw 270).
* Distances are millimetres (the ROS code used metres): 1.0 m -> 1000 mm, etc.
* The differential ``left_speed``/``right_speed`` of the source is collapsed to a
  single ``speed`` (this car steers with a servo, not by wheel differential);
  the per-phase speeds are config values so they can be retuned.

Every magic number from the source lives in :class:`ParkingHardConfig`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ctrl_zero.control import DriveCommand
from ctrl_zero.lidar import average_distance_mm_in_ros_sector


class ParkingHardState(Enum):
    SEARCH = 1              # drive forward, count the two parked cars
    FIRST_STOP = 2          # stop past the spot, pre-turn the wheels
    REVERSE_RIGHT = 3       # reverse into the spot with the wheels turned
    REVERSE_STRAIGHT = 4    # reverse straight, centering between both cars
    REVERSE_PAUSE = 5       # settle, stopped
    ADJUST_FORWARD = 6      # begin exit: drive forward
    GO_OUT_TURN = 7         # exit turn (to the right)
    GO_OUT_STRAIGHT = 8     # exit complete: drive straight


@dataclass(frozen=True)
class ParkingHardConfig:
    enabled: bool = True
    raw_angle_for_ros_zero_deg: float = 180.0
    # Sectors as (min_ros_deg, max_ros_deg).  Parked cars / spot are on the right.
    search_sector_ros_deg: tuple[float, float] = (-95.0, -85.0)   # right side
    left_sector_ros_deg: tuple[float, float] = (85.0, 95.0)
    right_sector_ros_deg: tuple[float, float] = (-95.0, -85.0)

    # Detection thresholds (mm) and debounce frame counts.
    search_detect_mm: float = 2000.0
    side_detect_mm: float = 1000.0
    go_out_detect_mm: float = 1500.0
    detect_frames: int = 3
    clear_frames: int = 3

    # Steering (CTRL_ZERO steer units; -1 == straight due to pot calibration).
    init_steer: int = 7
    straight_steer: int = -1
    first_stop_steer: int = 30
    reverse_right_steer: int = 7
    go_out_turn_steer: int = 30
    go_out_straight_steer: int = -1
    centering_deadband_mm: float = 400.0
    centering_steer: int = 3

    # Speeds (CTRL_ZERO speed units; + forward, - reverse).
    search_speed: int = 100
    reverse_right_speed: int = -50       # source (-70, -30) collapsed
    reverse_straight_speed: int = -70
    adjust_forward_speed: int = 70
    go_out_turn_speed: int = 35          # source (50, 20) collapsed
    go_out_straight_speed: int = 70

    # Durations (seconds).
    first_stop_seconds: float = 2.0
    reverse_pause_seconds: float = 4.0
    adjust_forward_min_seconds: float = 3.0
    go_out_turn_min_seconds: float = 25.0


@dataclass
class _SideTracker:
    detect: int = 0
    clear: int = 0
    state: str = "no_obstacle"  # -> "passing" -> "out"

    def update(self, distance_mm: float, threshold_mm: float) -> None:
        if distance_mm <= threshold_mm:
            self.detect += 1
            self.clear = 0
        else:
            self.clear += 1
            self.detect = 0


class ParkingHardMission:
    """Timed/LiDAR-driven parallel parking, independent of the lane follower."""

    def __init__(self, config: ParkingHardConfig | None = None):
        self.config = config or ParkingHardConfig()
        self.state = ParkingHardState.SEARCH
        self.state_start_s = 0.0

        # SEARCH car-counting state.
        self.obs_detect = 0
        self.obs_clear = 0
        self.obstacle_state = "no_obstacle"  # -> passed_first -> between -> passed_second

        self.left = _SideTracker()
        self.right = _SideTracker()

    @property
    def done(self) -> bool:
        return self.state == ParkingHardState.GO_OUT_STRAIGHT

    def step(self, scan: np.ndarray | None, now_s: float | None = None) -> DriveCommand | None:
        if not self.config.enabled:
            return None
        now = time.monotonic() if now_s is None else float(now_s)
        handler = {
            ParkingHardState.SEARCH: self._search,
            ParkingHardState.FIRST_STOP: self._first_stop,
            ParkingHardState.REVERSE_RIGHT: self._reverse_right,
            ParkingHardState.REVERSE_STRAIGHT: self._reverse_straight,
            ParkingHardState.REVERSE_PAUSE: self._reverse_pause,
            ParkingHardState.ADJUST_FORWARD: self._adjust_forward,
            ParkingHardState.GO_OUT_TURN: self._go_out_turn,
            ParkingHardState.GO_OUT_STRAIGHT: self._go_out_straight,
        }[self.state]
        return handler(scan, now)

    # --- sensing -----------------------------------------------------------
    def _sector_mean_mm(self, scan: np.ndarray | None, sector: tuple[float, float]) -> float:
        value = average_distance_mm_in_ros_sector(
            scan, sector[0], sector[1], self.config.raw_angle_for_ros_zero_deg
        )
        return float("inf") if value is None else value

    def _cmd(self, speed: int, steer: int, reason: str) -> DriveCommand:
        return DriveCommand(steer=int(steer), speed=int(speed), reason=reason)

    # --- states ------------------------------------------------------------
    def _search(self, scan: np.ndarray | None, now_s: float) -> DriveCommand:
        cfg = self.config
        distance = self._sector_mean_mm(scan, cfg.search_sector_ros_deg)
        if distance <= cfg.search_detect_mm:
            self.obs_detect += 1
            self.obs_clear = 0
        else:
            self.obs_clear += 1
            self.obs_detect = 0

        if self.obstacle_state == "no_obstacle" and self.obs_detect >= cfg.detect_frames:
            self.obstacle_state = "passed_first"
        elif self.obstacle_state == "passed_first" and self.obs_clear >= cfg.clear_frames:
            self.obstacle_state = "between"
        elif self.obstacle_state == "between" and self.obs_detect >= cfg.detect_frames:
            self.obstacle_state = "passed_second"
        elif self.obstacle_state == "passed_second" and self.obs_clear >= cfg.clear_frames:
            self._enter(ParkingHardState.FIRST_STOP, now_s)
            return self._cmd(0, cfg.init_steer, "parking_search_done_stop")

        # Otherwise keep driving forward past the parked cars.
        return self._cmd(cfg.search_speed, cfg.straight_steer, "parking_search")

    def _first_stop(self, scan: np.ndarray | None, now_s: float) -> DriveCommand:
        cfg = self.config
        if now_s - self.state_start_s >= cfg.first_stop_seconds:
            self._enter(ParkingHardState.REVERSE_RIGHT, now_s)
        return self._cmd(0, cfg.first_stop_steer, "parking_first_stop")

    def _reverse_right(self, scan: np.ndarray | None, now_s: float) -> DriveCommand:
        cfg = self.config
        self.left.update(self._sector_mean_mm(scan, cfg.left_sector_ros_deg), cfg.side_detect_mm)
        self.right.update(self._sector_mean_mm(scan, cfg.right_sector_ros_deg), cfg.side_detect_mm)

        if self.left.state == "no_obstacle" and self.left.detect >= cfg.detect_frames:
            self.left.state = "passing"
        if self.right.state == "no_obstacle" and self.right.detect >= cfg.detect_frames:
            self.right.state = "passing"

        if self.left.state == "passing" and self.right.state == "passing":
            self._enter(ParkingHardState.REVERSE_STRAIGHT, now_s)
        return self._cmd(cfg.reverse_right_speed, cfg.reverse_right_steer, "parking_reverse_right")

    def _reverse_straight(self, scan: np.ndarray | None, now_s: float) -> DriveCommand:
        cfg = self.config
        left_mm = self._sector_mean_mm(scan, cfg.left_sector_ros_deg)
        right_mm = self._sector_mean_mm(scan, cfg.right_sector_ros_deg)
        self.left.update(left_mm, cfg.side_detect_mm)
        self.right.update(right_mm, cfg.side_detect_mm)

        if self.left.state == "passing" and self.left.clear >= cfg.clear_frames:
            self.left.state = "out"
        if self.right.state == "passing" and self.right.clear >= cfg.clear_frames:
            self.right.state = "out"

        steering = self._centering_steer(left_mm, right_mm)

        if self.left.state == "out" and self.right.state == "out":
            self._enter(ParkingHardState.REVERSE_PAUSE, now_s)
        return self._cmd(cfg.reverse_straight_speed, steering, "parking_reverse_straight")

    def _centering_steer(self, left_mm: float, right_mm: float) -> int:
        cfg = self.config
        if not (np.isfinite(left_mm) and np.isfinite(right_mm)):
            return cfg.straight_steer
        diff = left_mm - right_mm
        if abs(diff) < cfg.centering_deadband_mm:
            return 0
        # Closer on the left -> steer right (positive); closer on the right -> left.
        return cfg.centering_steer if diff < 0 else -cfg.centering_steer

    def _reverse_pause(self, scan: np.ndarray | None, now_s: float) -> DriveCommand:
        cfg = self.config
        if now_s - self.state_start_s >= cfg.reverse_pause_seconds:
            self._enter(ParkingHardState.ADJUST_FORWARD, now_s)
        return self._cmd(0, 0, "parking_reverse_pause")

    def _adjust_forward(self, scan: np.ndarray | None, now_s: float) -> DriveCommand:
        cfg = self.config
        self.right.update(self._sector_mean_mm(scan, cfg.right_sector_ros_deg), cfg.side_detect_mm)

        if (
            now_s - self.state_start_s >= cfg.adjust_forward_min_seconds
            and self.right.state == "out"
            and self.right.clear >= cfg.clear_frames
        ):
            self.right.state = "passing"
            self._enter(ParkingHardState.GO_OUT_TURN, now_s)
        return self._cmd(cfg.adjust_forward_speed, cfg.init_steer, "parking_adjust_forward")

    def _go_out_turn(self, scan: np.ndarray | None, now_s: float) -> DriveCommand:
        cfg = self.config
        self.right.update(self._sector_mean_mm(scan, cfg.right_sector_ros_deg), cfg.go_out_detect_mm)

        if (
            now_s - self.state_start_s >= cfg.go_out_turn_min_seconds
            and self.right.state == "passing"
            and self.right.clear >= cfg.clear_frames
        ):
            self.right.state = "out"
            self._enter(ParkingHardState.GO_OUT_STRAIGHT, now_s)
        return self._cmd(cfg.go_out_turn_speed, cfg.go_out_turn_steer, "parking_go_out_turn")

    def _go_out_straight(self, scan: np.ndarray | None, now_s: float) -> DriveCommand:
        cfg = self.config
        return self._cmd(cfg.go_out_straight_speed, cfg.go_out_straight_steer, "parking_go_out_straight")

    def _enter(self, state: ParkingHardState, now_s: float) -> None:
        self.state = state
        self.state_start_s = now_s
