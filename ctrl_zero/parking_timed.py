"""Timed two-arc perpendicular parking (simulation-derived).

Maneuver, driving west along the parking aisle with the bays on the right:

1. ``SEARCH``       – drive straight past the FIRST parked car; the instant its
                      trailing edge clears the right abeam LiDAR sector is the
                      anchor that starts the maneuver.
2. ``APPROACH``     – keep driving straight for a fixed time.
3. ``SWING``        – full-lock LEFT forward for a fixed time (rotate theta1
                      away from the bays).
4. ``GEAR_PAUSE``   – stop briefly; steering is pre-turned full RIGHT.
5. ``ARC_BACK``     – full-lock RIGHT reverse for a fixed time (rotate
                      theta2 = 90 deg - theta1, tail swings into the bay).
6. ``STRAIGHT_BACK``– reverse straight; stop when BOTH +/-90 deg sectors see
                      the neighbouring cars within ``side_detect_mm``.
7. ``DONE``         – hold still.

Segment durations come from the closed-form two-arc geometry validated in
the MuJoCo simulation (sim/parking_scenario.timed_plan); they depend only on
the start-marker column because the marker grid shifts the start laterally.
Sensor debounce is time-based, so the mission behaves the same whether it
is stepped at the camera frame rate (real car) or the physics rate (sim).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ctrl_zero.control import DriveCommand
from ctrl_zero.lidar import average_distance_mm_in_ros_sector


class TimedParkingState(Enum):
    SEARCH = 1
    APPROACH = 2
    SWING = 3
    GEAR_PAUSE = 4
    ARC_BACK = 5
    STRAIGHT_BACK = 6
    DONE = 7


@dataclass(frozen=True)
class TimedParkingConfig:
    enabled: bool = True
    raw_angle_for_ros_zero_deg: float = 180.0
    search_sector_ros_deg: tuple[float, float] = (-95.0, -85.0)
    left_sector_ros_deg: tuple[float, float] = (85.0, 95.0)
    right_sector_ros_deg: tuple[float, float] = (-95.0, -85.0)
    search_detect_mm: float = 2000.0
    side_detect_mm: float = 1000.0
    # Time-based debounce: a condition must hold this long before it counts.
    detect_hold_s: float = 0.2
    clear_hold_s: float = 0.2
    side_hold_s: float = 0.2

    # Steering commands (-100..100, positive = right; straight is the pot
    # calibration value, not necessarily 0).
    swing_steer: int = -100
    arc_steer: int = 100
    straight_steer: int = -1

    # Speeds (CTRL_ZERO units; + forward, - reverse).
    forward_speed: int = 100
    reverse_speed: int = -70

    # Timed segments (seconds); depend on the drawn start-marker column.
    approach_s: float = 3.65
    swing_s: float = 1.90
    gear_pause_s: float = 1.0
    arc_s: float = 12.04
    straight_back_max_s: float = 12.0


class _HoldTracker:
    """Tracks how long a boolean condition has held, in seconds."""

    def __init__(self) -> None:
        self.true_since: float | None = None
        self.false_since: float | None = None

    def update(self, condition: bool, now_s: float) -> None:
        if condition:
            if self.true_since is None:
                self.true_since = now_s
            self.false_since = None
        else:
            if self.false_since is None:
                self.false_since = now_s
            self.true_since = None

    def held_true(self, now_s: float, hold_s: float) -> bool:
        return self.true_since is not None and now_s - self.true_since >= hold_s

    def held_false(self, now_s: float, hold_s: float) -> bool:
        return (
            self.false_since is not None
            and now_s - self.false_since >= hold_s
        )


class TimedParkingMission:
    """Frame-driven timed parking; call ``step(scan, now_s)`` once per frame."""

    def __init__(self, config: TimedParkingConfig | None = None):
        self.config = config or TimedParkingConfig()
        self.state = TimedParkingState.SEARCH
        self.state_start_s: float | None = None
        self.search_stage = "no_car"  # -> first_car -> gap -> (trigger)
        self.search_tracker = _HoldTracker()
        self.side_tracker = _HoldTracker()

    @property
    def done(self) -> bool:
        return self.state == TimedParkingState.DONE

    def step(
        self, scan: np.ndarray | None, now_s: float | None = None
    ) -> DriveCommand | None:
        if not self.config.enabled:
            return None
        import time

        now = time.monotonic() if now_s is None else float(now_s)
        if self.state_start_s is None:
            self.state_start_s = now
        elapsed = now - self.state_start_s
        handler = {
            TimedParkingState.SEARCH: self._search,
            TimedParkingState.APPROACH: self._approach,
            TimedParkingState.SWING: self._swing,
            TimedParkingState.GEAR_PAUSE: self._gear_pause,
            TimedParkingState.ARC_BACK: self._arc_back,
            TimedParkingState.STRAIGHT_BACK: self._straight_back,
            TimedParkingState.DONE: self._done,
        }[self.state]
        return handler(scan, now, elapsed)

    # --- sensing -----------------------------------------------------------
    def _sector_mean_mm(
        self, scan: np.ndarray | None, sector: tuple[float, float]
    ) -> float:
        value = average_distance_mm_in_ros_sector(
            scan, sector[0], sector[1], self.config.raw_angle_for_ros_zero_deg
        )
        return float("inf") if value is None else value

    def _enter(self, state: TimedParkingState, now_s: float) -> None:
        self.state = state
        self.state_start_s = now_s

    # --- states ------------------------------------------------------------
    def _search(self, scan, now_s: float, elapsed: float) -> DriveCommand:
        cfg = self.config
        detected = (
            self._sector_mean_mm(scan, cfg.search_sector_ros_deg)
            <= cfg.search_detect_mm
        )
        self.search_tracker.update(detected, now_s)
        if self.search_stage == "no_car" and self.search_tracker.held_true(
            now_s, cfg.detect_hold_s
        ):
            self.search_stage = "first_car"
        elif self.search_stage == "first_car" and self.search_tracker.held_false(
            now_s, cfg.clear_hold_s
        ):
            # Anchor on the FIRST car: the instant we have driven past it (its
            # trailing edge clears the right abeam sector), start the maneuver.
            self._enter(TimedParkingState.APPROACH, now_s)
        return DriveCommand(cfg.straight_steer, cfg.forward_speed, "parking_sim_search")

    def _approach(self, scan, now_s: float, elapsed: float) -> DriveCommand:
        cfg = self.config
        if elapsed >= cfg.approach_s:
            self._enter(TimedParkingState.SWING, now_s)
        return DriveCommand(
            cfg.straight_steer, cfg.forward_speed, "parking_sim_approach"
        )

    def _swing(self, scan, now_s: float, elapsed: float) -> DriveCommand:
        cfg = self.config
        if elapsed >= cfg.swing_s:
            self._enter(TimedParkingState.GEAR_PAUSE, now_s)
        return DriveCommand(
            cfg.swing_steer, cfg.forward_speed, "parking_sim_swing_left"
        )

    def _gear_pause(self, scan, now_s: float, elapsed: float) -> DriveCommand:
        cfg = self.config
        if elapsed >= cfg.gear_pause_s:
            self._enter(TimedParkingState.ARC_BACK, now_s)
        # Pre-turn the wheels to full right while stopped.
        return DriveCommand(cfg.arc_steer, 0, "parking_sim_gear_pause")

    def _arc_back(self, scan, now_s: float, elapsed: float) -> DriveCommand:
        cfg = self.config
        if elapsed >= cfg.arc_s:
            self.side_tracker = _HoldTracker()
            self._enter(TimedParkingState.STRAIGHT_BACK, now_s)
        return DriveCommand(cfg.arc_steer, cfg.reverse_speed, "parking_sim_arc_back")

    def _straight_back(self, scan, now_s: float, elapsed: float) -> DriveCommand:
        cfg = self.config
        left = self._sector_mean_mm(scan, cfg.left_sector_ros_deg)
        right = self._sector_mean_mm(scan, cfg.right_sector_ros_deg)
        self.side_tracker.update(
            left <= cfg.side_detect_mm and right <= cfg.side_detect_mm, now_s
        )
        if self.side_tracker.held_true(now_s, cfg.side_hold_s):
            # Stop on THIS frame (speed 0), not one frame later, so the reverse
            # ends the instant both sides are seen.
            self._enter(TimedParkingState.DONE, now_s)
            return DriveCommand(cfg.straight_steer, 0, "parking_sim_straight_back_stop")
        if elapsed >= cfg.straight_back_max_s:
            # Safety stop: never reverse past the bay if the sensors miss.
            self._enter(TimedParkingState.DONE, now_s)
            return DriveCommand(cfg.straight_steer, 0, "parking_sim_straight_back_timeout")
        return DriveCommand(
            cfg.straight_steer, cfg.reverse_speed, "parking_sim_straight_back"
        )

    def _done(self, scan, now_s: float, elapsed: float) -> DriveCommand:
        return DriveCommand(0, 0, "parking_sim_done")
