"""--mode parking_logic: rear-camera parking-line reverse-in mission.

Phases (all times/speeds/steers live in :class:`ParkingLogicConfig`):

1. ``SEARCH``       - drive forward; the right abeam LiDAR sector sees the FIRST
                      parked car (nearest return <= ``search_detect_mm``).
2. ``REVERSE_BACK`` - straight reverse for ``back_s`` seconds ("x1").
3. ``TURN_LEFT``    - full-left steer, drive forward for ``turn_s`` (6.7 s).
4. ``LINE_FOLLOW``  - reverse into the bay.  Steering does NOT come from this
                      FSM: the main loop crops the rear camera's bottom ROI,
                      extends the broken parking lines, runs the lane model and
                      injects the resulting steer each frame (the FSM only
                      supplies the phase + reverse speed).  Ends when BOTH side
                      sectors see the neighbour cars within ``side_detect_mm``
                      (nearest return), or after ``follow_max_s``.
5. ``DONE``         - hold still.

Same frame-driven contract as the other parking missions:
``step(scan, now_s)`` once per frame -> :class:`~ctrl_zero.control.DriveCommand`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ctrl_zero.control import DriveCommand
from ctrl_zero.lidar import min_distance_mm_in_ros_sector


class ParkingLogicState(Enum):
    SEARCH = 1
    REVERSE_BACK = 2
    TURN_LEFT = 3
    LINE_FOLLOW = 4
    FINAL_BACK = 5      # straight reverse a little further after the far_y stop
    PARKED_PAUSE = 6    # hold still, parked
    EXIT_APPROACH = 7   # pull straight forward out of the bay
    EXIT_ARC = 8        # full-lock forward turn back onto the aisle
    DONE = 9


@dataclass(frozen=True)
class ParkingLogicConfig:
    enabled: bool = True
    raw_angle_for_ros_zero_deg: float = 180.0

    # LiDAR sectors (ROS deg: forward=0, left=+90, right=-90/270).
    search_sector_ros_deg: tuple[float, float] = (270.0, 275.0)
    left_sector_ros_deg: tuple[float, float] = (65.0, 75.0)
    right_sector_ros_deg: tuple[float, float] = (285.0, 295.0)
    search_detect_mm: float = 2500.0   # first car: nearest return within this
    side_detect_mm: float = 1500.0     # line-follow stop: both sides within this
    detect_hold_s: float = 0.0         # first-car debounce (0 = immediate)

    # Steering (+right / -left).
    straight_steer: int = 0
    back_steer: int = 100              # full RIGHT while reversing in REVERSE_BACK
    turn_steer: int = -100             # full LEFT during TURN_LEFT

    # Speeds (+ forward / - reverse).
    search_speed: int = 150
    back_speed: int = -150
    turn_speed: int = 150
    follow_speed: int = -80            # reverse while following the parking line

    # Exit steering (same convention as parking_sim: +100 = full right).
    exit_arc_steer: int = 100

    # Durations (seconds).
    back_s: float = 1.0                # "x1": straight reverse after first car
    turn_s: float = 6.7                # full-left driving time
    follow_max_s: float = 12.0         # safety cap on the line-follow reverse
    # Opening stretch of LINE_FOLLOW: reverse dead straight and ignore every
    # sensor.  Right after the swing the bay is still entering the rear view, so
    # neither the camera nor the side LiDAR says anything trustworthy yet.
    follow_open_loop_s: float = 3.0
    final_back_s: float = 1.0          # extra straight reverse after the far_y stop
    parked_pause_s: float = 1.0        # hold still once parked
    exit_approach_s: float = 1.2       # straight forward out of the bay
    exit_arc_s: float = 7.5            # full-lock forward turn back onto the aisle


class ParkingLogicMission:
    """Frame-driven mission; call ``step(scan, now_s)`` once per frame."""

    def __init__(self, config: ParkingLogicConfig | None = None):
        self.config = config or ParkingLogicConfig()
        self.state = ParkingLogicState.SEARCH
        self.state_start_s: float | None = None
        self._detect_since: float | None = None
        self._elapsed = 0.0  # time in the current state, for the phase properties

    @property
    def done(self) -> bool:
        return self.state == ParkingLogicState.DONE

    @property
    def in_line_follow(self) -> bool:
        return self.state == ParkingLogicState.LINE_FOLLOW

    @property
    def line_follow_camera_active(self) -> bool:
        """True once LINE_FOLLOW is past its open-loop opening stretch.

        The caller uses this to skip the rear-camera pipeline entirely during the
        opening: the command there is plain straight reverse, so neither the slot
        model nor the lane model needs to run.
        """
        return (
            self.state == ParkingLogicState.LINE_FOLLOW
            and self._elapsed >= self.config.follow_open_loop_s
        )

    @property
    def in_initial_straight(self) -> bool:
        # SEARCH: driving straight looking for the first car — the phase where the
        # vanishing-point guide may hold heading instead of the open-loop steer.
        return self.state == ParkingLogicState.SEARCH

    def stop_line_follow(self, now_s: float | None = None) -> bool:
        """End LINE_FOLLOW from outside (rear-camera bay-end-line trigger).

        The LiDAR side check alone is weak: once the bay lines leave the rear
        view the model heading goes junk.  The main loop calls this when the
        white horizontal bay-end line rises past its far_y threshold.  Steering
        stops being camera-driven here, but the car still backs straight for
        ``final_back_s`` to seat itself fully in the bay.
        """
        if self.state != ParkingLogicState.LINE_FOLLOW:
            return False
        now = time.monotonic() if now_s is None else float(now_s)
        self._enter(ParkingLogicState.FINAL_BACK, now)
        return True

    def step(
        self, scan: np.ndarray | None, now_s: float | None = None
    ) -> DriveCommand | None:
        if not self.config.enabled:
            return None
        now = time.monotonic() if now_s is None else float(now_s)
        if self.state_start_s is None:
            self.state_start_s = now
        elapsed = now - self.state_start_s
        self._elapsed = elapsed
        handler = {
            ParkingLogicState.SEARCH: self._search,
            ParkingLogicState.REVERSE_BACK: self._reverse_back,
            ParkingLogicState.TURN_LEFT: self._turn_left,
            ParkingLogicState.LINE_FOLLOW: self._line_follow,
            ParkingLogicState.FINAL_BACK: self._final_back,
            ParkingLogicState.PARKED_PAUSE: self._parked_pause,
            ParkingLogicState.EXIT_APPROACH: self._exit_approach,
            ParkingLogicState.EXIT_ARC: self._exit_arc,
            ParkingLogicState.DONE: self._done,
        }[self.state]
        return handler(scan, now, elapsed)

    # --- sensing -----------------------------------------------------------
    def _sector_min_mm(self, scan, sector: tuple[float, float]) -> float:
        value = min_distance_mm_in_ros_sector(
            scan, sector[0], sector[1], self.config.raw_angle_for_ros_zero_deg
        )
        return float("inf") if value is None else value

    def _enter(self, state: ParkingLogicState, now_s: float) -> None:
        self.state = state
        self.state_start_s = now_s

    # --- states ------------------------------------------------------------
    def _search(self, scan, now_s: float, elapsed: float) -> DriveCommand:
        cfg = self.config
        detected = self._sector_min_mm(scan, cfg.search_sector_ros_deg) <= cfg.search_detect_mm
        if detected:
            if self._detect_since is None:
                self._detect_since = now_s
            if now_s - self._detect_since >= cfg.detect_hold_s:
                self._enter(ParkingLogicState.REVERSE_BACK, now_s)
                return DriveCommand(cfg.back_steer, cfg.back_speed, "parking_logic_back_start")
        else:
            self._detect_since = None
        return DriveCommand(cfg.straight_steer, cfg.search_speed, "parking_logic_search")

    def _reverse_back(self, scan, now_s: float, elapsed: float) -> DriveCommand:
        cfg = self.config
        if elapsed >= cfg.back_s:
            self._enter(ParkingLogicState.TURN_LEFT, now_s)
            return DriveCommand(cfg.turn_steer, cfg.turn_speed, "parking_logic_turn_left")
        return DriveCommand(cfg.back_steer, cfg.back_speed, "parking_logic_reverse_back")

    def _turn_left(self, scan, now_s: float, elapsed: float) -> DriveCommand:
        cfg = self.config
        if elapsed >= cfg.turn_s:
            self._enter(ParkingLogicState.LINE_FOLLOW, now_s)
            return DriveCommand(cfg.straight_steer, cfg.follow_speed, "parking_logic_line_follow")
        return DriveCommand(cfg.turn_steer, cfg.turn_speed, "parking_logic_turn_left")

    def _line_follow(self, scan, now_s: float, elapsed: float) -> DriveCommand:
        cfg = self.config
        if elapsed < cfg.follow_open_loop_s:
            # Opening stretch: straight reverse, every sensor ignored.
            return DriveCommand(cfg.straight_steer, cfg.follow_speed,
                                "parking_logic_line_follow_open")
        left = self._sector_min_mm(scan, cfg.left_sector_ros_deg)
        right = self._sector_min_mm(scan, cfg.right_sector_ros_deg)
        if left <= cfg.side_detect_mm and right <= cfg.side_detect_mm:
            self._enter(ParkingLogicState.FINAL_BACK, now_s)
            return DriveCommand(cfg.straight_steer, cfg.follow_speed, "parking_logic_final_back")
        if elapsed >= cfg.follow_max_s:
            self._enter(ParkingLogicState.FINAL_BACK, now_s)
            return DriveCommand(cfg.straight_steer, cfg.follow_speed, "parking_logic_final_back")
        # Steer is replaced by the rear-camera line pipeline in the main loop.
        return DriveCommand(cfg.straight_steer, cfg.follow_speed, "parking_logic_line_follow")

    def _final_back(self, scan, now_s: float, elapsed: float) -> DriveCommand:
        # Straight reverse a little further to seat the car in the bay.  Camera
        # steering is off here: the bay lines have already left the rear view.
        cfg = self.config
        if elapsed >= cfg.final_back_s:
            self._enter(ParkingLogicState.PARKED_PAUSE, now_s)
            return DriveCommand(cfg.straight_steer, 0, "parking_logic_parked_pause")
        return DriveCommand(cfg.straight_steer, cfg.follow_speed, "parking_logic_final_back")

    def _parked_pause(self, scan, now_s: float, elapsed: float) -> DriveCommand:
        cfg = self.config
        if elapsed >= cfg.parked_pause_s:
            self._enter(ParkingLogicState.EXIT_APPROACH, now_s)
            return DriveCommand(cfg.straight_steer, cfg.turn_speed, "parking_logic_exit_approach")
        return DriveCommand(cfg.straight_steer, 0, "parking_logic_parked_pause")

    def _exit_approach(self, scan, now_s: float, elapsed: float) -> DriveCommand:
        cfg = self.config
        if elapsed >= cfg.exit_approach_s:
            self._enter(ParkingLogicState.EXIT_ARC, now_s)
            return DriveCommand(cfg.exit_arc_steer, cfg.turn_speed, "parking_logic_exit_arc")
        return DriveCommand(cfg.straight_steer, cfg.turn_speed, "parking_logic_exit_approach")

    def _exit_arc(self, scan, now_s: float, elapsed: float) -> DriveCommand:
        cfg = self.config
        if elapsed >= cfg.exit_arc_s:
            self._enter(ParkingLogicState.DONE, now_s)
            return DriveCommand(cfg.straight_steer, cfg.turn_speed, "parking_logic_exit_straight")
        return DriveCommand(cfg.exit_arc_steer, cfg.turn_speed, "parking_logic_exit_arc")

    def _done(self, scan, now_s: float, elapsed: float) -> DriveCommand:
        # Exit complete: keep driving straight out of the parking area.
        cfg = self.config
        return DriveCommand(cfg.straight_steer, cfg.turn_speed, "parking_logic_exit_straight")
