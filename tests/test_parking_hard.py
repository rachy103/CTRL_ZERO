from __future__ import annotations

import numpy as np

from ctrl_zero.control import DriveCommand
from ctrl_zero.parking_hard import ParkingHardConfig, ParkingHardMission, ParkingHardState

INF = float("inf")


def scan(left_mm: float = INF, right_mm: float = INF) -> np.ndarray:
    """Build a scan with returns at raw 90 deg (left) and raw 270 deg (right)."""
    rows: list[list[float]] = []
    if np.isfinite(left_mm):
        rows += [[90.0, left_mm], [91.0, left_mm]]
    if np.isfinite(right_mm):
        rows += [[270.0, right_mm], [271.0, right_mm]]
    if not rows:
        rows = [[0.0, 5000.0]]  # only a far forward return
    return np.array(rows, dtype=np.float32)


def fast_config(**overrides) -> ParkingHardConfig:
    values = dict(
        detect_frames=3,
        clear_frames=3,
        first_stop_seconds=1.0,
        reverse_pause_seconds=1.0,
        adjust_forward_min_seconds=1.0,
        go_out_turn_min_seconds=1.0,
    )
    values.update(overrides)
    return ParkingHardConfig(**values)


def run(mission, count, *, now_s, scan_arg):
    cmd = None
    for _ in range(count):
        cmd = mission.step(scan_arg, now_s=now_s)
    return cmd


def test_search_counts_two_cars_then_stops():
    mission = ParkingHardMission(fast_config())
    near = scan(right_mm=1500.0)   # car on the right within search_detect (2000)
    far = scan(right_mm=INF)       # nothing on the right

    # First car: 3 near frames.
    run(mission, 3, now_s=0.0, scan_arg=near)
    assert mission.obstacle_state == "passed_first"
    # Gap: 3 clear frames.
    run(mission, 3, now_s=0.0, scan_arg=far)
    assert mission.obstacle_state == "between"
    # Second car: 3 near frames.
    run(mission, 3, now_s=0.0, scan_arg=near)
    assert mission.obstacle_state == "passed_second"
    # Past the second car: 3 clear frames -> stop and switch to FIRST_STOP.
    stop_cmd = run(mission, 3, now_s=0.0, scan_arg=far)
    assert mission.state == ParkingHardState.FIRST_STOP
    assert stop_cmd == DriveCommand(mission.config.init_steer, 0, "parking_search_done_stop")


def test_search_drives_forward_while_scanning():
    mission = ParkingHardMission(fast_config())
    cmd = mission.step(scan(right_mm=INF), now_s=0.0)
    assert cmd == DriveCommand(mission.config.straight_steer, mission.config.search_speed, "parking_search")


def test_first_stop_waits_then_reverses():
    mission = ParkingHardMission(fast_config(first_stop_seconds=1.0))
    mission.state = ParkingHardState.FIRST_STOP
    mission.state_start_s = 10.0

    holding = mission.step(scan(), now_s=10.5)
    assert holding == DriveCommand(mission.config.first_stop_steer, 0, "parking_first_stop")
    assert mission.state == ParkingHardState.FIRST_STOP

    mission.step(scan(), now_s=11.01)
    assert mission.state == ParkingHardState.REVERSE_RIGHT


def test_reverse_right_transitions_when_both_sides_seen():
    mission = ParkingHardMission(fast_config())
    mission._enter(ParkingHardState.REVERSE_RIGHT, 0.0)
    both = scan(left_mm=800.0, right_mm=800.0)  # within side_detect (1000)

    cmd = run(mission, 3, now_s=0.0, scan_arg=both)
    assert mission.left.state == "passing" and mission.right.state == "passing"
    assert mission.state == ParkingHardState.REVERSE_STRAIGHT
    assert cmd.reason in ("parking_reverse_right", "parking_reverse_straight")


def test_reverse_straight_centering_and_completion():
    mission = ParkingHardMission(fast_config())
    mission._enter(ParkingHardState.REVERSE_STRAIGHT, 0.0)
    mission.left.state = "passing"
    mission.right.state = "passing"

    # Closer on the left -> steer right (+centering_steer).
    biased = mission.step(scan(left_mm=500.0, right_mm=1000.0), now_s=0.0)
    assert biased.steer == mission.config.centering_steer
    assert biased.speed == mission.config.reverse_straight_speed

    # Both sides clear for clear_frames -> both "out" -> REVERSE_PAUSE.
    run(mission, 3, now_s=0.0, scan_arg=scan(left_mm=INF, right_mm=INF))
    assert mission.left.state == "out" and mission.right.state == "out"
    assert mission.state == ParkingHardState.REVERSE_PAUSE


def test_reverse_pause_waits_then_adjust_forward():
    mission = ParkingHardMission(fast_config(reverse_pause_seconds=1.0))
    mission._enter(ParkingHardState.REVERSE_PAUSE, 5.0)

    paused = mission.step(scan(), now_s=5.5)
    assert paused == DriveCommand(0, 0, "parking_reverse_pause")
    assert mission.state == ParkingHardState.REVERSE_PAUSE

    mission.step(scan(), now_s=6.01)
    assert mission.state == ParkingHardState.ADJUST_FORWARD


def test_full_flow_reaches_exit():
    cfg = fast_config()
    mission = ParkingHardMission(cfg)
    near = scan(right_mm=1500.0)
    far = scan()

    # SEARCH: two cars then clear.
    run(mission, 3, now_s=0.0, scan_arg=near)
    run(mission, 3, now_s=0.0, scan_arg=far)
    run(mission, 3, now_s=0.0, scan_arg=near)
    run(mission, 3, now_s=0.0, scan_arg=far)
    assert mission.state == ParkingHardState.FIRST_STOP

    # FIRST_STOP -> REVERSE_RIGHT.
    mission.step(far, now_s=2.01)
    assert mission.state == ParkingHardState.REVERSE_RIGHT

    # REVERSE_RIGHT: both sides seen.
    run(mission, 3, now_s=2.01, scan_arg=scan(left_mm=800.0, right_mm=800.0))
    assert mission.state == ParkingHardState.REVERSE_STRAIGHT

    # REVERSE_STRAIGHT: both sides clear.
    run(mission, 3, now_s=2.01, scan_arg=far)
    assert mission.state == ParkingHardState.REVERSE_PAUSE

    # REVERSE_PAUSE -> ADJUST_FORWARD.
    mission.step(far, now_s=4.02)
    assert mission.state == ParkingHardState.ADJUST_FORWARD

    # ADJUST_FORWARD: right already "out", min time elapsed, clear frames.
    run(mission, 3, now_s=6.0, scan_arg=far)
    assert mission.state == ParkingHardState.GO_OUT_TURN

    # GO_OUT_TURN: min time elapsed, right passing then clear.
    run(mission, 3, now_s=8.0, scan_arg=far)
    assert mission.state == ParkingHardState.GO_OUT_STRAIGHT
    assert mission.done

    out = mission.step(far, now_s=9.0)
    assert out == DriveCommand(cfg.go_out_straight_steer, cfg.go_out_straight_speed, "parking_go_out_straight")
