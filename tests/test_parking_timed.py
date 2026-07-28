from __future__ import annotations

import numpy as np

from ctrl_zero.control import DriveCommand
from ctrl_zero.parking_timed import (
    TimedParkingConfig,
    TimedParkingMission,
    TimedParkingState,
)


def both_sides_scan(distance_mm: float = 800.0) -> np.ndarray:
    return np.array(
        [[90.0, distance_mm], [270.0, distance_mm]], dtype=np.float32
    )


def exit_config() -> TimedParkingConfig:
    return TimedParkingConfig(
        left_sector_ros_deg=(85.0, 95.0),
        right_sector_ros_deg=(265.0, 275.0),
        side_detect_mm=1000.0,
        side_hold_s=0.0,
        straight_steer=0,
        forward_speed=150,
        reverse_speed=-150,
        parked_pause_s=5.0,
        exit_approach_s=1.0,
        exit_arc_s=4.8,
        exit_arc_steer=100,
    )


def test_parking_completion_pauses_then_exits_on_arc_and_straight() -> None:
    cfg = exit_config()
    mission = TimedParkingMission(cfg)
    mission._enter(TimedParkingState.STRAIGHT_BACK, 10.0)

    stopped = mission.step(both_sides_scan(), now_s=10.0)
    assert stopped == DriveCommand(0, 0, "parking_sim_straight_back_stop")
    assert mission.state == TimedParkingState.PARKED_PAUSE

    paused = mission.step(both_sides_scan(), now_s=14.99)
    assert paused == DriveCommand(0, 0, "parking_sim_parked_pause")

    approaching = mission.step(both_sides_scan(), now_s=15.0)
    assert approaching == DriveCommand(0, 150, "parking_sim_exit_approach")
    assert mission.state == TimedParkingState.EXIT_APPROACH

    still_approaching = mission.step(both_sides_scan(), now_s=15.99)
    assert still_approaching == DriveCommand(
        0, 150, "parking_sim_exit_approach"
    )

    exiting = mission.step(both_sides_scan(), now_s=16.0)
    assert exiting == DriveCommand(100, 150, "parking_sim_exit_arc")
    assert mission.state == TimedParkingState.EXIT_ARC

    still_exiting = mission.step(both_sides_scan(), now_s=20.79)
    assert still_exiting == DriveCommand(100, 150, "parking_sim_exit_arc")

    straight = mission.step(both_sides_scan(), now_s=20.8)
    assert straight == DriveCommand(0, 150, "parking_sim_exit_straight")
    assert mission.state == TimedParkingState.DONE
    assert mission.done

    continuing = mission.step(both_sides_scan(), now_s=30.0)
    assert continuing == DriveCommand(0, 150, "parking_sim_exit_straight")


def test_straight_back_timeout_holds_stopped_without_auto_exit() -> None:
    cfg = exit_config()
    mission = TimedParkingMission(cfg)
    mission._enter(TimedParkingState.STRAIGHT_BACK, 0.0)

    no_side_returns = np.array([[180.0, 5000.0]], dtype=np.float32)
    stopped = mission.step(
        no_side_returns, now_s=cfg.straight_back_max_s
    )

    assert stopped == DriveCommand(0, 0, "parking_sim_straight_back_timeout")
    assert mission.state == TimedParkingState.ABORTED
    assert mission.done

    still_stopped = mission.step(no_side_returns, now_s=30.0)
    assert still_stopped == DriveCommand(0, 0, "parking_sim_aborted")
