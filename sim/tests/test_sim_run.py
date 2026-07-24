from __future__ import annotations

import math
from argparse import Namespace
from copy import deepcopy
from typing import Any

import pytest

from sim import sim_run
from sim.config import ConfigurationError, set_value

import main as real_main


def _measured_config(*, zero_pwm_behavior: str = "brake") -> dict[str, Any]:
    """Return complete, test-only measurements for sim_run validation."""

    return {
        "schema_version": 1,
        "map": {
            "physical_width_m": 9.0,
            "physical_height_m": 13.0,
            "spawn": {
                "rear_axle_x_m": 1.25,
                "rear_axle_y_m": -2.5,
                "yaw_deg": 90.0,
            },
        },
        "vehicle": {
            "body_length_m": 0.99,
            "body_width_m": 0.55,
            "body_height_m": 0.41,
            "rear_axle_from_rear_m": 0.20,
            "front_axle_from_rear_m": 0.78,
            "wheelbase_m": 0.58,
            "front_track_m": 0.46,
            "rear_track_m": 0.45,
            "wheel_radius_m": 0.10,
            "wheel_width_m": 0.065,
        },
        "hardware": {
            "command": {
                "steer_min": -100,
                "steer_max": 100,
                "drive_pwm_min": -255,
                "drive_pwm_max": 255,
                "drive_pwm_deadband": 8,
            },
            "steering": {
                "straight_command": 0,
                "left_angle_deg_at_steer_min": 30.0,
                "right_angle_deg_at_steer_max": 24.0,
                "command_slew_per_second": 900.0,
            },
            "drive": {
                "driven_axle": "rear",
                "pwm_speed_table": [
                    [-255, -1.50],
                    [-8, 0.0],
                    [0, 0.0],
                    [8, 0.0],
                    [255, 2.00],
                ],
                "acceleration_limit_mps2": 2.0,
                "braking_deceleration_mps2": 4.0,
                "zero_pwm_behavior": zero_pwm_behavior,
                "coast_deceleration_mps2": (
                    0.5 if zero_pwm_behavior == "coast" else None
                ),
            },
            "camera": _valid_camera_config(),
        },
    }


def _valid_camera_config() -> dict[str, Any]:
    return {
        "width_px": 640,
        "height_px": 480,
        "fps": 30.0,
        "horizontal_fov_deg": 70.0,
        "position_from_rear_axle_m": [0.24, 0.0, 0.31],
        "pitch_deg": -12.0,
        "yaw_deg": 1.5,
    }


def _spawn_args(
    *,
    spawn_x: float | None = None,
    spawn_y: float | None = None,
    spawn_yaw_deg: float | None = None,
) -> Namespace:
    return Namespace(
        spawn_x=spawn_x,
        spawn_y=spawn_y,
        spawn_yaw_deg=spawn_yaw_deg,
    )


def _manual_control() -> sim_run.ManualControl:
    return sim_run.ManualControl(sim_run.build_real_drive_controller())


def test_validate_config_accepts_complete_measured_drive_schema() -> None:
    sim_run._validate_config(_measured_config(), auto=False)
    sim_run._validate_config(
        _measured_config(zero_pwm_behavior="coast"),
        auto=False,
    )


@pytest.mark.parametrize(
    "missing_key",
    [
        "hardware.drive.driven_axle",
        "hardware.drive.pwm_speed_table",
        "hardware.drive.acceleration_limit_mps2",
        "hardware.drive.braking_deceleration_mps2",
        "hardware.drive.zero_pwm_behavior",
    ],
)
def test_validate_config_blocks_missing_measured_drive_schema(
    missing_key: str,
) -> None:
    config = _measured_config()
    set_value(config, missing_key, None)

    with pytest.raises(ConfigurationError, match=missing_key):
        sim_run._validate_config(config, auto=False)


def test_validate_config_requires_coast_measurement_only_for_coast() -> None:
    braking_config = _measured_config(zero_pwm_behavior="brake")
    assert (
        braking_config["hardware"]["drive"]["coast_deceleration_mps2"] is None
    )
    sim_run._validate_config(braking_config, auto=False)

    coast_config = _measured_config(zero_pwm_behavior="coast")
    coast_config["hardware"]["drive"]["coast_deceleration_mps2"] = None
    with pytest.raises(
        ConfigurationError,
        match="hardware.drive.coast_deceleration_mps2",
    ):
        sim_run._validate_config(coast_config, auto=False)


def test_validate_build_config_does_not_require_steering_hardware() -> None:
    config = _measured_config()
    config.pop("hardware")
    config["map"].pop("spawn")

    sim_run._validate_build_config(config)


def test_apply_spawn_overrides_replaces_all_three_values() -> None:
    config = _measured_config()

    sim_run._apply_spawn_overrides(
        config,
        _spawn_args(spawn_x=-1.5, spawn_y=2.75, spawn_yaw_deg=-35.0),
    )

    assert config["map"]["spawn"] == {
        "rear_axle_x_m": -1.5,
        "rear_axle_y_m": 2.75,
        "yaw_deg": -35.0,
    }


@pytest.mark.parametrize(
    "args",
    [
        _spawn_args(spawn_x=1.0),
        _spawn_args(spawn_y=2.0),
        _spawn_args(spawn_yaw_deg=3.0),
        _spawn_args(spawn_x=1.0, spawn_y=2.0),
        _spawn_args(spawn_x=1.0, spawn_yaw_deg=3.0),
        _spawn_args(spawn_y=2.0, spawn_yaw_deg=3.0),
    ],
)
def test_apply_spawn_overrides_rejects_every_partial_combination(
    args: Namespace,
) -> None:
    config = _measured_config()
    original_spawn = deepcopy(config["map"]["spawn"])

    with pytest.raises(ConfigurationError, match="must be supplied together"):
        sim_run._apply_spawn_overrides(config, args)

    assert config["map"]["spawn"] == original_spawn


def test_apply_spawn_overrides_with_no_values_is_a_noop() -> None:
    config = _measured_config()
    original = deepcopy(config)

    sim_run._apply_spawn_overrides(config, _spawn_args())

    assert config == original


@pytest.mark.parametrize(
    ("dotted_key", "value"),
    [
        ("map.spawn.rear_axle_x_m", math.nan),
        ("map.spawn.rear_axle_y_m", math.inf),
        ("map.spawn.yaw_deg", "north"),
    ],
)
def test_validate_config_rejects_nonfinite_spawn_values(
    dotted_key: str,
    value: object,
) -> None:
    config = _measured_config()
    set_value(config, dotted_key, value)

    with pytest.raises(ConfigurationError, match=dotted_key):
        sim_run._validate_config(config, auto=False)


def test_validate_config_rejects_spawn_with_vehicle_off_floor() -> None:
    config = _measured_config()
    config["map"]["spawn"] = {
        "rear_axle_x_m": 4.45,
        "rear_axle_y_m": 0.0,
        "yaw_deg": 0.0,
    }

    with pytest.raises(ConfigurationError, match="footprint does not fit"):
        sim_run._validate_config(config, auto=False)


def test_validate_auto_camera_config_accepts_complete_measurements() -> None:
    config = {"hardware": {"camera": _valid_camera_config()}}

    sim_run._validate_auto_camera_config(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("width_px", None, "width_px requires a measured value"),
        ("width_px", 640.5, "whole pixels"),
        ("height_px", 0, "must be positive"),
        ("fps", 0, "must be positive"),
        ("horizontal_fov_deg", 180.0, "between 0 and 180"),
        ("position_from_rear_axle_m", [0.2, 0.0], "exactly three"),
        (
            "position_from_rear_axle_m",
            [0.2, math.nan, 0.3],
            r"position_from_rear_axle_m\[1\].*finite",
        ),
        ("pitch_deg", math.inf, "pitch_deg.*measured value"),
        ("yaw_deg", "right", "yaw_deg must be a finite"),
    ],
)
def test_validate_auto_camera_config_rejects_invalid_measurements(
    field: str,
    value: Any,
    message: str,
) -> None:
    camera = _valid_camera_config()
    camera[field] = value
    config = {"hardware": {"camera": camera}}

    with pytest.raises(ConfigurationError, match=message):
        sim_run._validate_auto_camera_config(config)


def test_manual_control_first_d_only_opens_main_start_gate() -> None:
    manual = _manual_control()

    assert manual.speed_step == real_main.MANUAL_SPEED_STEP
    assert manual.steer_power == real_main.MANUAL_STEER_POWER
    assert manual.steer_hold_s == pytest.approx(
        real_main.MANUAL_STEER_HOLD_MS / 1000.0
    )
    assert manual.min_speed == real_main.MIN_SPEED
    assert manual.max_speed == real_main.MAX_SPEED
    assert manual.max_steer == real_main.MAX_STEER
    assert ord("D") in real_main.START_KEYS

    manual.on_key(ord("W"))
    assert manual.manual_speed == 0
    assert not manual.drive_started

    manual.on_key(ord("D"))
    assert manual.drive_started
    assert manual.manual_speed == 0
    assert manual.manual_steer == 0
    command = manual.command()
    assert (command.steer, command.speed) == (0, 0)


def test_manual_control_w_and_s_follow_main_step_and_limits() -> None:
    manual = _manual_control()
    manual.on_key(ord("D"))

    manual.on_key(ord("W"))
    assert manual.manual_speed == real_main.MANUAL_SPEED_STEP
    assert manual.command().speed == real_main.MANUAL_SPEED_STEP

    for _ in range(100):
        manual.on_key(ord("W"))
    assert manual.manual_speed == real_main.MAX_SPEED
    assert manual.command().speed == real_main.MAX_SPEED

    manual.on_key(ord("S"))
    assert manual.manual_speed == (
        real_main.MAX_SPEED - real_main.MANUAL_SPEED_STEP
    )
    for _ in range(100):
        manual.on_key(ord("S"))
    assert manual.manual_speed == -real_main.MAX_SPEED
    assert manual.command().speed == -real_main.MAX_SPEED


def test_manual_control_a_d_and_c_follow_main_steering_pulse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 123.0
    monkeypatch.setattr(sim_run.time, "monotonic", lambda: now)
    manual = _manual_control()
    manual.on_key(ord("D"))
    expected_power = min(
        max(real_main.MANUAL_STEER_POWER, 0),
        real_main.MAX_STEER,
    )

    manual.on_key(ord("A"))
    assert manual.manual_steer == -expected_power
    assert manual.manual_steer_until == pytest.approx(
        now + real_main.MANUAL_STEER_HOLD_MS / 1000.0
    )
    assert manual.command().steer == -expected_power

    manual.on_key(ord("D"))
    assert manual.manual_steer == expected_power
    assert manual.command().steer == expected_power

    manual.on_key(ord("C"))
    assert manual.manual_steer == 0
    assert manual.manual_steer_until == 0.0
    assert manual.command().steer == 0


def test_manual_control_space_stops_and_arms_manual_takeover() -> None:
    manual = _manual_control()
    manual.on_key(ord("D"))
    manual.on_key(ord("W"))
    manual.on_key(ord("A"))
    assert manual.manual_speed != 0
    assert manual.manual_steer != 0

    manual.on_key(32)

    assert manual.drive_started
    assert manual.manual_takeover
    assert manual.manual_speed == 0
    assert manual.manual_steer == 0
    assert manual.manual_steer_until == 0.0
    command = manual.command()
    assert (command.steer, command.speed) == (0, 0)


def test_manual_control_can_be_explicitly_armed_for_headless_auto() -> None:
    manual = _manual_control()

    manual.arm_auto()

    assert manual.drive_started
    assert not manual.manual_takeover
