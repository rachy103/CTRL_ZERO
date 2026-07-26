from __future__ import annotations

import math
from dataclasses import replace

import pytest

from sim.config import ConfigurationError
from sim.vehicle_kinematics import (
    KinematicCalibration,
    KinematicVehicle,
    ackermann_front_angles,
    wrap_angle,
)


@pytest.fixture
def calibration() -> KinematicCalibration:
    return KinematicCalibration(
        wheelbase_m=0.60,
        front_track_m=0.50,
        rear_track_m=0.48,
        wheel_radius_m=0.10,
        steer_min=-100.0,
        steer_max=100.0,
        straight_command=0.0,
        left_angle_rad=math.radians(30.0),
        right_angle_rad=math.radians(24.0),
        command_slew_per_second=50.0,
        pwm_min=-255,
        pwm_max=255,
        pwm_deadband=8,
        driven_axle="rear",
        pwm_speed_table=(
            (-255, -1.50),
            (-128, -0.60),
            (-8, 0.0),
            (0, 0.0),
            (8, 0.0),
            (64, 0.25),
            (160, 1.10),
            (255, 2.00),
        ),
        speed_response_mode="rate_limited",
        acceleration_limit_mps2=2.0,
        braking_deceleration_mps2=4.0,
        zero_pwm_behavior="brake",
        coast_deceleration_mps2=None,
    )


@pytest.mark.parametrize(
    ("command", "expected_angle_deg"),
    [
        (-200.0, 30.0),
        (-100.0, 30.0),
        (-50.0, 15.0),
        (0.0, 0.0),
        (50.0, -12.0),
        (100.0, -24.0),
        (200.0, -24.0),
    ],
)
def test_steering_command_maps_piecewise_and_clamps(
    calibration: KinematicCalibration,
    command: float,
    expected_angle_deg: float,
) -> None:
    assert math.degrees(
        calibration.steering_angle_from_command(command)
    ) == pytest.approx(expected_angle_deg)


@pytest.mark.parametrize(
    ("pwm", "expected_speed_mps"),
    [
        (-500, -1.5),
        (-255, -1.5),
        (-128, -0.6),
        (-68, -0.3),
        (-8, 0.0),
        (-4, 0.0),
        (0, 0.0),
        (4, 0.0),
        (8, 0.0),
        (36, 0.125),
        (112, 0.675),
        (160, 1.1),
        (208, 1.1 + 0.9 * 48.0 / 95.0),
        (255, 2.0),
        (500, 2.0),
    ],
)
def test_pwm_speed_table_clamps_and_interpolates_between_measured_rows(
    calibration: KinematicCalibration,
    pwm: int,
    expected_speed_mps: float,
) -> None:
    assert calibration.speed_from_pwm(pwm) == pytest.approx(expected_speed_mps)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (
            {
                "pwm_speed_table": (
                    (-255, -1.5),
                    (-8, 0.0),
                    (0, 0.0),
                    (8, 0.0),
                    (255, 2.0),
                    (128, 1.0),
                )
            },
            "strictly increasing",
        ),
        (
            {
                "pwm_speed_table": (
                    (-255, -1.5),
                    (-8, 0.0),
                    (0, 0.0),
                    (9, 0.0),
                    (255, 2.0),
                )
            },
            r"\+deadband",
        ),
        (
            {
                "pwm_speed_table": (
                    (-255, -1.5),
                    (-8, 0.1),
                    (0, 0.0),
                    (8, 0.0),
                    (255, 2.0),
                )
            },
            "Measured speed must be 0",
        ),
        (
            {
                "pwm_speed_table": (
                    (-255, -1.5),
                    (-8, 0.0),
                    (0, 0.0),
                    (8, 0.0),
                    (128, 1.2),
                    (255, 1.0),
                )
            },
            "nondecreasing",
        ),
        (
            {
                "pwm_speed_table": (
                    (-254, -1.5),
                    (-8, 0.0),
                    (0, 0.0),
                    (8, 0.0),
                    (255, 2.0),
                )
            },
            "min and max PWM endpoints",
        ),
    ],
)
def test_invalid_pwm_speed_tables_are_rejected(
    calibration: KinematicCalibration,
    replacement: dict[str, object],
    message: str,
) -> None:
    invalid = replace(calibration, **replacement)

    with pytest.raises(ConfigurationError, match=message):
        invalid.validate()


def test_steering_command_slew_limits_each_step(
    calibration: KinematicCalibration,
) -> None:
    vehicle = KinematicVehicle(
        calibration,
        rear_axle_x_m=0.0,
        rear_axle_y_m=0.0,
        yaw_rad=0.0,
    )
    vehicle.set_command(steer=-100, pwm=0)

    first = vehicle.step(0.20)
    assert first.steer_command == pytest.approx(-10.0)
    assert math.degrees(first.center_steering_angle_rad) == pytest.approx(3.0)

    second = vehicle.step(0.20)
    assert second.steer_command == pytest.approx(-20.0)
    assert math.degrees(second.center_steering_angle_rad) == pytest.approx(6.0)


def test_ackermann_uses_inner_and_outer_wheel_angles() -> None:
    center = math.radians(20.0)
    positive_left, positive_right = ackermann_front_angles(center, 0.60, 0.50)
    negative_left, negative_right = ackermann_front_angles(-center, 0.60, 0.50)

    assert positive_left > positive_right > 0.0
    assert negative_right < negative_left < 0.0
    assert negative_left == pytest.approx(-positive_right)
    assert negative_right == pytest.approx(-positive_left)
    assert ackermann_front_angles(0.0, 0.60, 0.50) == (0.0, 0.0)


def test_straight_integration_advances_along_current_yaw(
    calibration: KinematicCalibration,
) -> None:
    vehicle = KinematicVehicle(
        calibration,
        rear_axle_x_m=1.0,
        rear_axle_y_m=-2.0,
        yaw_rad=math.pi / 2.0,
    )
    vehicle.set_command(steer=0, pwm=255)

    state = vehicle.step(0.25)

    assert state.rear_axle_x_m == pytest.approx(1.0, abs=1e-12)
    assert state.rear_axle_y_m == pytest.approx(-1.9375)
    assert state.yaw_rad == pytest.approx(math.pi / 2.0)
    assert state.speed_mps == pytest.approx(0.5)
    assert state.center_steering_angle_rad == pytest.approx(0.0)
    assert state.wheel_spin_rad == pytest.approx(wrap_angle(0.625))


@pytest.mark.parametrize(
    ("steer", "expected_yaw_sign"),
    [
        (-100, 1.0),
        (100, -1.0),
    ],
)
def test_turning_integration_updates_yaw_with_steering_sign(
    calibration: KinematicCalibration,
    steer: int,
    expected_yaw_sign: float,
) -> None:
    fast_slew = KinematicCalibration(
        **{
            **calibration.__dict__,
            "command_slew_per_second": 10_000.0,
            "acceleration_limit_mps2": 1_000.0,
        }
    )
    vehicle = KinematicVehicle(
        fast_slew,
        rear_axle_x_m=0.0,
        rear_axle_y_m=0.0,
        yaw_rad=0.0,
    )
    vehicle.set_command(steer=steer, pwm=255)

    state = vehicle.step(0.10)
    expected_angle = fast_slew.steering_angle_from_command(steer)
    expected_yaw = (
        1.0
        * math.tan(expected_angle)
        / fast_slew.wheelbase_m
        * 0.10
    )

    assert state.rear_axle_x_m == pytest.approx(0.10)
    assert state.rear_axle_y_m == pytest.approx(0.0)
    assert state.yaw_rad == pytest.approx(expected_yaw)
    assert math.copysign(1.0, state.yaw_rad) == expected_yaw_sign


def test_nonpositive_step_is_rejected(
    calibration: KinematicCalibration,
) -> None:
    vehicle = KinematicVehicle(
        calibration,
        rear_axle_x_m=0.0,
        rear_axle_y_m=0.0,
        yaw_rad=0.0,
    )

    with pytest.raises(ValueError, match="dt must be positive"):
        vehicle.step(0.0)


def test_measured_acceleration_limit_controls_speed_and_travel(
    calibration: KinematicCalibration,
) -> None:
    vehicle = KinematicVehicle(
        calibration,
        rear_axle_x_m=0.0,
        rear_axle_y_m=0.0,
        yaw_rad=0.0,
    )
    vehicle.set_command(steer=0, pwm=255)

    first = vehicle.step(0.25)
    assert first.speed_mps == pytest.approx(0.5)
    assert first.rear_axle_x_m == pytest.approx(0.0625)

    second = vehicle.step(0.25)
    assert second.speed_mps == pytest.approx(1.0)
    assert second.rear_axle_x_m == pytest.approx(0.25)


def test_instantaneous_response_reaches_target_without_rate_values(
    calibration: KinematicCalibration,
) -> None:
    instantaneous = replace(
        calibration,
        speed_response_mode="instantaneous",
        acceleration_limit_mps2=None,
        braking_deceleration_mps2=None,
    )
    instantaneous.validate()
    vehicle = KinematicVehicle(
        instantaneous,
        rear_axle_x_m=0.0,
        rear_axle_y_m=0.0,
        yaw_rad=0.0,
    )

    vehicle.set_command(steer=0, pwm=255)
    forward = vehicle.step(0.25)
    assert forward.speed_mps == pytest.approx(2.0)
    assert forward.rear_axle_x_m == pytest.approx(0.5)

    vehicle.set_command(steer=0, pwm=-255)
    reverse = vehicle.step(0.25)
    assert reverse.speed_mps == pytest.approx(-1.5)
    assert reverse.rear_axle_x_m == pytest.approx(0.125)


def test_brake_behavior_uses_measured_braking_deceleration(
    calibration: KinematicCalibration,
) -> None:
    vehicle = KinematicVehicle(
        calibration,
        rear_axle_x_m=0.0,
        rear_axle_y_m=0.0,
        yaw_rad=0.0,
    )
    vehicle.set_command(steer=0, pwm=255)
    vehicle.step(1.0)
    assert vehicle.state.speed_mps == pytest.approx(2.0)

    vehicle.set_command(steer=0, pwm=0)
    state = vehicle.step(0.25)
    assert state.speed_mps == pytest.approx(1.0)


def test_coast_behavior_uses_separate_measured_deceleration(
    calibration: KinematicCalibration,
) -> None:
    coast = replace(
        calibration,
        zero_pwm_behavior="coast",
        coast_deceleration_mps2=0.5,
    )
    coast.validate()
    vehicle = KinematicVehicle(
        coast,
        rear_axle_x_m=0.0,
        rear_axle_y_m=0.0,
        yaw_rad=0.0,
    )
    vehicle.set_command(steer=0, pwm=255)
    vehicle.step(1.0)
    vehicle.set_command(steer=0, pwm=0)

    state = vehicle.step(0.5)

    assert state.speed_mps == pytest.approx(1.75)


def test_coast_behavior_requires_measured_coast_deceleration(
    calibration: KinematicCalibration,
) -> None:
    coast_without_measurement = replace(
        calibration,
        zero_pwm_behavior="coast",
        coast_deceleration_mps2=None,
    )

    with pytest.raises(ConfigurationError, match="coast_deceleration_mps2"):
        coast_without_measurement.validate()


def test_from_config_blocks_null_measured_hardware() -> None:
    config = {
        "vehicle": {
            "wheelbase_m": 0.60,
            "front_track_m": 0.50,
            "rear_track_m": 0.48,
            "wheel_radius_m": 0.10,
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
                "left_angle_deg_at_steer_min": 30,
                "right_angle_deg_at_steer_max": 24,
                "command_slew_per_second": 900,
            },
            "drive": {
                "driven_axle": "rear",
                "pwm_speed_table": None,
                "speed_response_mode": "rate_limited",
                "acceleration_limit_mps2": None,
                "braking_deceleration_mps2": None,
                "zero_pwm_behavior": None,
            },
        },
    }

    with pytest.raises(ConfigurationError) as captured:
        KinematicCalibration.from_config(config)

    message = str(captured.value)
    assert "hardware.drive.pwm_speed_table" in message
    assert "hardware.drive.acceleration_limit_mps2" in message
    assert "hardware.drive.braking_deceleration_mps2" in message
    assert "hardware.drive.zero_pwm_behavior" in message
    assert "No fallback or estimated value was substituted." in message
