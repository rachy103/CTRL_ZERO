from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from .config import (
        KINEMATIC_HARDWARE_REQUIRED,
        ConfigurationError,
        get_value,
        is_known,
        require_parameters,
    )
except ImportError:  # Direct execution through sim_run.py
    from config import (
        KINEMATIC_HARDWARE_REQUIRED,
        ConfigurationError,
        get_value,
        is_known,
        require_parameters,
    )


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def wrap_angle(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def axis_angle_quaternion(axis: tuple[float, float, float], angle_rad: float) -> np.ndarray:
    axis_vector = np.asarray(axis, dtype=float)
    axis_vector /= np.linalg.norm(axis_vector)
    half = angle_rad / 2.0
    return np.asarray(
        [math.cos(half), *(axis_vector * math.sin(half))],
        dtype=float,
    )


@dataclass(frozen=True)
class KinematicCalibration:
    wheelbase_m: float
    front_track_m: float
    rear_track_m: float
    wheel_radius_m: float
    steer_min: float
    steer_max: float
    straight_command: float
    left_angle_rad: float
    right_angle_rad: float
    command_slew_per_second: float
    pwm_min: int
    pwm_max: int
    pwm_deadband: int
    driven_axle: str
    pwm_speed_table: tuple[tuple[int, float], ...]
    speed_response_mode: str
    acceleration_limit_mps2: float | None
    braking_deceleration_mps2: float | None
    zero_pwm_behavior: str
    coast_deceleration_mps2: float | None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "KinematicCalibration":
        response_mode = str(
            get_value(config, "hardware.drive.speed_response_mode")
        ).strip().lower()
        required_hardware = list(KINEMATIC_HARDWARE_REQUIRED)
        if response_mode == "rate_limited":
            required_hardware.extend(
                (
                    "hardware.drive.acceleration_limit_mps2",
                    "hardware.drive.braking_deceleration_mps2",
                )
            )
        require_parameters(
            config,
            (
                "vehicle.wheelbase_m",
                "vehicle.front_track_m",
                "vehicle.rear_track_m",
                "vehicle.wheel_radius_m",
                *required_hardware,
            ),
            context="Kinematic vehicle calibration",
        )
        raw_table = get_value(config, "hardware.drive.pwm_speed_table")
        if not isinstance(raw_table, (list, tuple)):
            raise ConfigurationError(
                "hardware.drive.pwm_speed_table must be a measured list of [PWM, m/s] rows."
            )
        parsed_table: list[tuple[int, float]] = []
        for index, row in enumerate(raw_table):
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                raise ConfigurationError(
                    f"pwm_speed_table row {index} must contain exactly [PWM, speed_mps]."
                )
            try:
                pwm_float = float(row[0])
                speed = float(row[1])
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(
                    f"pwm_speed_table row {index} must contain finite numbers."
                ) from exc
            if not math.isfinite(pwm_float) or not math.isfinite(speed):
                raise ConfigurationError(
                    f"pwm_speed_table row {index} must contain finite numbers."
                )
            pwm = int(pwm_float)
            if pwm_float != pwm:
                raise ConfigurationError(
                    f"pwm_speed_table PWM at row {index} must be an integer."
                )
            parsed_table.append((pwm, speed))
        coast_value = get_value(config, "hardware.drive.coast_deceleration_mps2")
        acceleration_value = get_value(
            config, "hardware.drive.acceleration_limit_mps2"
        )
        braking_value = get_value(
            config, "hardware.drive.braking_deceleration_mps2"
        )
        calibration = cls(
            wheelbase_m=float(get_value(config, "vehicle.wheelbase_m")),
            front_track_m=float(get_value(config, "vehicle.front_track_m")),
            rear_track_m=float(get_value(config, "vehicle.rear_track_m")),
            wheel_radius_m=float(get_value(config, "vehicle.wheel_radius_m")),
            steer_min=float(get_value(config, "hardware.command.steer_min")),
            steer_max=float(get_value(config, "hardware.command.steer_max")),
            straight_command=float(get_value(config, "hardware.steering.straight_command")),
            left_angle_rad=math.radians(
                float(get_value(config, "hardware.steering.left_angle_deg_at_steer_min"))
            ),
            right_angle_rad=math.radians(
                float(get_value(config, "hardware.steering.right_angle_deg_at_steer_max"))
            ),
            command_slew_per_second=float(
                get_value(config, "hardware.steering.command_slew_per_second")
            ),
            pwm_min=int(get_value(config, "hardware.command.drive_pwm_min")),
            pwm_max=int(get_value(config, "hardware.command.drive_pwm_max")),
            pwm_deadband=int(get_value(config, "hardware.command.drive_pwm_deadband")),
            driven_axle=str(
                get_value(config, "hardware.drive.driven_axle")
            ).strip().lower(),
            pwm_speed_table=tuple(parsed_table),
            speed_response_mode=response_mode,
            acceleration_limit_mps2=(
                float(acceleration_value) if is_known(acceleration_value) else None
            ),
            braking_deceleration_mps2=(
                float(braking_value) if is_known(braking_value) else None
            ),
            zero_pwm_behavior=str(
                get_value(config, "hardware.drive.zero_pwm_behavior")
            ).strip().lower(),
            coast_deceleration_mps2=(
                float(coast_value) if is_known(coast_value) else None
            ),
        )
        calibration.validate()
        return calibration

    def validate(self) -> None:
        positive = {
            "wheelbase_m": self.wheelbase_m,
            "front_track_m": self.front_track_m,
            "rear_track_m": self.rear_track_m,
            "wheel_radius_m": self.wheel_radius_m,
            "left_angle_deg_at_steer_min": self.left_angle_rad,
            "right_angle_deg_at_steer_max": self.right_angle_rad,
            "command_slew_per_second": self.command_slew_per_second,
        }
        if self.speed_response_mode == "rate_limited":
            positive.update(
                {
                    "acceleration_limit_mps2": self.acceleration_limit_mps2,
                    "braking_deceleration_mps2": self.braking_deceleration_mps2,
                }
            )
        invalid = [
            name
            for name, value in positive.items()
            if not math.isfinite(value) or value <= 0
        ]
        if invalid:
            raise ConfigurationError(
                "Measured parameters must be positive: " + ", ".join(invalid)
            )
        finite_values = {
            "steer_min": self.steer_min,
            "steer_max": self.steer_max,
            "straight_command": self.straight_command,
        }
        non_finite = [
            name
            for name, value in finite_values.items()
            if not math.isfinite(value)
        ]
        if non_finite:
            raise ConfigurationError(
                "Measured parameters must be finite: " + ", ".join(non_finite)
            )
        if not self.steer_min < self.straight_command < self.steer_max:
            raise ConfigurationError(
                "straight_command must lie strictly inside the steering command range."
            )
        if self.left_angle_rad >= math.pi / 2 or self.right_angle_rad >= math.pi / 2:
            raise ConfigurationError(
                "Measured road-wheel steering angles must be less than 90 degrees."
            )
        if not self.pwm_min < 0 < self.pwm_max:
            raise ConfigurationError("Drive PWM range must cross zero.")
        if not 0 <= self.pwm_deadband < min(abs(self.pwm_min), self.pwm_max):
            raise ConfigurationError(
                "Drive PWM deadband must be non-negative and lie inside the PWM range."
            )
        if self.driven_axle not in {"front", "rear", "all"}:
            raise ConfigurationError(
                "hardware.drive.driven_axle must be front, rear, or all."
            )
        if self.speed_response_mode not in {"instantaneous", "rate_limited"}:
            raise ConfigurationError(
                "hardware.drive.speed_response_mode must be instantaneous or rate_limited."
            )
        if self.zero_pwm_behavior not in {"coast", "brake"}:
            raise ConfigurationError(
                "hardware.drive.zero_pwm_behavior must be coast or brake."
            )
        if self.zero_pwm_behavior == "coast":
            if (
                self.coast_deceleration_mps2 is None
                or not math.isfinite(self.coast_deceleration_mps2)
                or self.coast_deceleration_mps2 <= 0
            ):
                raise ConfigurationError(
                    "A measured positive coast_deceleration_mps2 is required for coast behavior."
                )
        self._validate_pwm_speed_table()

    def _validate_pwm_speed_table(self) -> None:
        if len(self.pwm_speed_table) < 5:
            raise ConfigurationError(
                "pwm_speed_table needs measured endpoints and deadband anchors."
            )
        pwms = [row[0] for row in self.pwm_speed_table]
        speeds = [row[1] for row in self.pwm_speed_table]
        if pwms != sorted(pwms) or len(set(pwms)) != len(pwms):
            raise ConfigurationError(
                "pwm_speed_table PWM values must be strictly increasing."
            )
        if pwms[0] != self.pwm_min or pwms[-1] != self.pwm_max:
            raise ConfigurationError(
                "pwm_speed_table must include measured min and max PWM endpoints."
            )
        required_zero_anchors = {-self.pwm_deadband, 0, self.pwm_deadband}
        table = dict(self.pwm_speed_table)
        if not required_zero_anchors.issubset(table):
            raise ConfigurationError(
                "pwm_speed_table must include -deadband, 0, and +deadband rows."
            )
        if any(abs(table[pwm]) > 1e-12 for pwm in required_zero_anchors):
            raise ConfigurationError(
                "Measured speed must be 0 at -deadband, 0, and +deadband."
            )
        if not speeds[0] < 0 < speeds[-1]:
            raise ConfigurationError(
                "Measured endpoint speeds must be negative at min PWM and positive at max PWM."
            )
        if any(second < first for first, second in zip(speeds, speeds[1:])):
            raise ConfigurationError(
                "Measured speeds must be nondecreasing with increasing PWM."
            )

    def steering_angle_from_command(self, command: float) -> float:
        command = clamp(command, self.steer_min, self.steer_max)
        if command <= self.straight_command:
            fraction = (self.straight_command - command) / (
                self.straight_command - self.steer_min
            )
            return fraction * self.left_angle_rad
        fraction = (command - self.straight_command) / (
            self.steer_max - self.straight_command
        )
        return -fraction * self.right_angle_rad

    def speed_from_pwm(self, pwm: int) -> float:
        pwm = int(clamp(pwm, self.pwm_min, self.pwm_max))
        pwms = [row[0] for row in self.pwm_speed_table]
        index = bisect_left(pwms, pwm)
        if index < len(pwms) and pwms[index] == pwm:
            return self.pwm_speed_table[index][1]
        lower_pwm, lower_speed = self.pwm_speed_table[index - 1]
        upper_pwm, upper_speed = self.pwm_speed_table[index]
        fraction = (pwm - lower_pwm) / (upper_pwm - lower_pwm)
        return lower_speed + fraction * (upper_speed - lower_speed)


def ackermann_front_angles(
    center_angle_rad: float,
    wheelbase_m: float,
    track_m: float,
) -> tuple[float, float]:
    """Return left/right road-wheel angles; left turns are positive."""

    if abs(center_angle_rad) < 1e-12:
        return 0.0, 0.0
    direction = 1.0 if center_angle_rad > 0 else -1.0
    radius = wheelbase_m / math.tan(abs(center_angle_rad))
    half_track = track_m / 2.0
    inner_denominator = max(radius - half_track, 1e-9)
    outer_denominator = radius + half_track
    inner = math.atan(wheelbase_m / inner_denominator)
    outer = math.atan(wheelbase_m / outer_denominator)
    if direction > 0:
        return inner, outer
    return -outer, -inner


@dataclass
class VehicleState:
    rear_axle_x_m: float
    rear_axle_y_m: float
    yaw_rad: float
    speed_mps: float = 0.0
    steer_command: float = 0.0
    center_steering_angle_rad: float = 0.0
    wheel_spin_rad: float = 0.0


class KinematicVehicle:
    """Measured-command bicycle model driving MuJoCo mocap visuals."""

    def __init__(
        self,
        calibration: KinematicCalibration,
        *,
        rear_axle_x_m: float,
        rear_axle_y_m: float,
        yaw_rad: float,
    ):
        self.calibration = calibration
        self.state = VehicleState(
            rear_axle_x_m=rear_axle_x_m,
            rear_axle_y_m=rear_axle_y_m,
            yaw_rad=yaw_rad,
            steer_command=calibration.straight_command,
        )
        self.target_steer_command = calibration.straight_command
        self.target_pwm = 0

    def set_command(self, steer: int, pwm: int) -> None:
        self.target_steer_command = clamp(
            float(steer),
            self.calibration.steer_min,
            self.calibration.steer_max,
        )
        self.target_pwm = int(
            clamp(pwm, self.calibration.pwm_min, self.calibration.pwm_max)
        )

    def step(self, dt: float) -> VehicleState:
        if dt <= 0:
            raise ValueError("dt must be positive")
        maximum_step = self.calibration.command_slew_per_second * dt
        command_error = self.target_steer_command - self.state.steer_command
        self.state.steer_command += clamp(
            command_error,
            -maximum_step,
            maximum_step,
        )
        steering_angle = self.calibration.steering_angle_from_command(
            self.state.steer_command
        )
        target_speed = self.calibration.speed_from_pwm(self.target_pwm)
        previous_speed = self.state.speed_mps
        if self.calibration.speed_response_mode == "instantaneous":
            speed = target_speed
        elif math.isclose(previous_speed, target_speed, abs_tol=1e-12):
            speed = target_speed
        else:
            reversing = previous_speed * target_speed < 0
            increasing_magnitude = (
                not reversing and abs(target_speed) > abs(previous_speed)
            )
            if increasing_magnitude:
                rate_limit = self.calibration.acceleration_limit_mps2
            elif (
                target_speed == 0.0
                and self.calibration.zero_pwm_behavior == "coast"
            ):
                rate_limit = float(self.calibration.coast_deceleration_mps2)
            else:
                rate_limit = self.calibration.braking_deceleration_mps2
            speed = previous_speed + clamp(
                target_speed - previous_speed,
                -rate_limit * dt,
                rate_limit * dt,
            )
            if reversing and previous_speed * speed < 0:
                speed = 0.0

        travel_speed = (
            speed
            if self.calibration.speed_response_mode == "instantaneous"
            else (previous_speed + speed) / 2.0
        )
        travel_m = travel_speed * dt
        self.state.rear_axle_x_m += travel_m * math.cos(self.state.yaw_rad)
        self.state.rear_axle_y_m += travel_m * math.sin(self.state.yaw_rad)
        yaw_rate = (
            travel_speed
            * math.tan(steering_angle)
            / self.calibration.wheelbase_m
        )
        self.state.yaw_rad = wrap_angle(self.state.yaw_rad + yaw_rate * dt)
        self.state.speed_mps = speed
        self.state.center_steering_angle_rad = steering_angle
        self.state.wheel_spin_rad = wrap_angle(
            self.state.wheel_spin_rad + travel_m / self.calibration.wheel_radius_m
        )
        return self.state

    @staticmethod
    def _mocap_id(model: Any, body_name: str) -> int:
        mocap_values = np.asarray(model.body(body_name).mocapid).reshape(-1)
        if mocap_values.size != 1:
            raise RuntimeError(f"Unexpected mocapid shape for body {body_name!r}.")
        mocap_id = int(mocap_values[0])
        if mocap_id < 0:
            raise RuntimeError(f"Body {body_name!r} is not a mocap body.")
        return mocap_id

    def _world_point(self, forward_m: float, left_m: float) -> tuple[float, float]:
        cosine = math.cos(self.state.yaw_rad)
        sine = math.sin(self.state.yaw_rad)
        return (
            self.state.rear_axle_x_m + cosine * forward_m - sine * left_m,
            self.state.rear_axle_y_m + sine * forward_m + cosine * left_m,
        )

    def sync_mocap(self, model: Any, data: Any) -> None:
        yaw_quaternion = axis_angle_quaternion((0.0, 0.0, 1.0), self.state.yaw_rad)
        chassis_id = self._mocap_id(model, "vehicle_body")
        data.mocap_pos[chassis_id] = (
            self.state.rear_axle_x_m,
            self.state.rear_axle_y_m,
            0.0,
        )
        data.mocap_quat[chassis_id] = yaw_quaternion

        front_left_angle, front_right_angle = ackermann_front_angles(
            self.state.center_steering_angle_rad,
            self.calibration.wheelbase_m,
            self.calibration.front_track_m,
        )
        wheel_specs = (
            (
                "front_left_wheel",
                self.calibration.wheelbase_m,
                self.calibration.front_track_m / 2.0,
                front_left_angle,
            ),
            (
                "front_right_wheel",
                self.calibration.wheelbase_m,
                -self.calibration.front_track_m / 2.0,
                front_right_angle,
            ),
            (
                "rear_left_wheel",
                0.0,
                self.calibration.rear_track_m / 2.0,
                0.0,
            ),
            (
                "rear_right_wheel",
                0.0,
                -self.calibration.rear_track_m / 2.0,
                0.0,
            ),
        )
        spin_quaternion = axis_angle_quaternion(
            (0.0, 1.0, 0.0),
            self.state.wheel_spin_rad,
        )
        for name, forward, left, steering in wheel_specs:
            x_m, y_m = self._world_point(forward, left)
            wheel_id = self._mocap_id(model, name)
            data.mocap_pos[wheel_id] = (
                x_m,
                y_m,
                self.calibration.wheel_radius_m,
            )
            steer_quaternion = axis_angle_quaternion(
                (0.0, 0.0, 1.0),
                steering,
            )
            data.mocap_quat[wheel_id] = quaternion_multiply(
                quaternion_multiply(yaw_quaternion, steer_quaternion),
                spin_quaternion,
            )
