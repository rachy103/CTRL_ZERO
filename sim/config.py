from __future__ import annotations

import json
import math
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


SIM_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SIM_ROOT / "config" / "sim_config.json"


class ConfigurationError(RuntimeError):
    """Raised when a required measured value is absent or invalid."""


def sim_path(value: str | os.PathLike[str], *, must_exist: bool = False) -> Path:
    """Resolve a path and reject every destination outside ``sim``."""

    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (SIM_ROOT / candidate).resolve()
    try:
        resolved.relative_to(SIM_ROOT)
    except ValueError as exc:
        raise ConfigurationError(f"Path escapes the sim directory: {resolved}") from exc
    if must_exist and not resolved.exists():
        raise ConfigurationError(f"Required sim file does not exist: {resolved}")
    return resolved


def readable_path(value: str | os.PathLike[str], *, base: Path = SIM_ROOT) -> Path:
    """Resolve a read-only input path; unlike outputs it may be outside ``sim``."""

    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"Configuration root must be an object: {path}")
    return value


def _deep_merge(target: dict[str, Any], overlay: dict[str, Any]) -> None:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = deepcopy(value)


def load_config(path: str | os.PathLike[str] = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = sim_path(path, must_exist=True)
    config = _read_json(config_path)
    if config.get("schema_version") != 1:
        raise ConfigurationError(
            f"Unsupported schema_version={config.get('schema_version')!r}; expected 1"
        )

    calibration_value = get_value(config, "vehicle.geometry_calibration")
    if calibration_value:
        calibration_path = sim_path(str(calibration_value), must_exist=True)
        calibration = _read_json(calibration_path)
        if calibration.get("schema_version") != 1:
            raise ConfigurationError(
                f"Unsupported vehicle calibration schema in {calibration_path}"
            )
        geometry = calibration.get("vehicle_geometry", calibration)
        if not isinstance(geometry, dict):
            raise ConfigurationError(
                f"vehicle_geometry must be an object: {calibration_path}"
            )
        expected_geometry_keys = {
            "rear_axle_from_rear_m",
            "front_axle_from_rear_m",
            "wheelbase_m",
            "wheel_radius_m",
        }
        if not expected_geometry_keys.issubset(geometry):
            missing_keys = sorted(expected_geometry_keys.difference(geometry))
            raise ConfigurationError(
                f"Vehicle calibration is missing keys in {calibration_path}: "
                + ", ".join(missing_keys)
            )
        vehicle_config = config.get("vehicle")
        if not isinstance(vehicle_config, dict):
            raise ConfigurationError(
                f"vehicle must be an object: {config_path}"
            )
        mergeable_geometry_keys = {
            *expected_geometry_keys,
            "front_track_m",
            "rear_track_m",
        }
        measured_geometry = {
            key: deepcopy(geometry[key])
            for key in mergeable_geometry_keys
            if key in geometry and is_known(geometry[key])
        }
        _deep_merge(vehicle_config, measured_geometry)

        # Preserve measurement provenance without copying calibration-file
        # schema/status fields into the vehicle configuration. Older flat
        # calibration payloads placed this metadata beside the geometry keys;
        # wrapped payloads may place it either at the root or in the wrapper.
        calibration_metadata = calibration.get("calibration")
        if not isinstance(calibration_metadata, dict):
            calibration_metadata = geometry.get("calibration")
        if isinstance(calibration_metadata, dict):
            existing_metadata = vehicle_config.get("calibration")
            if not isinstance(existing_metadata, dict):
                existing_metadata = {}
                vehicle_config["calibration"] = existing_metadata
            _deep_merge(existing_metadata, calibration_metadata)
    config["_config_path"] = str(config_path)
    return config


def get_value(config: dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def set_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current: dict[str, Any] = config
    for part in parts[:-1]:
        child = current.get(part)
        if child is None:
            child = {}
            current[part] = child
        if not isinstance(child, dict):
            raise ConfigurationError(f"Cannot set {dotted_key}; {part} is not an object")
        current = child
    current[parts[-1]] = value


def is_known(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return bool(value) and all(is_known(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


MAP_REQUIRED = (
    "map.physical_width_m",
    "map.physical_height_m",
)

VEHICLE_GEOMETRY_REQUIRED = (
    "vehicle.body_length_m",
    "vehicle.body_width_m",
    "vehicle.body_height_m",
    "vehicle.rear_axle_from_rear_m",
    "vehicle.front_axle_from_rear_m",
    "vehicle.wheelbase_m",
    "vehicle.front_track_m",
    "vehicle.rear_track_m",
    "vehicle.wheel_radius_m",
    "vehicle.wheel_width_m",
)

KINEMATIC_HARDWARE_REQUIRED = (
    "hardware.command.steer_min",
    "hardware.command.steer_max",
    "hardware.command.drive_pwm_min",
    "hardware.command.drive_pwm_max",
    "hardware.command.drive_pwm_deadband",
    "hardware.steering.straight_command",
    "hardware.steering.left_angle_deg_at_steer_min",
    "hardware.steering.right_angle_deg_at_steer_max",
    "hardware.steering.command_slew_per_second",
    "hardware.drive.driven_axle",
    "hardware.drive.pwm_speed_table",
    "hardware.drive.speed_response_mode",
    "hardware.drive.zero_pwm_behavior",
)

SPAWN_REQUIRED = (
    "map.spawn.rear_axle_x_m",
    "map.spawn.rear_axle_y_m",
    "map.spawn.yaw_deg",
)

AUTO_CAMERA_REQUIRED = (
    "hardware.camera.width_px",
    "hardware.camera.height_px",
    "hardware.camera.fps",
    "hardware.camera.horizontal_fov_deg",
    "hardware.camera.position_from_rear_axle_m",
    "hardware.camera.pitch_deg",
    "hardware.camera.yaw_deg",
)


def missing_parameters(
    config: dict[str, Any],
    *,
    include_spawn: bool = True,
    auto: bool = False,
) -> list[str]:
    required: list[str] = [
        *MAP_REQUIRED,
        *VEHICLE_GEOMETRY_REQUIRED,
        *KINEMATIC_HARDWARE_REQUIRED,
    ]
    if include_spawn:
        required.extend(SPAWN_REQUIRED)
    if auto:
        required.extend(AUTO_CAMERA_REQUIRED)
        required.append("auto.yolo_model")
    if get_value(config, "hardware.drive.speed_response_mode") == "rate_limited":
        required.extend(
            (
                "hardware.drive.acceleration_limit_mps2",
                "hardware.drive.braking_deceleration_mps2",
            )
        )
    return [key for key in required if not is_known(get_value(config, key))]


def require_parameters(
    config: dict[str, Any],
    keys: Iterable[str],
    *,
    context: str,
) -> None:
    missing = [key for key in keys if not is_known(get_value(config, key))]
    if missing:
        lines = "\n".join(f"  - {key}" for key in missing)
        raise ConfigurationError(
            f"{context} cannot run because measured parameters are missing:\n{lines}\n"
            "No fallback or estimated value was substituted."
        )


def write_json_in_sim(path: str | os.PathLike[str], value: dict[str, Any]) -> Path:
    destination = sim_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination
