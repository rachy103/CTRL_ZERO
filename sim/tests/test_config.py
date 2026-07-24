from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest

from sim.config import (
    DEFAULT_CONFIG_PATH,
    SIM_ROOT,
    ConfigurationError,
    get_value,
    load_config,
    missing_parameters,
    require_parameters,
    sim_path,
)


@pytest.fixture
def sim_temp_dir() -> Path:
    """Create test files strictly below sim and remove them after the test."""

    directory = SIM_ROOT / ".tmp" / "tests" / f"config-{uuid.uuid4().hex}"
    directory.mkdir(parents=True, exist_ok=False)
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)
        for empty_parent in (directory.parent, directory.parent.parent):
            try:
                empty_parent.rmdir()
            except OSError:
                pass


@pytest.mark.parametrize(
    "value",
    [
        "../requirements.txt",
        SIM_ROOT.parent / "requirements.txt",
    ],
)
def test_sim_path_rejects_paths_outside_sim(value: str | Path) -> None:
    with pytest.raises(ConfigurationError, match="escapes the sim directory"):
        sim_path(value)


def test_load_config_rejects_an_outside_config_path() -> None:
    with pytest.raises(ConfigurationError, match="escapes the sim directory"):
        load_config(SIM_ROOT.parent / "requirements.txt")


def test_actual_null_parameters_are_hard_errors_without_mutating_config() -> None:
    original_text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    config = load_config()

    expected_missing = {
        "hardware.steering.left_angle_deg_at_steer_min",
        "hardware.steering.right_angle_deg_at_steer_max",
        "hardware.drive.pwm_speed_table",
        "hardware.drive.acceleration_limit_mps2",
        "hardware.drive.braking_deceleration_mps2",
        "hardware.drive.zero_pwm_behavior",
    }
    assert expected_missing.issubset(set(missing_parameters(config)))
    assert get_value(config, "vehicle.wheelbase_m") > 0.0
    assert get_value(config, "vehicle.wheel_radius_m") > 0.0
    assert get_value(config, "vehicle.front_track_m") == pytest.approx(0.47)
    assert get_value(config, "vehicle.rear_track_m") == pytest.approx(0.47)
    assert get_value(config, "vehicle.wheel_width_m") == pytest.approx(0.08)
    assert get_value(config, "map.physical_width_m") == pytest.approx(13.0)
    assert get_value(config, "map.physical_height_m") == pytest.approx(9.0)
    assert get_value(config, "hardware.steering.straight_command") == 0
    assert get_value(config, "hardware.drive.driven_axle") == "rear"

    with pytest.raises(ConfigurationError) as captured:
        require_parameters(
            config,
            (
                "hardware.steering.left_angle_deg_at_steer_min",
                "hardware.drive.pwm_speed_table",
            ),
            context="Null policy test",
        )

    message = str(captured.value)
    assert "hardware.steering.left_angle_deg_at_steer_min" in message
    assert "hardware.drive.pwm_speed_table" in message
    assert "No fallback or estimated value was substituted." in message
    assert get_value(
        config, "hardware.steering.left_angle_deg_at_steer_min"
    ) is None
    assert get_value(config, "hardware.drive.pwm_speed_table") is None
    assert DEFAULT_CONFIG_PATH.read_text(encoding="utf-8") == original_text


@pytest.mark.parametrize("wrapped_vehicle_geometry", [False, True])
def test_geometry_calibration_deep_merges_without_replacing_vehicle(
    sim_temp_dir: Path,
    wrapped_vehicle_geometry: bool,
) -> None:
    measured_geometry = {
        "rear_axle_from_rear_m": 0.21,
        "front_axle_from_rear_m": 0.81,
        "wheelbase_m": 0.60,
        "wheel_radius_m": 0.10,
        "front_track_m": 0.46,
        "rear_track_m": 0.45,
        "calibration": {
            "method": "orthogonal_projection",
            "scale_m_per_px": 0.00495,
        },
    }
    calibration_payload = (
        {
            "schema_version": 1,
            "vehicle_geometry": measured_geometry,
        }
        if wrapped_vehicle_geometry
        else {
            "schema_version": 1,
            **measured_geometry,
        }
    )
    calibration_path = sim_temp_dir / "vehicle_geometry.json"
    calibration_path.write_text(
        json.dumps(calibration_payload),
        encoding="utf-8",
    )
    config_path = sim_temp_dir / "sim_config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "vehicle": {
                    "body_length_m": 0.99,
                    "body_width_m": 0.55,
                    "geometry_calibration": calibration_path.relative_to(
                        SIM_ROOT
                    ).as_posix(),
                    "calibration": {
                        "source_image": "assets/source/vehicle_axle_reference.png"
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    merged = load_config(config_path.relative_to(SIM_ROOT))

    assert get_value(merged, "vehicle.body_length_m") == pytest.approx(0.99)
    assert get_value(merged, "vehicle.body_width_m") == pytest.approx(0.55)
    assert get_value(merged, "vehicle.rear_axle_from_rear_m") == pytest.approx(
        0.21
    )
    assert get_value(merged, "vehicle.front_axle_from_rear_m") == pytest.approx(
        0.81
    )
    assert get_value(merged, "vehicle.wheelbase_m") == pytest.approx(0.60)
    assert get_value(merged, "vehicle.wheel_radius_m") == pytest.approx(0.10)
    assert get_value(merged, "vehicle.front_track_m") == pytest.approx(0.46)
    assert get_value(merged, "vehicle.rear_track_m") == pytest.approx(0.45)
    assert get_value(merged, "vehicle.calibration.source_image") == (
        "assets/source/vehicle_axle_reference.png"
    )
    assert get_value(merged, "vehicle.calibration.method") == (
        "orthogonal_projection"
    )
    assert get_value(merged, "vehicle.calibration.scale_m_per_px") == pytest.approx(
        0.00495
    )


def test_unknown_photo_radius_does_not_erase_separate_physical_measurement(
    sim_temp_dir: Path,
) -> None:
    calibration_path = sim_temp_dir / "perspective_geometry.json"
    calibration_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "vehicle_geometry": {
                    "rear_axle_from_rear_m": 0.21,
                    "front_axle_from_rear_m": 0.81,
                    "wheelbase_m": 0.60,
                    "front_track_m": 0.46,
                    "rear_track_m": 0.45,
                    "wheel_radius_m": None,
                },
            }
        ),
        encoding="utf-8",
    )
    config_path = sim_temp_dir / "physical_radius_config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "vehicle": {
                    "geometry_calibration": calibration_path.relative_to(
                        SIM_ROOT
                    ).as_posix(),
                    "wheel_radius_m": 0.11,
                },
            }
        ),
        encoding="utf-8",
    )

    merged = load_config(config_path)

    assert get_value(merged, "vehicle.wheel_radius_m") == pytest.approx(0.11)
    assert get_value(merged, "vehicle.front_track_m") == pytest.approx(0.46)
    assert get_value(merged, "vehicle.rear_track_m") == pytest.approx(0.45)


def test_geometry_calibration_path_cannot_escape_sim(sim_temp_dir: Path) -> None:
    config_path = sim_temp_dir / "escaping_config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "vehicle": {
                    "geometry_calibration": "../models/vehicle_geometry.json"
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="escapes the sim directory"):
        load_config(config_path)


def test_geometry_calibration_file_must_exist(sim_temp_dir: Path) -> None:
    missing_calibration = sim_temp_dir / "not_created.json"
    config_path = sim_temp_dir / "missing_calibration_config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "vehicle": {
                    "geometry_calibration": missing_calibration.relative_to(
                        SIM_ROOT
                    ).as_posix()
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="does not exist"):
        load_config(config_path)


@pytest.mark.parametrize(
    "missing_key",
    [
        "rear_axle_from_rear_m",
        "front_axle_from_rear_m",
        "wheelbase_m",
        "wheel_radius_m",
    ],
)
def test_partial_geometry_calibration_is_rejected(
    sim_temp_dir: Path,
    missing_key: str,
) -> None:
    geometry = {
        "rear_axle_from_rear_m": 0.21,
        "front_axle_from_rear_m": 0.81,
        "wheelbase_m": 0.60,
        "wheel_radius_m": 0.10,
    }
    geometry.pop(missing_key)
    calibration_path = sim_temp_dir / f"partial-{missing_key}.json"
    calibration_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "vehicle_geometry": geometry,
            }
        ),
        encoding="utf-8",
    )
    config_path = sim_temp_dir / f"partial-config-{missing_key}.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "vehicle": {
                    "geometry_calibration": calibration_path.relative_to(
                        SIM_ROOT
                    ).as_posix()
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match=missing_key):
        load_config(config_path)
