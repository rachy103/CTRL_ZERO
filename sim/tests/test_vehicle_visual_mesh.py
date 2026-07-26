from __future__ import annotations

import json
from pathlib import Path

import pytest

from sim.config import SIM_ROOT, get_value, load_config


MANIFEST_PATH = SIM_ROOT / "config" / "vehicle_visual_mesh.json"


def _bounds(entries: list[dict[str, object]]) -> tuple[list[float], list[float]]:
    minimum = [float("inf")] * 3
    maximum = [float("-inf")] * 3
    vertex_count = 0
    for entry in entries:
        mesh_path = (SIM_ROOT / str(entry["file"])).resolve(strict=True)
        mesh_path.relative_to(SIM_ROOT.resolve(strict=True))
        with mesh_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.startswith("v "):
                    continue
                point = [float(value) for value in line.split()[1:4]]
                vertex_count += 1
                for axis, value in enumerate(point):
                    minimum[axis] = min(minimum[axis], value)
                    maximum[axis] = max(maximum[axis], value)
    assert vertex_count > 0
    return minimum, maximum


def test_visual_mesh_manifest_matches_current_calibration() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    config = load_config()
    target = manifest["target_geometry_m"]

    expected = {
        "body_length": "vehicle.body_length_m",
        "body_width": "vehicle.body_width_m",
        "body_height": "vehicle.body_height_m",
        "rear_axle_from_rear": "vehicle.rear_axle_from_rear_m",
        "front_axle_from_rear": "vehicle.front_axle_from_rear_m",
        "wheelbase": "vehicle.wheelbase_m",
        "front_track": "vehicle.front_track_m",
        "rear_track": "vehicle.rear_track_m",
        "wheel_radius": "vehicle.wheel_radius_m",
        "wheel_width": "vehicle.wheel_width_m",
    }
    for manifest_key, config_key in expected.items():
        assert target[manifest_key] == pytest.approx(
            float(get_value(config, config_key))
        )


def test_body_visual_mesh_has_fixed_length_and_width() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    minimum, maximum = _bounds(manifest["body_meshes"])

    assert maximum[0] - minimum[0] == pytest.approx(0.99, abs=2e-7)
    assert maximum[1] - minimum[1] == pytest.approx(0.55, abs=2e-7)
    assert maximum[2] == pytest.approx(0.41, abs=2e-7)
    assert minimum[0] == pytest.approx(
        -manifest["target_geometry_m"]["rear_axle_from_rear"],
        abs=2e-7,
    )


def test_each_visual_wheel_has_measured_radius_and_width() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    target = manifest["target_geometry_m"]

    for entries in manifest["wheel_meshes"].values():
        minimum, maximum = _bounds(entries)
        assert maximum[1] - minimum[1] == pytest.approx(
            target["wheel_width"], abs=2e-7
        )
        assert maximum[0] - minimum[0] == pytest.approx(
            target["wheel_radius"] * 2.0,
            abs=5e-5,
        )
        assert maximum[2] - minimum[2] == pytest.approx(
            target["wheel_radius"] * 2.0,
            abs=5e-5,
        )
