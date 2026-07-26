from __future__ import annotations

import copy
import math
import os
import shutil
import struct
import uuid
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from typing import Any

import pytest

from sim import model_builder
from sim.config import SIM_ROOT, ConfigurationError


@pytest.fixture
def sim_temp_dir() -> Path:
    """Create and clean a test workspace strictly below ``sim``."""

    directory = (
        SIM_ROOT / ".tmp" / "tests" / f"model-builder-{uuid.uuid4().hex}"
    )
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


def _camera_values() -> dict[str, Any]:
    return {
        "position_from_rear_axle_m": [0.24, 0.0, 0.31],
        "width_px": 640,
        "height_px": 480,
        "fps": 30.0,
        "horizontal_fov_deg": 70.0,
        "pitch_deg": -12.0,
        "yaw_deg": 1.5,
    }


def _config(*, camera: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "map": {
            "physical_width_m": 9.0,
            "physical_height_m": 13.0,
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
            "camera": copy.deepcopy(
                camera
                if camera is not None
                else {
                    "position_from_rear_axle_m": None,
                    "width_px": None,
                    "height_px": None,
                    "fps": None,
                    "horizontal_fov_deg": None,
                    "pitch_deg": None,
                    "yaw_deg": None,
                }
            )
        },
    }


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _write_texture(path: Path) -> Path:
    """Write a small standards-compliant RGB PNG without external libraries."""

    path.parent.mkdir(parents=True, exist_ok=True)
    width = 4
    height = 4
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            row.extend((45 + x * 12, 95 + y * 15, 55))
        rows.append(bytes(row))
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
        + _png_chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(payload)
    return path


def _build_mjcf(
    sim_temp_dir: Path,
    config: dict[str, Any],
    *,
    name: str = "world.xml",
) -> Path:
    texture = _write_texture(sim_temp_dir / "track.png")
    return model_builder.build_mjcf(
        config,
        sim_temp_dir / name,
        texture_path=texture,
    )


def test_all_null_camera_is_omitted_from_mjcf(sim_temp_dir: Path) -> None:
    config = _config()
    config["hardware"]["camera"]["position_from_rear_axle_m"] = [
        None,
        None,
        None,
    ]

    world_path = _build_mjcf(sim_temp_dir, config)
    root = ET.parse(world_path).getroot()

    assert root.findall(".//camera") == []


def test_partial_camera_configuration_is_rejected(sim_temp_dir: Path) -> None:
    config = _config()
    config["hardware"]["camera"]["width_px"] = 640

    with pytest.raises(ConfigurationError, match="Camera configuration is partial"):
        _build_mjcf(sim_temp_dir, config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "position_from_rear_axle_m",
            [0.2, 0.0],
            "exactly three finite numbers",
        ),
        (
            "position_from_rear_axle_m",
            [0.2, math.nan, 0.3],
            "must be a finite number",
        ),
        ("width_px", 640.5, "must be a positive integer"),
        ("height_px", 0, "must be a positive integer"),
        ("horizontal_fov_deg", 180.0, "must be between 0 and 180"),
        ("pitch_deg", "down", "must be a finite number"),
        ("yaw_deg", math.inf, "must be a finite number"),
    ],
)
def test_invalid_camera_configuration_is_rejected(
    sim_temp_dir: Path,
    field: str,
    value: Any,
    message: str,
) -> None:
    camera = _camera_values()
    camera[field] = value

    with pytest.raises(ConfigurationError, match=message):
        _build_mjcf(sim_temp_dir, _config(camera=camera))


def test_valid_camera_is_written_to_mjcf(sim_temp_dir: Path) -> None:
    camera_values = _camera_values()
    world_path = _build_mjcf(
        sim_temp_dir,
        _config(camera=camera_values),
    )
    camera = ET.parse(world_path).getroot().find(
        ".//camera[@name='onboard']"
    )

    assert camera is not None
    assert [float(value) for value in camera.attrib["pos"].split()] == (
        pytest.approx(camera_values["position_from_rear_axle_m"])
    )
    expected_vertical_fov = 2.0 * math.degrees(
        math.atan(
            math.tan(
                math.radians(camera_values["horizontal_fov_deg"] / 2.0)
            )
            * camera_values["height_px"]
            / camera_values["width_px"]
        )
    )
    assert float(camera.attrib["fovy"]) == pytest.approx(
        expected_vertical_fov,
        rel=1e-8,
    )
    assert len(camera.attrib["xyaxes"].split()) == 6


def test_visual_camera_does_not_require_capture_fps(sim_temp_dir: Path) -> None:
    camera_values = _camera_values()
    camera_values["fps"] = None

    world_path = _build_mjcf(
        sim_temp_dir,
        _config(camera=camera_values),
    )

    assert ET.parse(world_path).getroot().find(
        ".//camera[@name='onboard']"
    ) is not None


def test_urdf_is_visual_only_single_link_artifact(sim_temp_dir: Path) -> None:
    urdf_path = model_builder.build_urdf(
        _config(),
        sim_temp_dir / "vehicle.urdf",
    )
    root = ET.parse(urdf_path).getroot()
    links = root.findall("./link")

    assert root.tag == "robot"
    assert len(links) == 1
    assert links[0].attrib["name"] == "rear_axle_frame"
    assert len(links[0].findall("./visual")) == 5
    assert len(links[0].findall("./visual/geometry/box")) == 1
    assert len(links[0].findall("./visual/geometry/cylinder")) == 4
    assert root.findall(".//inertial") == []
    assert root.findall(".//collision") == []
    assert root.findall(".//joint") == []
    assert root.findall(".//limit") == []

    compiler = root.find("./mujoco/compiler")
    assert compiler is not None
    assert compiler.attrib["discardvisual"] == "false"


def test_mujoco_loads_and_steps_mjcf_and_urdf(sim_temp_dir: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    world_path = _build_mjcf(
        sim_temp_dir,
        _config(camera=_camera_values()),
    )
    urdf_path = model_builder.build_urdf(
        _config(),
        sim_temp_dir / "vehicle.urdf",
    )

    world_model = mujoco.MjModel.from_xml_path(str(world_path))
    world_data = mujoco.MjData(world_model)
    mujoco.mj_step(world_model, world_data)
    assert world_data.time == pytest.approx(world_model.opt.timestep)
    assert world_model.ngeom >= 6
    assert mujoco.mj_name2id(
        world_model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        "onboard",
    ) >= 0

    urdf_model = mujoco.MjModel.from_xml_path(str(urdf_path))
    urdf_data = mujoco.MjData(urdf_model)
    mujoco.mj_step(urdf_model, urdf_data)
    assert urdf_data.time == pytest.approx(urdf_model.opt.timestep)
    assert urdf_model.ngeom == 5


def test_measured_hd_camera_sets_and_uses_matching_offscreen_buffer(
    sim_temp_dir: Path,
) -> None:
    mujoco = pytest.importorskip("mujoco")
    camera = _camera_values()
    camera.update({"width_px": 1280, "height_px": 720})
    world_path = _build_mjcf(sim_temp_dir, _config(camera=camera))
    root = ET.parse(world_path).getroot()
    global_visual = root.find("./visual/global")

    assert global_visual is not None
    assert global_visual.attrib["offwidth"] == "1280"
    assert global_visual.attrib["offheight"] == "720"

    model = mujoco.MjModel.from_xml_path(str(world_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=720, width=1280)
    try:
        renderer.update_scene(data, camera="onboard")
        frame = renderer.render()
    finally:
        renderer.close()
    assert frame.shape == (720, 1280, 3)


def _isolate_model_outputs(
    monkeypatch: pytest.MonkeyPatch,
    sim_temp_dir: Path,
) -> Path:
    def isolated_sim_path(
        value: str | os.PathLike[str],
        *,
        must_exist: bool = False,
    ) -> Path:
        candidate = Path(value)
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (sim_temp_dir / candidate).resolve()
        )
        try:
            resolved.relative_to(sim_temp_dir)
        except ValueError as exc:
            raise ConfigurationError(
                f"Path escapes the isolated sim test directory: {resolved}"
            ) from exc
        if must_exist and not resolved.exists():
            raise ConfigurationError(
                f"Required isolated sim test file does not exist: {resolved}"
            )
        return resolved

    generated_dir = isolated_sim_path("assets/generated")
    monkeypatch.setattr(model_builder, "sim_path", isolated_sim_path)
    monkeypatch.setattr(model_builder, "GENERATED_DIR", generated_dir)
    return generated_dir


def test_build_models_writes_complete_set_in_isolated_sim_tree(
    sim_temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_dir = _isolate_model_outputs(monkeypatch, sim_temp_dir)
    _write_texture(generated_dir / "track_texture.png")

    outputs = model_builder.build_models(_config())

    assert outputs == {
        "mjcf": generated_dir / "world.xml",
        "urdf": generated_dir / "vehicle.urdf",
        "metadata": generated_dir / "vehicle_model_metadata.json",
    }
    assert all(path.is_file() for path in outputs.values())
    assert list(generated_dir.glob(".model-build-*")) == []


def test_build_models_restores_every_previous_output_if_install_fails(
    sim_temp_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_dir = _isolate_model_outputs(monkeypatch, sim_temp_dir)
    _write_texture(generated_dir / "track_texture.png")
    previous = {
        generated_dir / "world.xml": b"previous world\n",
        generated_dir / "vehicle.urdf": b"previous urdf\n",
        generated_dir
        / "vehicle_model_metadata.json": b'{"previous": true}\n',
    }
    for path, content in previous.items():
        path.write_bytes(content)

    real_replace = os.replace
    final_urdf = (generated_dir / "vehicle.urdf").resolve()

    def fail_during_urdf_install(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
    ) -> None:
        source_path = Path(source).resolve()
        destination_path = Path(destination).resolve()
        if (
            destination_path == final_urdf
            and source_path.name.startswith(".vehicle.urdf.install-")
        ):
            raise OSError("simulated atomic install failure")
        real_replace(source, destination)

    monkeypatch.setattr(model_builder.os, "replace", fail_during_urdf_install)

    with pytest.raises(OSError, match="simulated atomic install failure"):
        model_builder.build_models(_config())

    assert {
        path: path.read_bytes() for path in previous
    } == previous
    assert list(generated_dir.glob(".model-build-*")) == []
