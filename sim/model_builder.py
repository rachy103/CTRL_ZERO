from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

try:
    from .config import (
        ConfigurationError,
        MAP_REQUIRED,
        VEHICLE_GEOMETRY_REQUIRED,
        get_value,
        require_parameters,
        sim_path,
        write_json_in_sim,
    )
except ImportError:  # Direct execution through sim_run.py
    from config import (
        ConfigurationError,
        MAP_REQUIRED,
        VEHICLE_GEOMETRY_REQUIRED,
        get_value,
        require_parameters,
        sim_path,
        write_json_in_sim,
    )


GENERATED_DIR = sim_path("assets/generated")
CAMERA_FIELDS = {
    "position": "hardware.camera.position_from_rear_axle_m",
    "width_px": "hardware.camera.width_px",
    "height_px": "hardware.camera.height_px",
    "fps": "hardware.camera.fps",
    "horizontal_fov_deg": "hardware.camera.horizontal_fov_deg",
    "pitch_deg": "hardware.camera.pitch_deg",
    "yaw_deg": "hardware.camera.yaw_deg",
}


def _number(value: float) -> str:
    return f"{float(value):.9g}"


def _vector(*values: float) -> str:
    return " ".join(_number(value) for value in values)


def _indent_and_write(root: ET.Element, destination: Path) -> None:
    ET.indent(root, space="  ")
    payload = ET.tostring(root, encoding="unicode", xml_declaration=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n' + payload + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _validated_geometry(config: dict[str, Any]) -> dict[str, float]:
    require_parameters(
        config,
        VEHICLE_GEOMETRY_REQUIRED,
        context="Vehicle model generation",
    )
    geometry = {
        key: float(get_value(config, f"vehicle.{key}"))
        for key in (
            "body_length_m",
            "body_width_m",
            "body_height_m",
            "rear_axle_from_rear_m",
            "front_axle_from_rear_m",
            "wheelbase_m",
            "front_track_m",
            "rear_track_m",
            "wheel_radius_m",
            "wheel_width_m",
        )
    }
    nonpositive = [key for key, value in geometry.items() if value <= 0]
    if nonpositive:
        raise ConfigurationError(
            "Vehicle dimensions must be positive: " + ", ".join(nonpositive)
        )
    if not math.isclose(
        geometry["front_axle_from_rear_m"] - geometry["rear_axle_from_rear_m"],
        geometry["wheelbase_m"],
        rel_tol=0.02,
        abs_tol=0.005,
    ):
        raise ConfigurationError(
            "wheelbase_m must equal front_axle_from_rear_m - rear_axle_from_rear_m."
        )
    if not (
        0
        < geometry["rear_axle_from_rear_m"]
        < geometry["front_axle_from_rear_m"]
        < geometry["body_length_m"]
    ):
        raise ConfigurationError(
            "Measured axle positions must lie inside the 0.99 m body, rear before front."
        )
    return geometry


def _load_visual_mesh_manifest(
    config: dict[str, Any],
    geometry: dict[str, float],
) -> dict[str, Any] | None:
    manifest_value = get_value(config, "vehicle.visual_mesh_manifest")
    if manifest_value is None or not str(manifest_value).strip():
        return None
    manifest_path = sim_path(str(manifest_value), must_exist=True)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(
            f"Invalid vehicle visual mesh manifest: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ConfigurationError(
            f"Unsupported vehicle visual mesh manifest: {manifest_path}"
        )
    target = manifest.get("target_geometry_m")
    if not isinstance(target, dict):
        raise ConfigurationError(
            f"Visual mesh manifest has no target_geometry_m: {manifest_path}"
        )
    expected = {
        "body_length": geometry["body_length_m"],
        "body_width": geometry["body_width_m"],
        "body_height": geometry["body_height_m"],
        "rear_axle_from_rear": geometry["rear_axle_from_rear_m"],
        "front_axle_from_rear": geometry["front_axle_from_rear_m"],
        "wheelbase": geometry["wheelbase_m"],
        "front_track": geometry["front_track_m"],
        "rear_track": geometry["rear_track_m"],
        "wheel_radius": geometry["wheel_radius_m"],
        "wheel_width": geometry["wheel_width_m"],
    }
    stale = [
        key
        for key, value in expected.items()
        if key not in target
        or not math.isclose(
            float(target[key]),
            value,
            rel_tol=1e-7,
            abs_tol=1e-7,
        )
    ]
    if stale:
        raise ConfigurationError(
            "Vehicle visual mesh is stale for: "
            + ", ".join(stale)
            + ". Rerun prepare_vehicle_mesh.py with Blender."
        )
    return manifest


def _register_visual_mesh_assets(
    asset: ET.Element,
    manifest: dict[str, Any],
    *,
    asset_directory: Path,
) -> tuple[list[tuple[str, str]], dict[str, list[tuple[str, str]]]]:
    materials = manifest.get("materials")
    if not isinstance(materials, dict):
        raise ConfigurationError("Visual mesh manifest materials must be an object.")
    for category, properties in sorted(materials.items()):
        if not isinstance(properties, dict):
            raise ConfigurationError(
                f"Visual mesh material must be an object: {category}"
            )
        rgba = properties.get("rgba")
        if not isinstance(rgba, list) or len(rgba) != 4:
            raise ConfigurationError(
                f"Visual mesh material rgba must have four values: {category}"
            )
        ET.SubElement(
            asset,
            "material",
            {
                "name": f"vehicle_mesh_material_{category}",
                "rgba": _vector(*(float(value) for value in rgba)),
                "specular": "0.25",
                "shininess": "0.3",
            },
        )

    def register_entries(
        entries: Any,
        scope: str,
    ) -> list[tuple[str, str]]:
        if not isinstance(entries, list):
            raise ConfigurationError(
                f"Visual mesh entries must be a list: {scope}"
            )
        registered: list[tuple[str, str]] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ConfigurationError(
                    f"Visual mesh entry must be an object: {scope}[{index}]"
                )
            category = str(entry.get("material", ""))
            if category not in materials:
                raise ConfigurationError(
                    f"Unknown visual mesh material {category!r} in {scope}"
                )
            mesh_path = sim_path(str(entry.get("file", "")), must_exist=True)
            mesh_name = f"vehicle_mesh_{scope}_{index}"
            relative_file = os.path.relpath(
                mesh_path,
                asset_directory,
            ).replace("\\", "/")
            ET.SubElement(
                asset,
                "mesh",
                {"name": mesh_name, "file": relative_file},
            )
            registered.append(
                (mesh_name, f"vehicle_mesh_material_{category}")
            )
        return registered

    body_meshes = register_entries(manifest.get("body_meshes"), "body")
    wheels_value = manifest.get("wheel_meshes")
    if not isinstance(wheels_value, dict):
        raise ConfigurationError(
            "Visual mesh manifest wheel_meshes must be an object."
        )
    wheel_meshes = {
        wheel_name: register_entries(
            wheels_value.get(wheel_name),
            wheel_name,
        )
        for wheel_name in (
            "front_left_wheel",
            "front_right_wheel",
            "rear_left_wheel",
            "rear_right_wheel",
        )
    }
    return body_meshes, wheel_meshes


def _is_explicit_null(value: Any) -> bool:
    """Treat a position made only of nulls as an unconfigured camera field."""

    if value is None:
        return True
    if isinstance(value, (list, tuple)) and value:
        return all(_is_explicit_null(item) for item in value)
    return False


def _finite_number(value: Any, dotted_key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{dotted_key} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigurationError(f"{dotted_key} must be a finite number.")
    return number


def _validated_camera(config: dict[str, Any]) -> dict[str, Any] | None:
    raw = {name: get_value(config, key) for name, key in CAMERA_FIELDS.items()}
    configured = {
        name: not _is_explicit_null(value) for name, value in raw.items()
    }
    if not any(configured.values()):
        return None
    if not all(configured.values()):
        missing = [
            CAMERA_FIELDS[name] for name, present in configured.items() if not present
        ]
        lines = "\n".join(f"  - {key}" for key in missing)
        raise ConfigurationError(
            "Camera configuration is partial. Set every camera field or set all of "
            f"them to null:\n{lines}"
        )

    position = raw["position"]
    position_key = CAMERA_FIELDS["position"]
    if (
        not isinstance(position, (list, tuple))
        or isinstance(position, (str, bytes))
        or len(position) != 3
    ):
        raise ConfigurationError(
            f"{position_key} must contain exactly three finite numbers."
        )
    validated_position = tuple(
        _finite_number(value, f"{position_key}[{index}]")
        for index, value in enumerate(position)
    )

    width = _finite_number(raw["width_px"], CAMERA_FIELDS["width_px"])
    height = _finite_number(raw["height_px"], CAMERA_FIELDS["height_px"])
    if width <= 0 or not width.is_integer():
        raise ConfigurationError(
            f"{CAMERA_FIELDS['width_px']} must be a positive integer."
        )
    if height <= 0 or not height.is_integer():
        raise ConfigurationError(
            f"{CAMERA_FIELDS['height_px']} must be a positive integer."
        )

    fps = _finite_number(raw["fps"], CAMERA_FIELDS["fps"])
    if fps <= 0:
        raise ConfigurationError(f"{CAMERA_FIELDS['fps']} must be greater than zero.")

    horizontal_fov = _finite_number(
        raw["horizontal_fov_deg"],
        CAMERA_FIELDS["horizontal_fov_deg"],
    )
    if not 0 < horizontal_fov < 180:
        raise ConfigurationError(
            f"{CAMERA_FIELDS['horizontal_fov_deg']} must be between 0 and 180 "
            "degrees (exclusive)."
        )

    pitch = _finite_number(raw["pitch_deg"], CAMERA_FIELDS["pitch_deg"])
    yaw = _finite_number(raw["yaw_deg"], CAMERA_FIELDS["yaw_deg"])
    return {
        "position": validated_position,
        "width_px": int(width),
        "height_px": int(height),
        "fps": fps,
        "horizontal_fov_deg": horizontal_fov,
        "pitch_deg": pitch,
        "yaw_deg": yaw,
    }


def _camera_xyaxes(yaw_deg: float, pitch_deg: float) -> str:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    forward = (
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
    )
    right = (math.sin(yaw), -math.cos(yaw), 0.0)
    up = (
        -math.sin(pitch) * math.cos(yaw),
        -math.sin(pitch) * math.sin(yaw),
        math.cos(pitch),
    )
    return _vector(*right, *up)


def _add_visual_wheel(
    worldbody: ET.Element,
    name: str,
    radius_m: float,
    width_m: float,
    mesh_entries: list[tuple[str, str]] | None = None,
) -> None:
    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": name,
            "mocap": "true",
            "pos": _vector(0, 0, radius_m),
        },
    )
    if mesh_entries:
        for index, (mesh_name, material_name) in enumerate(mesh_entries):
            ET.SubElement(
                body,
                "geom",
                {
                    "name": f"{name}_mesh_{index}",
                    "type": "mesh",
                    "mesh": mesh_name,
                    "material": material_name,
                    "contype": "0",
                    "conaffinity": "0",
                },
            )
    else:
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{name}_visual",
                "type": "cylinder",
                "size": _vector(radius_m, width_m / 2.0),
                "quat": _vector(math.sqrt(0.5), math.sqrt(0.5), 0, 0),
                "material": "tire_material",
                "contype": "0",
                "conaffinity": "0",
            },
        )
        ET.SubElement(
            body,
            "site",
            {
                "name": f"{name}_hub",
                "type": "cylinder",
                "size": _vector(radius_m * 0.22, width_m * 0.51),
                "quat": _vector(math.sqrt(0.5), math.sqrt(0.5), 0, 0),
                "rgba": _vector(0.72, 0.72, 0.75, 1),
            },
        )
    ET.SubElement(
        body,
        "site",
        {
            "name": f"{name}_rotation_marker",
            "type": "sphere",
            "pos": _vector(0, 0, radius_m * 0.65),
            "size": _number(radius_m * 0.075),
            "rgba": _vector(0.92, 0.22, 0.12, 1),
        },
    )


def build_mjcf(
    config: dict[str, Any],
    destination: Path,
    *,
    texture_path: Path | None = None,
) -> Path:
    require_parameters(config, MAP_REQUIRED, context="MuJoCo world generation")
    geometry = _validated_geometry(config)
    visual_mesh_manifest = _load_visual_mesh_manifest(config, geometry)
    camera = _validated_camera(config)
    texture_path = sim_path(
        texture_path or "assets/generated/track_texture.png",
        must_exist=True,
    )
    map_width = float(get_value(config, "map.physical_width_m"))
    map_height = float(get_value(config, "map.physical_height_m"))
    if map_width <= 0 or map_height <= 0:
        raise ConfigurationError("Map dimensions must be positive.")

    length = geometry["body_length_m"]
    width = geometry["body_width_m"]
    height = geometry["body_height_m"]
    rear_axle_from_rear = geometry["rear_axle_from_rear_m"]
    body_center_from_rear_axle = length / 2.0 - rear_axle_from_rear

    root = ET.Element("mujoco", {"model": "ctrl_zero_track"})
    ET.SubElement(
        root,
        "compiler",
        {
            "angle": "radian",
            "coordinate": "local",
            "assetdir": str(texture_path.parent),
        },
    )
    ET.SubElement(
        root,
        "option",
        {
            "timestep": "0.01",
            "gravity": "0 0 -9.81",
            "integrator": "RK4",
        },
    )
    ET.SubElement(
        root,
        "statistic",
        {
            "center": "0 0 0",
            "extent": _number(max(map_width, map_height) * 0.58),
        },
    )
    visual = ET.SubElement(root, "visual")
    if camera is not None:
        ET.SubElement(
            visual,
            "global",
            {
                "offwidth": str(camera["width_px"]),
                "offheight": str(camera["height_px"]),
            },
        )
    ET.SubElement(
        visual,
        "headlight",
        {"ambient": "0.55 0.55 0.55", "diffuse": "0.6 0.6 0.6"},
    )
    ET.SubElement(
        visual,
        "map",
        {"znear": "0.01", "zfar": _number(max(map_width, map_height) * 5.0)},
    )
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "track_texture",
            "type": "2d",
            "file": texture_path.name,
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "track_material",
            "texture": "track_texture",
            "texrepeat": "1 1",
            "texuniform": "false",
            "reflectance": "0",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "body_material",
            "rgba": "0.92 0.94 0.97 1",
            "specular": "0.35",
            "shininess": "0.45",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "tire_material",
            "rgba": "0.035 0.035 0.04 1",
            "specular": "0.05",
            "shininess": "0.1",
        },
    )
    body_visual_meshes: list[tuple[str, str]] = []
    wheel_visual_meshes: dict[str, list[tuple[str, str]]] = {}
    if visual_mesh_manifest is not None:
        body_visual_meshes, wheel_visual_meshes = (
            _register_visual_mesh_assets(
                asset,
                visual_mesh_manifest,
                asset_directory=texture_path.parent,
            )
        )

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "sun",
            "directional": "true",
            "pos": "0 0 10",
            "dir": "-0.2 -0.3 -1",
            "diffuse": "0.85 0.85 0.85",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "track_floor",
            "type": "plane",
            "size": _vector(map_width / 2.0, map_height / 2.0, 0.02),
            "material": "track_material",
            "contype": "0",
            "conaffinity": "0",
        },
    )

    vehicle = ET.SubElement(
        worldbody,
        "body",
        {"name": "vehicle_body", "mocap": "true", "pos": "0 0 0"},
    )
    if body_visual_meshes:
        for index, (mesh_name, material_name) in enumerate(body_visual_meshes):
            ET.SubElement(
                vehicle,
                "geom",
                {
                    "name": f"vehicle_body_mesh_{index}",
                    "type": "mesh",
                    "mesh": mesh_name,
                    "material": material_name,
                    "contype": "0",
                    "conaffinity": "0",
                },
            )
    else:
        ET.SubElement(
            vehicle,
            "geom",
            {
                "name": "vehicle_body_visual",
                "type": "ellipsoid",
                "pos": _vector(body_center_from_rear_axle, 0, height / 2.0),
                "size": _vector(length / 2.0, width / 2.0, height / 2.0),
                "material": "body_material",
                "contype": "0",
                "conaffinity": "0",
            },
        )
    ET.SubElement(
        vehicle,
        "site",
        {
            "name": "rear_axle_origin",
            "type": "sphere",
            "pos": "0 0 0.01",
            "size": "0.012",
            "rgba": "0.1 0.4 1 1",
        },
    )

    if camera is not None:
        camera_x, camera_y, camera_z = camera["position"]
        ET.SubElement(
            vehicle,
            "camera",
            {
                "name": "onboard",
                "mode": "fixed",
                "pos": _vector(camera_x, camera_y, camera_z),
                "xyaxes": _camera_xyaxes(
                    camera["yaw_deg"],
                    camera["pitch_deg"],
                ),
                "fovy": _number(
                    2.0
                    * math.degrees(
                        math.atan(
                            math.tan(
                                math.radians(
                                    camera["horizontal_fov_deg"]
                                    / 2.0
                                )
                            )
                            * camera["height_px"]
                            / camera["width_px"]
                        )
                    )
                ),
            },
        )

    for wheel_name in (
        "front_left_wheel",
        "front_right_wheel",
        "rear_left_wheel",
        "rear_right_wheel",
    ):
        _add_visual_wheel(
            worldbody,
            wheel_name,
            geometry["wheel_radius_m"],
            geometry["wheel_width_m"],
            wheel_visual_meshes.get(wheel_name),
        )

    destination = sim_path(destination)
    _indent_and_write(root, destination)
    return destination


def _add_urdf_visual(
    link: ET.Element,
    name: str,
    *,
    geometry_tag: str,
    geometry_attributes: dict[str, str],
    origin_xyz: str,
    origin_rpy: str = "0 0 0",
    material_name: str,
) -> None:
    visual = ET.SubElement(link, "visual", {"name": name})
    ET.SubElement(
        visual,
        "origin",
        {"xyz": origin_xyz, "rpy": origin_rpy},
    )
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, geometry_tag, geometry_attributes)
    ET.SubElement(visual, "material", {"name": material_name})


def build_urdf(config: dict[str, Any], destination: Path) -> Path:
    geometry = _validated_geometry(config)
    visual_mesh_manifest = _load_visual_mesh_manifest(config, geometry)
    length = geometry["body_length_m"]
    width = geometry["body_width_m"]
    height = geometry["body_height_m"]
    rear_offset = geometry["rear_axle_from_rear_m"]
    chassis_center_x = length / 2.0 - rear_offset
    wheel_radius = geometry["wheel_radius_m"]
    wheel_width = geometry["wheel_width_m"]

    robot = ET.Element("robot", {"name": "ctrl_zero_child_car"})
    mujoco = ET.SubElement(robot, "mujoco")
    ET.SubElement(mujoco, "compiler", {"discardvisual": "false"})
    ET.SubElement(
        robot,
        "material",
        {"name": "body_white"},
    ).append(ET.Element("color", {"rgba": "0.92 0.94 0.97 1"}))
    ET.SubElement(
        robot,
        "material",
        {"name": "tire_black"},
    ).append(ET.Element("color", {"rgba": "0.035 0.035 0.04 1"}))
    if visual_mesh_manifest is not None:
        for category, properties in sorted(
            visual_mesh_manifest["materials"].items()
        ):
            rgba = properties["rgba"]
            ET.SubElement(
                robot,
                "material",
                {"name": f"vehicle_mesh_material_{category}"},
            ).append(
                ET.Element(
                    "color",
                    {
                        "rgba": _vector(
                            *(float(value) for value in rgba)
                        )
                    },
                )
            )

    # The URDF is deliberately a single-link visual artifact. Adding inertial,
    # collision, joint effort, or velocity data here would invent unmeasured
    # hardware properties and would make MuJoCo interpret it as a physics model.
    link = ET.SubElement(robot, "link", {"name": "rear_axle_frame"})
    if visual_mesh_manifest is not None:
        for index, entry in enumerate(
            visual_mesh_manifest["body_meshes"]
        ):
            mesh_path = sim_path(str(entry["file"]), must_exist=True)
            relative_mesh = os.path.relpath(
                mesh_path,
                GENERATED_DIR,
            ).replace("\\", "/")
            _add_urdf_visual(
                link,
                f"chassis_mesh_{index}",
                geometry_tag="mesh",
                geometry_attributes={"filename": relative_mesh},
                origin_xyz="0 0 0",
                material_name=(
                    f"vehicle_mesh_material_{entry['material']}"
                ),
            )
    else:
        _add_urdf_visual(
            link,
            "chassis_visual",
            geometry_tag="box",
            geometry_attributes={"size": _vector(length, width, height)},
            origin_xyz=_vector(chassis_center_x, 0, height / 2.0),
            material_name="body_white",
        )

    wheel_positions = {
        "front_left_wheel_visual": (
            geometry["wheelbase_m"],
            geometry["front_track_m"] / 2.0,
            wheel_radius,
        ),
        "front_right_wheel_visual": (
            geometry["wheelbase_m"],
            -geometry["front_track_m"] / 2.0,
            wheel_radius,
        ),
        "rear_left_wheel_visual": (
            0,
            geometry["rear_track_m"] / 2.0,
            wheel_radius,
        ),
        "rear_right_wheel_visual": (
            0,
            -geometry["rear_track_m"] / 2.0,
            wheel_radius,
        ),
    }
    for name, position in wheel_positions.items():
        wheel_name = name.removesuffix("_visual")
        if visual_mesh_manifest is not None:
            for index, entry in enumerate(
                visual_mesh_manifest["wheel_meshes"][wheel_name]
            ):
                mesh_path = sim_path(str(entry["file"]), must_exist=True)
                relative_mesh = os.path.relpath(
                    mesh_path,
                    GENERATED_DIR,
                ).replace("\\", "/")
                _add_urdf_visual(
                    link,
                    f"{name}_mesh_{index}",
                    geometry_tag="mesh",
                    geometry_attributes={"filename": relative_mesh},
                    origin_xyz=_vector(*position),
                    material_name=(
                        f"vehicle_mesh_material_{entry['material']}"
                    ),
                )
        else:
            _add_urdf_visual(
                link,
                name,
                geometry_tag="cylinder",
                geometry_attributes={
                    "radius": _number(wheel_radius),
                    "length": _number(wheel_width),
                },
                origin_xyz=_vector(*position),
                origin_rpy=_vector(math.pi / 2.0, 0, 0),
                material_name="tire_black",
            )

    destination = sim_path(destination)
    _indent_and_write(robot, destination)
    return destination


def build_models(config: dict[str, Any]) -> dict[str, Path]:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    def usable_output_path(preferred_name: str, fallback_name: str) -> Path:
        preferred = sim_path(f"assets/generated/{preferred_name}")
        if preferred.exists():
            try:
                with preferred.open("rb"):
                    pass
            except PermissionError:
                return sim_path(f"assets/generated/{fallback_name}")
        return preferred

    world_path = usable_output_path("world.xml", "world_mesh.xml")
    urdf_path = usable_output_path("vehicle.urdf", "vehicle_mesh.urdf")
    metadata_path = usable_output_path(
        "vehicle_model_metadata.json",
        "vehicle_mesh_model_metadata.json",
    )
    final_paths = (world_path, urdf_path, metadata_path)

    # Build the complete set before touching any previous generated model. Each
    # final write uses os.replace; if one replacement fails, restore every old
    # file so callers never observe a mixed old/new model set after an error.
    with tempfile.TemporaryDirectory(
        prefix=".model-build-",
        dir=GENERATED_DIR,
    ) as temporary_name:
        staging_dir = sim_path(temporary_name)
        staged_world = build_mjcf(config, staging_dir / world_path.name)
        staged_urdf = build_urdf(config, staging_dir / urdf_path.name)
        staged_metadata = write_json_in_sim(
            staging_dir / metadata_path.name,
            {
                "schema_version": 1,
                "model_kind": "kinematic_visual",
                "mjcf": world_path.relative_to(sim_path(".")).as_posix(),
                "urdf": urdf_path.relative_to(sim_path(".")).as_posix(),
                "physics_note": (
                    "No guessed mass, inertia, tire friction, motor torque, joint "
                    "effort, or joint velocity is present. The runtime advances a "
                    "calibrated kinematic bicycle state and syncs MuJoCo mocap "
                    "visuals before mj_step(). The URDF is a visual-only single-link "
                    "artifact."
                ),
            },
        )
        staged_paths = (staged_world, staged_urdf, staged_metadata)
        backups: dict[Path, Path] = {}
        installed: list[Path] = []

        def install_with_parent_acl(source: Path, destination: Path) -> None:
            handle, install_name = tempfile.mkstemp(
                prefix=f".{destination.name}.install-",
                dir=destination.parent,
            )
            os.close(handle)
            installing = Path(install_name)
            try:
                shutil.copyfile(source, installing)
                os.replace(installing, destination)
            finally:
                if installing.exists():
                    installing.unlink()

        try:
            for final in final_paths:
                if final.exists():
                    backup = staging_dir / f".{final.name}.previous"
                    shutil.copy2(final, backup)
                    backups[final] = backup
            for staged, final in zip(staged_paths, final_paths, strict=True):
                install_with_parent_acl(staged, final)
                installed.append(final)
        except BaseException:
            for final in reversed(installed):
                backup = backups.get(final)
                if backup is not None and backup.exists():
                    install_with_parent_acl(backup, final)
                elif final.exists():
                    final.unlink()
            raise

    return {"mjcf": world_path, "urdf": urdf_path, "metadata": metadata_path}
