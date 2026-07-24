"""Run with Blender to reshape the supplied AMG mesh for the measured toy car.

Example:
    blender --background --python sim/prepare_vehicle_mesh.py
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import bpy
from mathutils import Vector


SIM_ROOT = Path(__file__).resolve().parent
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from config import get_value, load_config, sim_path, write_json_in_sim  # noqa: E402


SOURCE_BLEND = sim_path("assets/source/vehicle_mesh/amg_gt_r_source.blend")
OUTPUT_DIR = sim_path("assets/generated/vehicle_mesh")
MANIFEST_PATH = sim_path("config/vehicle_visual_mesh.json")

WHEEL_KEYS = (
    "front_left_wheel",
    "front_right_wheel",
    "rear_left_wheel",
    "rear_right_wheel",
)

MATERIAL_RGBA = {
    "white": (0.92, 0.94, 0.97, 1.0),
    "black": (0.025, 0.025, 0.03, 1.0),
    "glass": (0.10, 0.16, 0.20, 0.62),
    "metal": (0.55, 0.57, 0.60, 1.0),
    "chrome": (0.78, 0.80, 0.83, 1.0),
    "red": (0.72, 0.015, 0.02, 1.0),
    "light": (0.78, 0.72, 0.50, 1.0),
    "tire": (0.018, 0.018, 0.022, 1.0),
}


def _world_vertices(obj: bpy.types.Object) -> list[Vector]:
    matrix = obj.matrix_world
    return [matrix @ vertex.co for vertex in obj.data.vertices]


def _bounds(objects: Iterable[bpy.types.Object]) -> tuple[Vector, Vector]:
    minimum = Vector((math.inf, math.inf, math.inf))
    maximum = Vector((-math.inf, -math.inf, -math.inf))
    found = False
    for obj in objects:
        for point in _world_vertices(obj):
            found = True
            for axis in range(3):
                minimum[axis] = min(minimum[axis], point[axis])
                maximum[axis] = max(maximum[axis], point[axis])
    if not found:
        raise RuntimeError("No mesh vertices were available for bounds.")
    return minimum, maximum


def _wheel_key(name: str) -> str | None:
    lowered = name.lower()
    prefixes = {
        "front left ": "front_left_wheel",
        "front right ": "front_right_wheel",
        "rear left ": "rear_left_wheel",
        "rear right ": "rear_right_wheel",
    }
    for prefix, key in prefixes.items():
        if lowered.startswith(prefix):
            return key
    rim_logo_names = {
        "mercedes logo rim": "front_left_wheel",
        "mercedes logo rim.001": "rear_left_wheel",
        "mercedes logo rim.002": "rear_right_wheel",
        "mercedes logo rim.003": "front_right_wheel",
    }
    return rim_logo_names.get(lowered)


def _material_category(
    obj: bpy.types.Object,
    material_name: str,
    wheel_key: str | None,
) -> str:
    lowered_object = obj.name.lower()
    lowered = material_name.lower().replace(" ", "_")
    if wheel_key and "tyre" in lowered_object:
        return "tire"
    if lowered in {"car_paint", "material.005"}:
        return "white"
    if lowered in {
        "gloss_black",
        "carbon_fiber",
        "carbon_fiber_interior",
        "leather",
        "alcantara",
    }:
        return "black"
    if lowered in {"glass", "reflective_glass"}:
        return "glass"
    if lowered == "chrome":
        return "chrome"
    if lowered in {"reflective_red", "red_emmision"}:
        return "red"
    if lowered in {
        "front_led",
        "yello_emission",
        "yellow_",
        "screen_emisson",
    }:
        return "light"
    if lowered in {"wheel", "metal", "diffuse_grey", "disc_pedel"}:
        return "metal"
    return "black"


def _piecewise_map(
    value: float,
    source: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
) -> float:
    for index in range(3):
        source_low = source[index]
        source_high = source[index + 1]
        if value <= source_high or index == 2:
            ratio = (value - source_low) / (source_high - source_low)
            return target[index] + ratio * (
                target[index + 1] - target[index]
            )
    raise AssertionError("unreachable")


def _safe_filename(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")
    return normalized or "mesh"


def _write_obj(
    destination: Path,
    triangles: list[tuple[Vector, Vector, Vector]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# CTRL_ZERO measured vehicle visual mesh\n")
        stream.write(f"o {_safe_filename(destination.stem)}\n")
        vertex_index = 1
        for triangle in triangles:
            for point in triangle:
                stream.write(
                    f"v {point.x:.9g} {point.y:.9g} {point.z:.9g}\n"
                )
            stream.write(
                f"f {vertex_index} {vertex_index + 1} "
                f"{vertex_index + 2}\n"
            )
            vertex_index += 3


def prepare_vehicle_mesh() -> dict[str, object]:
    if not SOURCE_BLEND.is_file():
        raise RuntimeError(f"Source blend is missing: {SOURCE_BLEND}")
    if Path(bpy.data.filepath).resolve() != SOURCE_BLEND.resolve():
        bpy.ops.wm.open_mainfile(filepath=str(SOURCE_BLEND))

    # The source uses Mirror/Subdivision and other non-destructive modifiers.
    # Several front parts contain only one modeled half, so every visual mesh
    # must be converted before export. Decimation happens only after conversion
    # and uses a per-object triangle cap so small logos and trim stay intact.
    # The source .blend is never saved or overwritten.
    bpy.ops.object.select_all(action="DESELECT")
    for source_object in list(bpy.data.objects):
        if (
            source_object.type != "MESH"
            or source_object.name.lower() == "backdrop"
        ):
            continue
        source_object.select_set(True)
        bpy.context.view_layer.objects.active = source_object
        bpy.ops.object.convert(target="MESH")
        polygon_limit = (
            6000 if _wheel_key(source_object.name) is not None else 12000
        )
        polygon_count = len(source_object.data.polygons)
        if polygon_count > polygon_limit:
            modifier = source_object.modifiers.new(
                name="CTRL_ZERO_lightweight_decimate",
                type="DECIMATE",
            )
            modifier.ratio = polygon_limit / polygon_count
            bpy.ops.object.convert(target="MESH")
        source_object.select_set(False)

    config = load_config()
    vehicle_length = float(get_value(config, "vehicle.body_length_m"))
    vehicle_width = float(get_value(config, "vehicle.body_width_m"))
    vehicle_height = float(get_value(config, "vehicle.body_height_m"))
    rear_axle_target = float(
        get_value(config, "vehicle.rear_axle_from_rear_m")
    )
    front_axle_target = float(
        get_value(config, "vehicle.front_axle_from_rear_m")
    )
    wheel_radius_target = float(get_value(config, "vehicle.wheel_radius_m"))
    wheel_width_target = float(get_value(config, "vehicle.wheel_width_m"))
    front_track_target = float(get_value(config, "vehicle.front_track_m"))
    rear_track_target = float(get_value(config, "vehicle.rear_track_m"))

    mesh_objects = [
        obj
        for obj in bpy.data.objects
        if obj.type == "MESH" and obj.name.lower() != "backdrop"
    ]
    body_objects = [obj for obj in mesh_objects if _wheel_key(obj.name) is None]
    wheel_objects: dict[str, list[bpy.types.Object]] = {
        key: [] for key in WHEEL_KEYS
    }
    for obj in mesh_objects:
        key = _wheel_key(obj.name)
        if key:
            wheel_objects[key].append(obj)

    tyre_objects = {
        key: next(
            obj for obj in objects if "tyre" in obj.name.lower()
        )
        for key, objects in wheel_objects.items()
    }
    tyre_bounds = {
        key: _bounds((obj,)) for key, obj in tyre_objects.items()
    }
    wheel_centers = {
        key: (minimum + maximum) * 0.5
        for key, (minimum, maximum) in tyre_bounds.items()
    }

    body_min, body_max = _bounds(body_objects)
    source_body_width = body_max.x - body_min.x
    if source_body_width <= 0:
        raise RuntimeError("Source vehicle body envelope has an invalid width.")

    source_rear_axle = -(
        wheel_centers["rear_left_wheel"].y
        + wheel_centers["rear_right_wheel"].y
    ) / 2.0
    source_front_axle = -(
        wheel_centers["front_left_wheel"].y
        + wheel_centers["front_right_wheel"].y
    ) / 2.0
    source_rear_end = -body_max.y
    source_front_end = -body_min.y
    source_knots = (
        source_rear_end,
        source_rear_axle,
        source_front_axle,
        source_front_end,
    )
    if not all(
        source_knots[index] < source_knots[index + 1]
        for index in range(3)
    ):
        raise RuntimeError(f"Invalid source longitudinal landmarks: {source_knots}")
    target_knots = (
        0.0,
        rear_axle_target,
        front_axle_target,
        vehicle_length,
    )

    wheel_ground = min(bounds[0].z for bounds in tyre_bounds.values())
    source_top = body_max.z
    body_vertical_scale = vehicle_height / (source_top - wheel_ground)
    body_width_scale = vehicle_width / source_body_width

    target_wheel_centers = {
        "front_left_wheel": Vector(
            (front_axle_target - rear_axle_target, front_track_target / 2.0, wheel_radius_target)
        ),
        "front_right_wheel": Vector(
            (front_axle_target - rear_axle_target, -front_track_target / 2.0, wheel_radius_target)
        ),
        "rear_left_wheel": Vector(
            (0.0, rear_track_target / 2.0, wheel_radius_target)
        ),
        "rear_right_wheel": Vector(
            (0.0, -rear_track_target / 2.0, wheel_radius_target)
        ),
    }

    triangles_by_mesh: dict[
        tuple[str, str], list[tuple[Vector, Vector, Vector]]
    ] = defaultdict(list)

    for obj in mesh_objects:
        source_vertices = _world_vertices(obj)
        wheel_key = _wheel_key(obj.name)
        if wheel_key:
            source_center = wheel_centers[wheel_key]
            minimum, maximum = tyre_bounds[wheel_key]
            source_radius = max(
                maximum.y - minimum.y,
                maximum.z - minimum.z,
            ) / 2.0
            source_width = maximum.x - minimum.x
            radial_scale = wheel_radius_target / source_radius
            width_scale = wheel_width_target / source_width

            def transform(point: Vector) -> Vector:
                delta = point - source_center
                return Vector(
                    (
                        -delta.y * radial_scale,
                        delta.x * width_scale,
                        delta.z * radial_scale,
                    )
                )

            scope = wheel_key
        else:

            def transform(point: Vector) -> Vector:
                source_forward = -point.y
                target_from_rear = _piecewise_map(
                    source_forward,
                    source_knots,
                    target_knots,
                )
                return Vector(
                    (
                        target_from_rear - rear_axle_target,
                        point.x * body_width_scale,
                        (point.z - wheel_ground) * body_vertical_scale,
                    )
                )

            scope = "body"

        obj.data.calc_loop_triangles()
        for triangle in obj.data.loop_triangles:
            polygon = obj.data.polygons[triangle.polygon_index]
            material_index = polygon.material_index
            material_name = ""
            if material_index < len(obj.material_slots):
                material = obj.material_slots[material_index].material
                if material:
                    material_name = material.name
            category = _material_category(obj, material_name, wheel_key)
            triangles_by_mesh[(scope, category)].append(
                tuple(
                    transform(source_vertices[index])
                    for index in triangle.vertices
                )
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    body_entries: list[dict[str, object]] = []
    wheel_entries: dict[str, list[dict[str, object]]] = {
        key: [] for key in WHEEL_KEYS
    }
    for (scope, category), triangles in sorted(triangles_by_mesh.items()):
        filename = f"{scope}_{category}.obj"
        destination = OUTPUT_DIR / filename
        _write_obj(destination, triangles)
        entry = {
            "file": destination.relative_to(SIM_ROOT).as_posix(),
            "material": category,
            "triangle_count": len(triangles),
        }
        if scope == "body":
            body_entries.append(entry)
        else:
            wheel_entries[scope].append(entry)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "measured_piecewise_reshape",
        "source": SOURCE_BLEND.relative_to(SIM_ROOT).as_posix(),
        "coordinate_frame": {
            "origin": "rear_axle_center",
            "x": "forward",
            "y": "left",
            "z": "up",
            "unit": "meter",
        },
        "target_geometry_m": {
            "body_length": vehicle_length,
            "body_width": vehicle_width,
            "body_height": vehicle_height,
            "rear_axle_from_rear": rear_axle_target,
            "front_axle_from_rear": front_axle_target,
            "wheelbase": front_axle_target - rear_axle_target,
            "front_track": front_track_target,
            "rear_track": rear_track_target,
            "wheel_radius": wheel_radius_target,
            "wheel_width": wheel_width_target,
        },
        "source_landmarks": {
            "rear_end": source_rear_end,
            "rear_axle": source_rear_axle,
            "front_axle": source_front_axle,
            "front_end": source_front_end,
            "body_envelope_width": source_body_width,
            "wheel_ground_z": wheel_ground,
            "body_top_z": source_top,
        },
        "reshape": {
            "longitudinal": "piecewise linear through rear end, rear axle, front axle, front end",
            "body_lateral_scale": body_width_scale,
            "body_vertical_scale": body_vertical_scale,
            "wheel_policy": "each source wheel assembly independently centered and scaled to measured radius and width",
            "modifier_policy": "apply every source modifier before export; decimate only after conversion with per-object limits",
        },
        "materials": {
            name: {"rgba": list(rgba)}
            for name, rgba in MATERIAL_RGBA.items()
        },
        "body_meshes": body_entries,
        "wheel_meshes": wheel_entries,
        "triangle_count": sum(len(value) for value in triangles_by_mesh.values()),
    }
    write_json_in_sim(MANIFEST_PATH, manifest)
    return manifest


if __name__ == "__main__":
    result = prepare_vehicle_mesh()
    print(json.dumps(result, indent=2))
