from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Callable

try:
    from .config import get_value, load_config, sim_path, write_json_in_sim
except ImportError:
    from config import get_value, load_config, sim_path, write_json_in_sim


SOURCE_DIR = sim_path("prius_hybrid/prius_hybrid")
SOURCE_MESH_DIR = SOURCE_DIR / "meshes"
OUTPUT_DIR = sim_path("assets/generated/vehicle_mesh_prius")
MANIFEST_PATH = sim_path("config/vehicle_visual_mesh.json")


def _parse_obj(path: Path) -> tuple[
    list[tuple[float, float, float]],
    list[str],
    dict[str, list[str]],
]:
    vertices: list[tuple[float, float, float]] = []
    texture_coordinates: list[str] = []
    faces: dict[str, list[str]] = defaultdict(list)
    material = ""
    for raw_line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        line = raw_line.strip()
        if line.startswith("v "):
            x, y, z = (float(value) for value in line.split()[1:4])
            vertices.append((x, y, z))
        elif line.startswith("vt "):
            texture_coordinates.append(line)
        elif line.startswith("usemtl "):
            material = line.split(maxsplit=1)[1]
        elif line.startswith("f "):
            if not material:
                raise ValueError(f"Face appears before usemtl in {path}")
            faces[material].append(line)
    if not vertices or not faces:
        raise ValueError(f"OBJ has no usable geometry: {path}")
    return vertices, texture_coordinates, dict(faces)


def _face_without_normals(line: str, *, include_texcoords: bool) -> str:
    rewritten: list[str] = []
    for token in line.split()[1:]:
        parts = token.split("/")
        vertex = parts[0]
        texture = parts[1] if len(parts) > 1 else ""
        rewritten.append(
            f"{vertex}/{texture}" if include_texcoords and texture else vertex
        )
    return "f " + " ".join(rewritten)


def _write_obj(
    destination: Path,
    vertices: list[tuple[float, float, float]],
    texture_coordinates: list[str],
    faces: list[str],
    transform: Callable[[float, float, float], tuple[float, float, float]],
) -> int:
    lines = ["# Generated inside sim from the supplied Prius Hybrid mesh."]
    for source_x, source_y, source_z in vertices:
        x, y, z = transform(source_x, source_y, source_z)
        lines.append(f"v {x:.10g} {y:.10g} {z:.10g}")
    lines.extend(texture_coordinates)
    include_texcoords = bool(texture_coordinates)
    lines.extend(
        _face_without_normals(face, include_texcoords=include_texcoords)
        for face in faces
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sum(max(len(face.split()) - 3, 1) for face in faces)


def _piecewise(
    value: float,
    source: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
) -> float:
    if value <= source[1]:
        low = 0
    elif value <= source[2]:
        low = 1
    else:
        low = 2
    fraction = (value - source[low]) / (source[low + 1] - source[low])
    return target[low] + fraction * (target[low + 1] - target[low])


def prepare(config_path: Path) -> Path:
    config = load_config(config_path)
    length = float(get_value(config, "vehicle.body_length_m"))
    width = float(get_value(config, "vehicle.body_width_m"))
    height = float(get_value(config, "vehicle.body_height_m"))
    rear_axle_from_rear = float(
        get_value(config, "vehicle.rear_axle_from_rear_m")
    )
    front_axle_from_rear = float(
        get_value(config, "vehicle.front_axle_from_rear_m")
    )
    wheelbase = float(get_value(config, "vehicle.wheelbase_m"))
    front_track = float(get_value(config, "vehicle.front_track_m"))
    rear_track = float(get_value(config, "vehicle.rear_track_m"))
    wheel_radius = float(get_value(config, "vehicle.wheel_radius_m"))
    wheel_width = float(get_value(config, "vehicle.wheel_width_m"))

    body_vertices, body_texcoords, body_faces = _parse_obj(
        SOURCE_MESH_DIR / "hybrid_body.obj"
    )
    wheel_vertices, wheel_texcoords, wheel_faces = _parse_obj(
        SOURCE_MESH_DIR / "wheel.obj"
    )

    source_rear_end = -max(vertex[1] for vertex in body_vertices) / 100.0
    source_rear_axle = -1.45
    source_front_axle = 1.41
    source_front_end = -min(vertex[1] for vertex in body_vertices) / 100.0
    source_longitudinal = (
        source_rear_end,
        source_rear_axle,
        source_front_axle,
        source_front_end,
    )
    target_longitudinal = (
        -rear_axle_from_rear,
        0.0,
        wheelbase,
        length - rear_axle_from_rear,
    )
    source_lateral_span_m = (
        max(vertex[0] for vertex in body_vertices)
        - min(vertex[0] for vertex in body_vertices)
    ) / 100.0
    lateral_scale = width / source_lateral_span_m
    source_wheel_radius_m = max(
        max(abs(vertex[1]), abs(vertex[2])) for vertex in wheel_vertices
    ) / 100.0
    source_wheel_width_m = (
        max(vertex[0] for vertex in wheel_vertices)
        - min(vertex[0] for vertex in wheel_vertices)
    ) / 100.0
    source_ground_m = 0.30 - source_wheel_radius_m
    source_body_top_m = max(vertex[2] for vertex in body_vertices) / 100.0
    vertical_scale = height / (source_body_top_m - source_ground_m)

    def transform_body(
        source_x: float,
        source_y: float,
        source_z: float,
    ) -> tuple[float, float, float]:
        return (
            _piecewise(
                -source_y / 100.0,
                source_longitudinal,
                target_longitudinal,
            ),
            source_x / 100.0 * lateral_scale,
            (source_z / 100.0 - source_ground_m) * vertical_scale,
        )

    wheel_radius_scale = wheel_radius / source_wheel_radius_m
    wheel_width_scale = wheel_width / source_wheel_width_m

    def transform_wheel(
        source_x: float,
        source_y: float,
        source_z: float,
    ) -> tuple[float, float, float]:
        return (
            -source_y / 100.0 * wheel_radius_scale,
            source_x / 100.0 * wheel_width_scale,
            source_z / 100.0 * wheel_radius_scale,
        )

    material_map = {
        "Hybrid": ("body", "body"),
        "Hybrid_Interior": ("interior", "interior"),
        "Windows": ("glass", "glass"),
    }
    body_entries: list[dict[str, object]] = []
    triangle_count = 0
    for source_material, (filename, material) in material_map.items():
        destination = OUTPUT_DIR / f"{filename}.obj"
        triangles = _write_obj(
            destination,
            body_vertices,
            body_texcoords,
            body_faces[source_material],
            transform_body,
        )
        triangle_count += triangles
        body_entries.append(
            {
                "file": destination.relative_to(sim_path(".")).as_posix(),
                "material": material,
                "triangle_count": triangles,
            }
        )

    wheel_destination = OUTPUT_DIR / "wheel.obj"
    wheel_triangles = _write_obj(
        wheel_destination,
        wheel_vertices,
        wheel_texcoords,
        wheel_faces["Wheels3"],
        transform_wheel,
    )
    triangle_count += wheel_triangles * 4
    wheel_entry = {
        "file": wheel_destination.relative_to(sim_path(".")).as_posix(),
        "material": "wheel",
        "triangle_count": wheel_triangles,
    }

    texture_root = "prius_hybrid/prius_hybrid/materials/textures"
    manifest = {
        "schema_version": 1,
        "status": "prius_piecewise_reshape",
        "source": "prius_hybrid/prius_hybrid/meshes/hybrid_body.obj",
        "coordinate_frame": {
            "origin": "rear_axle_center",
            "x": "forward",
            "y": "left",
            "z": "up",
            "unit": "meter",
        },
        "target_geometry_m": {
            "body_length": length,
            "body_width": width,
            "body_height": height,
            "rear_axle_from_rear": rear_axle_from_rear,
            "front_axle_from_rear": front_axle_from_rear,
            "wheelbase": wheelbase,
            "front_track": front_track,
            "rear_track": rear_track,
            "wheel_radius": wheel_radius,
            "wheel_width": wheel_width,
        },
        "source_landmarks": {
            "rear_end": source_rear_end,
            "rear_axle": source_rear_axle,
            "front_axle": source_front_axle,
            "front_end": source_front_end,
            "wheel_center_z": 0.30,
        },
        "reshape": {
            "longitudinal": (
                "piecewise linear through rear end, rear axle, "
                "front axle, front end"
            ),
            "body_lateral_scale": lateral_scale,
            "body_vertical_scale": vertical_scale,
            "wheel_radius_scale": wheel_radius_scale,
            "wheel_width_scale": wheel_width_scale,
        },
        "materials": {
            "body": {
                "rgba": [1.0, 1.0, 1.0, 1.0],
                "texture": f"{texture_root}/Hybrid.png",
            },
            "interior": {
                "rgba": [1.0, 1.0, 1.0, 1.0],
                "texture": f"{texture_root}/Hybrid_Interior.png",
            },
            "glass": {
                "rgba": [0.08, 0.12, 0.16, 0.42],
            },
            "wheel": {
                "rgba": [1.0, 1.0, 1.0, 1.0],
                "texture": f"{texture_root}/Wheels3.png",
            },
        },
        "body_meshes": body_entries,
        "wheel_meshes": {
            name: [dict(wheel_entry)]
            for name in (
                "front_left_wheel",
                "front_right_wheel",
                "rear_left_wheel",
                "rear_right_wheel",
            )
        },
        "triangle_count": triangle_count,
    }
    write_json_in_sim(MANIFEST_PATH, manifest)
    return MANIFEST_PATH


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reshape the supplied Prius mesh to the measured child-car geometry."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=sim_path("config/sim_config.json"),
    )
    args = parser.parse_args()
    manifest = prepare(args.config)
    print(f"Prepared Prius visual mesh: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
