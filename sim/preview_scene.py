from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from .config import get_value, load_config, sim_path
    from .model_builder import build_models
    from .build_map import build_map_assets
    from .vehicle_kinematics import axis_angle_quaternion
except ImportError:
    from config import get_value, load_config, sim_path
    from model_builder import build_models
    from build_map import build_map_assets
    from vehicle_kinematics import axis_angle_quaternion


OVERVIEW_PATH = sim_path("assets/generated/scene_spawn_overview.png")
CLOSEUP_PATH = sim_path("assets/generated/scene_spawn_closeup.png")
VEHICLE_PATH = sim_path("assets/generated/vehicle_mesh_preview.png")
VEHICLE_SIDE_PATH = sim_path(
    "assets/generated/vehicle_mesh_side_preview.png"
)
ONBOARD_CAMERA_PATH = sim_path(
    "assets/generated/onboard_camera_preview.png"
)


def _mocap_id(model: Any, name: str) -> int:
    mocap_id = int(model.body(name).mocapid[0])
    if mocap_id < 0:
        raise RuntimeError(f"Body is not a mocap body: {name}")
    return mocap_id


def position_vehicle(model: Any, data: Any, config: dict[str, Any]) -> None:
    spawn_x = float(get_value(config, "map.spawn.rear_axle_x_m"))
    spawn_y = float(get_value(config, "map.spawn.rear_axle_y_m"))
    yaw = math.radians(float(get_value(config, "map.spawn.yaw_deg")))
    wheelbase = float(get_value(config, "vehicle.wheelbase_m"))
    front_track = float(get_value(config, "vehicle.front_track_m"))
    rear_track = float(get_value(config, "vehicle.rear_track_m"))
    wheel_radius = float(get_value(config, "vehicle.wheel_radius_m"))
    yaw_quaternion = axis_angle_quaternion((0.0, 0.0, 1.0), yaw)

    chassis_id = _mocap_id(model, "vehicle_body")
    data.mocap_pos[chassis_id] = (spawn_x, spawn_y, 0.0)
    data.mocap_quat[chassis_id] = yaw_quaternion

    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    wheel_specs = (
        ("front_left_wheel", wheelbase, front_track / 2.0),
        ("front_right_wheel", wheelbase, -front_track / 2.0),
        ("rear_left_wheel", 0.0, rear_track / 2.0),
        ("rear_right_wheel", 0.0, -rear_track / 2.0),
    )
    for name, forward, left in wheel_specs:
        mocap_id = _mocap_id(model, name)
        data.mocap_pos[mocap_id] = (
            spawn_x + cosine * forward - sine * left,
            spawn_y + sine * forward + cosine * left,
            wheel_radius,
        )
        data.mocap_quat[mocap_id] = yaw_quaternion


def _render(
    model: Any,
    data: Any,
    destination: Path,
    *,
    lookat: tuple[float, float, float],
    distance: float,
    elevation: float,
    azimuth: float = 90.0,
) -> Path:
    import mujoco

    renderer = mujoco.Renderer(model, height=443, width=640)
    try:
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = lookat
        camera.distance = distance
        camera.azimuth = azimuth
        camera.elevation = elevation
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
    finally:
        renderer.close()
    destination = sim_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(destination)
    return destination


def _render_onboard(
    model: Any,
    data: Any,
    destination: Path,
    config: dict[str, Any],
) -> Path:
    import mujoco

    width = int(get_value(config, "hardware.camera.width_px"))
    height = int(get_value(config, "hardware.camera.height_px"))
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera="onboard")
        pixels = renderer.render()
    finally:
        renderer.close()
    destination = sim_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(destination)
    return destination


def build_preview(
    config_path: Path,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    import mujoco

    config = load_config(config_path)
    build_map_assets(config, preview_only=False)
    outputs = build_models(config)
    model = mujoco.MjModel.from_xml_path(str(outputs["mjcf"]))
    data = mujoco.MjData(model)
    position_vehicle(model, data, config)
    mujoco.mj_forward(model, data)

    spawn_x = float(get_value(config, "map.spawn.rear_axle_x_m"))
    spawn_y = float(get_value(config, "map.spawn.rear_axle_y_m"))
    map_extent = max(
        float(get_value(config, "map.physical_width_m")),
        float(get_value(config, "map.physical_height_m")),
    )
    overview = _render(
        model,
        data,
        OVERVIEW_PATH,
        lookat=(0.0, 0.0, 0.0),
        distance=map_extent * 0.92,
        elevation=-90.0,
    )
    closeup = _render(
        model,
        data,
        CLOSEUP_PATH,
        lookat=(spawn_x, spawn_y - 0.3, 0.0),
        distance=2.5,
        elevation=-72.0,
    )
    vehicle = _render(
        model,
        data,
        VEHICLE_PATH,
        lookat=(spawn_x, spawn_y - 0.25, 0.18),
        distance=1.25,
        elevation=-18.0,
    )
    vehicle_side = _render(
        model,
        data,
        VEHICLE_SIDE_PATH,
        lookat=(spawn_x, spawn_y - 0.25, 0.16),
        distance=1.55,
        elevation=-10.0,
        azimuth=0.0,
    )
    onboard = _render_onboard(
        model,
        data,
        ONBOARD_CAMERA_PATH,
        config,
    )
    return (
        outputs["mjcf"],
        overview,
        closeup,
        vehicle,
        vehicle_side,
        onboard,
    )


def launch_interactive(world_path: Path, config_path: Path) -> None:
    import mujoco
    import mujoco.viewer

    config = load_config(config_path)
    model = mujoco.MjModel.from_xml_path(str(world_path))
    data = mujoco.MjData(model)
    position_vehicle(model, data, config)
    mujoco.mj_forward(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (0.0, 0.0, 0.0)
        viewer.cam.distance = max(
            float(get_value(config, "map.physical_width_m")),
            float(get_value(config, "map.physical_height_m")),
        ) * 0.92
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -70.0
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and preview the static map and configured start pose."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=sim_path("config/sim_config.json"),
    )
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    world, overview, closeup, vehicle, vehicle_side, onboard = build_preview(
        args.config
    )
    print(f"Built: {world}")
    print(f"Overview: {overview}")
    print(f"Close-up: {closeup}")
    print(f"Vehicle: {vehicle}")
    print(f"Vehicle side: {vehicle_side}")
    print(f"Onboard camera: {onboard}")
    if args.interactive:
        launch_interactive(world, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
