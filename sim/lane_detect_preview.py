r"""Render onboard-camera frames in the simulator and run the lane
segmentation model on them.

Usage:
    python -B .\sim\lane_detect_preview.py --model <openvino model dir>

For each requested pose the vehicle mocap bodies are moved along the
spawn heading, the fixed "onboard" camera is rendered at the configured
resolution, and the YOLO segmentation model is applied. Raw frames and
annotated overlays are written to sim/assets/generated/lane_detect/.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

_YOLO_CONFIG_DIR = Path(__file__).resolve().parent / ".cache" / "ultralytics"
_YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_YOLO_CONFIG_DIR))

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


OUTPUT_DIR = sim_path("assets/generated/lane_detect")


def _mocap_id(model: Any, name: str) -> int:
    mocap_id = int(model.body(name).mocapid[0])
    if mocap_id < 0:
        raise RuntimeError(f"Body is not a mocap body: {name}")
    return mocap_id


def place_vehicle(
    model: Any,
    data: Any,
    config: dict[str, Any],
    rear_axle_x: float,
    rear_axle_y: float,
    yaw_rad: float,
) -> None:
    wheelbase = float(get_value(config, "vehicle.wheelbase_m"))
    front_track = float(get_value(config, "vehicle.front_track_m"))
    rear_track = float(get_value(config, "vehicle.rear_track_m"))
    wheel_radius = float(get_value(config, "vehicle.wheel_radius_m"))
    yaw_quaternion = axis_angle_quaternion((0.0, 0.0, 1.0), yaw_rad)

    chassis_id = _mocap_id(model, "vehicle_body")
    data.mocap_pos[chassis_id] = (rear_axle_x, rear_axle_y, 0.0)
    data.mocap_quat[chassis_id] = yaw_quaternion

    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    wheel_specs = (
        ("front_left_wheel", wheelbase, front_track / 2.0),
        ("front_right_wheel", wheelbase, -front_track / 2.0),
        ("rear_left_wheel", 0.0, rear_track / 2.0),
        ("rear_right_wheel", 0.0, -rear_track / 2.0),
    )
    for name, forward, left in wheel_specs:
        mocap_id = _mocap_id(model, name)
        data.mocap_pos[mocap_id] = (
            rear_axle_x + cosine * forward - sine * left,
            rear_axle_y + sine * forward + cosine * left,
            wheel_radius,
        )
        data.mocap_quat[mocap_id] = yaw_quaternion


def render_onboard(model: Any, data: Any, config: dict[str, Any]) -> Any:
    import mujoco

    width = int(get_value(config, "hardware.camera.width_px"))
    height = int(get_value(config, "hardware.camera.height_px"))
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera="onboard")
        pixels = renderer.render()
    finally:
        renderer.close()
    return pixels


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render onboard camera frames along the spawn heading and run "
            "the lane segmentation model on them."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=sim_path("config/sim_config.json"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the trained segmentation model (OpenVINO dir or .pt).",
    )
    parser.add_argument(
        "--advances-m",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0, 3.0],
        help="Forward offsets from spawn along the spawn heading, in meters.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for the detector.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model path does not exist: {model_path}")

    import cv2
    import mujoco
    from ultralytics import YOLO

    config = load_config(args.config)
    build_map_assets(config, preview_only=False)
    outputs = build_models(config)
    mj_model = mujoco.MjModel.from_xml_path(str(outputs["mjcf"]))
    mj_data = mujoco.MjData(mj_model)

    spawn_x = float(get_value(config, "map.spawn.rear_axle_x_m"))
    spawn_y = float(get_value(config, "map.spawn.rear_axle_y_m"))
    yaw = math.radians(float(get_value(config, "map.spawn.yaw_deg")))
    heading = (math.cos(yaw), math.sin(yaw))

    detector = YOLO(str(model_path), task="segment")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for advance in args.advances_m:
        rear_x = spawn_x + heading[0] * advance
        rear_y = spawn_y + heading[1] * advance
        place_vehicle(mj_model, mj_data, config, rear_x, rear_y, yaw)
        mujoco.mj_forward(mj_model, mj_data)
        frame_rgb = render_onboard(mj_model, mj_data, config)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        results = detector.predict(frame_bgr, conf=args.conf, verbose=False)
        result = results[0]
        annotated_bgr = result.plot()

        tag = f"{advance:.1f}".replace(".", "p").replace("-", "m")
        raw_path = OUTPUT_DIR / f"frame_{tag}m_raw.png"
        overlay_path = OUTPUT_DIR / f"frame_{tag}m_overlay.png"
        cv2.imwrite(str(raw_path), frame_bgr)
        cv2.imwrite(str(overlay_path), annotated_bgr)

        names = result.names
        if result.boxes is None or len(result.boxes) == 0:
            summary = "no detections"
        else:
            parts = []
            for box in result.boxes:
                cls_name = names[int(box.cls[0])]
                parts.append(f"{cls_name} {float(box.conf[0]):.2f}")
            summary = ", ".join(parts)
        print(f"advance {advance:+.1f} m -> {summary}")
        print(f"  raw:     {raw_path}")
        print(f"  overlay: {overlay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
