from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .config import (
        DEFAULT_CONFIG_PATH,
        ConfigurationError,
        get_value,
        is_known,
        load_config,
        require_parameters,
        set_value,
        sim_path,
        write_json_in_sim,
    )
except ImportError:  # Direct execution: python sim/build_map.py
    from config import (
        DEFAULT_CONFIG_PATH,
        ConfigurationError,
        get_value,
        is_known,
        load_config,
        require_parameters,
        set_value,
        sim_path,
        write_json_in_sim,
    )


GENERATED_DIR = sim_path("assets/generated")


def _largest_component_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        raise ConfigurationError("Could not find a colored map region in the source image.")
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, width, height, _ = (int(value) for value in stats[component])
    return x, y, x + width, y + height


def detect_rgb_artwork_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    """Find the saturated rectangular artwork while excluding the white screenshot."""

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    colored = ((hsv[:, :, 1] > 5) & (hsv[:, :, 2] < 254)).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    connected = cv2.morphologyEx(colored, cv2.MORPH_CLOSE, kernel, iterations=2)
    return _largest_component_bbox(connected)


def detect_plan_frame_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    """Detect the pale cyan drawing frame used as the plan's scale boundary."""

    blue, green, red = cv2.split(image.astype(np.int16))
    cyan = (
        (blue - red > 2)
        & (green - red > 2)
        & (blue > 180)
        & (green > 180)
    )
    ys, xs = np.where(cyan)
    if len(xs) < 50:
        raise ConfigurationError("Could not detect the cyan plan boundary.")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def target_texture_size(
    physical_width_m: float,
    physical_height_m: float,
    long_edge_px: int,
) -> tuple[int, int]:
    if physical_width_m <= 0 or physical_height_m <= 0:
        raise ConfigurationError("Map dimensions must be positive.")
    if long_edge_px < 256:
        raise ConfigurationError("map.texture_long_edge_px must be at least 256.")
    if physical_width_m >= physical_height_m:
        width = long_edge_px
        height = max(1, int(round(long_edge_px * physical_height_m / physical_width_m)))
    else:
        height = long_edge_px
        width = max(1, int(round(long_edge_px * physical_width_m / physical_height_m)))
    return width, height


def _read_image(path: Path, label: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ConfigurationError(f"Could not read {label}: {path}")
    return image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plan_line_mask(plan: np.ndarray) -> np.ndarray:
    blue, green, red = cv2.split(plan.astype(np.int16))
    chromatic = (
        ((red - blue > 2) & (red - green > 1))
        | ((blue - red > 2) & (green - red > 2))
        | ((green - red > 2) & (green - blue > 1))
    )
    # Dimension text and leaders are gray/black.  Excluding them keeps this a
    # geometric registration preview instead of covering the RGB artwork with
    # drafting annotations.
    return chromatic.astype(np.uint8) * 255


def _alignment_preview(texture: np.ndarray, rotated_plan: np.ndarray) -> np.ndarray:
    height, width = texture.shape[:2]
    plan_scaled = cv2.resize(rotated_plan, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(
        _plan_line_mask(rotated_plan),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    preview = texture.copy()
    overlay_color = np.zeros_like(preview)
    overlay_color[:, :] = (255, 0, 255)
    alpha = (mask.astype(np.float32) / 255.0 * 0.55)[:, :, None]
    return np.clip(
        preview.astype(np.float32) * (1.0 - alpha)
        + overlay_color.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)


def build_map_assets(
    config: dict[str, Any],
    *,
    preview_only: bool = False,
) -> dict[str, Path]:
    rgb_path = sim_path(str(get_value(config, "map.rgb_image")), must_exist=True)
    plan_path = sim_path(str(get_value(config, "map.plan_image")), must_exist=True)
    long_edge = int(get_value(config, "map.texture_long_edge_px"))

    rgb_source = _read_image(rgb_path, "RGB map image")
    plan_source = _read_image(plan_path, "track plan image")

    rgb_bbox = detect_rgb_artwork_bbox(rgb_source)
    plan_bbox = detect_plan_frame_bbox(plan_source)
    rx1, ry1, rx2, ry2 = rgb_bbox
    px1, py1, px2, py2 = plan_bbox
    rgb_crop = rgb_source[ry1:ry2, rx1:rx2]
    plan_crop = plan_source[py1:py2, px1:px2]
    quarters = int(get_value(config, "map.plan_rotation_clockwise_quarters") or 0) % 4
    rotated_plan = np.rot90(plan_crop, k=-quarters).copy()

    width_value = get_value(config, "map.physical_width_m")
    height_value = get_value(config, "map.physical_height_m")
    if preview_only and (not is_known(width_value) or not is_known(height_value)):
        # The preview has no meter scale yet, but it must already obey the
        # rotated drawing frame's aspect ratio rather than preserving the
        # distorted screenshot aspect.
        plan_height, plan_width = rotated_plan.shape[:2]
        if plan_width >= plan_height:
            target_width = long_edge
            target_height = int(round(long_edge * plan_height / plan_width))
        else:
            target_height = long_edge
            target_width = int(round(long_edge * plan_width / plan_height))
        texture_name = "track_texture_unscaled_preview.png"
        physical_size = None
    else:
        require_parameters(
            config,
            ("map.physical_width_m", "map.physical_height_m"),
            context="Final map generation",
        )
        physical_width = float(width_value)
        physical_height = float(height_value)
        target_width, target_height = target_texture_size(
            physical_width,
            physical_height,
            long_edge,
        )
        texture_name = "track_texture.png"
        physical_size = [physical_width, physical_height]

    texture = cv2.resize(
        rgb_crop,
        (target_width, target_height),
        interpolation=cv2.INTER_LANCZOS4,
    )
    preview = _alignment_preview(texture, rotated_plan)

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    texture_path = sim_path(f"assets/generated/{texture_name}")
    alignment_path = sim_path("assets/generated/track_alignment_preview.png")
    metadata_path = sim_path("assets/generated/track_metadata.json")
    if not cv2.imwrite(str(texture_path), texture):
        raise ConfigurationError(f"Failed to write map texture: {texture_path}")
    if not cv2.imwrite(str(alignment_path), preview):
        raise ConfigurationError(f"Failed to write alignment preview: {alignment_path}")

    metadata = {
        "schema_version": 1,
        "status": "preview_only" if physical_size is None else "final",
        "registration_method": (
            "four_corner_normalization_of_rgb_artwork_and_rotated_plan_frame"
        ),
        "rgb_source": rgb_path.relative_to(sim_path(".")).as_posix(),
        "rgb_source_sha256": _sha256(rgb_path),
        "rgb_crop_xyxy_px": list(rgb_bbox),
        "rgb_crop_size_px": [rgb_crop.shape[1], rgb_crop.shape[0]],
        "plan_source": plan_path.relative_to(sim_path(".")).as_posix(),
        "plan_source_sha256": _sha256(plan_path),
        "plan_frame_xyxy_px": list(plan_bbox),
        "plan_rotation_clockwise_quarters": quarters,
        "rotated_plan_frame_size_px": [
            rotated_plan.shape[1],
            rotated_plan.shape[0],
        ],
        "physical_size_m": physical_size,
        "texture_size_px": [target_width, target_height],
        "rgb_scale_xy": [
            target_width / rgb_crop.shape[1],
            target_height / rgb_crop.shape[0],
        ],
        "plan_scale_xy": [
            target_width / rotated_plan.shape[1],
            target_height / rotated_plan.shape[0],
        ],
        "interpolation": "OpenCV INTER_LANCZOS4",
        "texture": texture_path.relative_to(sim_path(".")).as_posix(),
        "alignment_preview": alignment_path.relative_to(sim_path(".")).as_posix(),
        "alignment_preview_note": (
            "Magenta pixels are plan linework resized over the RGB texture. "
            "They are QA evidence and are not baked into the floor texture."
        ),
    }
    write_json_in_sim(metadata_path, metadata)
    return {
        "texture": texture_path,
        "alignment_preview": alignment_path,
        "metadata": metadata_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crop and Lanczos-resample the RGB floor map to the plan dimensions."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--width-m", type=float)
    parser.add_argument("--height-m", type=float)
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Persist explicitly supplied --width-m/--height-m values inside sim/config.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if (args.width_m is None) != (args.height_m is None):
        raise ConfigurationError("--width-m and --height-m must be supplied together.")
    if args.width_m is not None:
        set_value(config, "map.physical_width_m", args.width_m)
        set_value(config, "map.physical_height_m", args.height_m)
        if args.write_config:
            persisted = {
                key: value for key, value in config.items() if not key.startswith("_")
            }
            write_json_in_sim(args.config, persisted)
    outputs = build_map_assets(config, preview_only=args.preview_only)
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
