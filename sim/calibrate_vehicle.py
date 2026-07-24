from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

try:
    from .config import write_json_in_sim
except ImportError:  # Direct execution: python -B sim/calibrate_vehicle.py
    from config import write_json_in_sim


SIM_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE_PATH = SIM_ROOT / "assets" / "source" / "vehicle_side_profile.png"
DEFAULT_OUTPUT_PATH = SIM_ROOT / "config" / "vehicle_geometry.json"
VEHICLE_LENGTH_M = 0.99
REQUIRED_CLICK_COUNT = 6

Point = tuple[float, float]


class CalibrationError(ValueError):
    """Raised when the visible-side annotations are invalid."""


class IncompleteCalibrationError(CalibrationError):
    """Raised when save is requested before all six points are present."""


@dataclass(frozen=True)
class VehicleMeasurements:
    """Points that are actually visible in the supplied oblique photograph."""

    body_rear: Point
    body_front: Point
    rear_left_wheel_center: Point
    rear_left_wheel_edge: Point
    front_left_wheel_center: Point
    front_left_wheel_edge: Point

    def points(self) -> tuple[Point, Point, Point, Point, Point, Point]:
        return (
            self.body_rear,
            self.body_front,
            self.rear_left_wheel_center,
            self.rear_left_wheel_edge,
            self.front_left_wheel_center,
            self.front_left_wheel_edge,
        )


@dataclass(frozen=True)
class VehicleGeometry:
    rear_axle_from_rear_m: float
    front_axle_from_rear_m: float
    wheelbase_m: float
    rear_wheel_radius_m: float
    front_wheel_radius_m: float
    wheel_radius_m: float
    scale_m_per_px: float
    body_length_px: float
    rear_wheel_projection_px: Point
    front_wheel_projection_px: Point

    def config_values(self) -> dict[str, float | None]:
        # A single visible side provides longitudinal axle positions only.
        # Track remains null because a side view cannot reveal lateral spacing.
        return {
            "rear_axle_from_rear_m": self.rear_axle_from_rear_m,
            "front_axle_from_rear_m": self.front_axle_from_rear_m,
            "wheelbase_m": self.wheelbase_m,
            "front_track_m": None,
            "rear_track_m": None,
            "wheel_radius_m": self.wheel_radius_m,
        }


@dataclass
class ClickCalibrationState:
    """Ordered click history so undo/reset cannot leave stale partial fields."""

    clicks: list[Point] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return len(self.clicks) == REQUIRED_CLICK_COUNT

    def add_click(self, point: Sequence[float]) -> bool:
        if self.complete:
            return False
        self.clicks.append(_validated_point(point, "click"))
        return True

    def undo(self) -> bool:
        if not self.clicks:
            return False
        self.clicks.pop()
        return True

    def reset(self) -> None:
        self.clicks.clear()

    def to_measurements(self) -> VehicleMeasurements:
        if not self.complete:
            raise IncompleteCalibrationError(
                f"Calibration is incomplete: {len(self.clicks)}/"
                f"{REQUIRED_CLICK_COUNT} points collected."
            )
        return VehicleMeasurements(
            body_rear=self.clicks[0],
            body_front=self.clicks[1],
            rear_left_wheel_center=self.clicks[2],
            rear_left_wheel_edge=self.clicks[3],
            front_left_wheel_center=self.clicks[4],
            front_left_wheel_edge=self.clicks[5],
        )

    def next_prompt(self) -> str:
        prompts = (
            "1/6 BODY: click REAR end",
            "2/6 BODY: click FRONT end",
            "3/6 REAR WHEEL: click center",
            "4/6 REAR WHEEL: click outer tire edge",
            "5/6 FRONT WHEEL: click center",
            "6/6 FRONT WHEEL: click outer tire edge",
        )
        if self.complete:
            return "COMPLETE: press S to validate and save"
        return prompts[len(self.clicks)]


def _validated_point(point: Sequence[float], name: str) -> Point:
    if len(point) != 2:
        raise CalibrationError(f"{name} must contain exactly two coordinates.")
    x = float(point[0])
    y = float(point[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        raise CalibrationError(f"{name} coordinates must be finite.")
    return x, y


def _distance(first: Point, second: Point) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])


def project_distance_from_rear_px(
    point: Point,
    body_rear: Point,
    body_front: Point,
) -> float:
    """Project a visible wheel center onto the rear-to-front body axis."""

    point = _validated_point(point, "point")
    body_rear = _validated_point(body_rear, "body_rear")
    body_front = _validated_point(body_front, "body_front")
    axis_x = body_front[0] - body_rear[0]
    axis_y = body_front[1] - body_rear[1]
    axis_length = math.hypot(axis_x, axis_y)
    if axis_length <= 0.0:
        raise CalibrationError("Body length line must have two distinct endpoints.")
    return (
        (point[0] - body_rear[0]) * axis_x
        + (point[1] - body_rear[1]) * axis_y
    ) / axis_length


def projected_point_on_body_axis(
    point: Point,
    body_rear: Point,
    body_front: Point,
) -> Point:
    distance_px = project_distance_from_rear_px(point, body_rear, body_front)
    axis_x = body_front[0] - body_rear[0]
    axis_y = body_front[1] - body_rear[1]
    axis_length = math.hypot(axis_x, axis_y)
    return (
        body_rear[0] + distance_px * axis_x / axis_length,
        body_rear[1] + distance_px * axis_y / axis_length,
    )


def calculate_vehicle_geometry(
    measurements: VehicleMeasurements,
) -> VehicleGeometry:
    """Derive longitudinal axle positions from the annotated visible side.

    The known 0.99 m length supplies the longitudinal image scale. This is an
    affine measurement along the chosen body axis; the saved provenance states
    that a single oblique image cannot resolve transverse geometry.
    """

    body_rear = _validated_point(measurements.body_rear, "body_rear")
    body_front = _validated_point(measurements.body_front, "body_front")
    rear_wheel = _validated_point(
        measurements.rear_left_wheel_center,
        "rear_left_wheel_center",
    )
    rear_wheel_edge = _validated_point(
        measurements.rear_left_wheel_edge,
        "rear_left_wheel_edge",
    )
    front_wheel = _validated_point(
        measurements.front_left_wheel_center,
        "front_left_wheel_center",
    )
    front_wheel_edge = _validated_point(
        measurements.front_left_wheel_edge,
        "front_left_wheel_edge",
    )

    body_length_px = _distance(body_rear, body_front)
    if body_length_px <= 0.0:
        raise CalibrationError("Body length line must have two distinct endpoints.")

    rear_axle_px = project_distance_from_rear_px(
        rear_wheel,
        body_rear,
        body_front,
    )
    front_axle_px = project_distance_from_rear_px(
        front_wheel,
        body_rear,
        body_front,
    )
    scale_m_per_px = VEHICLE_LENGTH_M / body_length_px
    rear_axle_m = rear_axle_px * scale_m_per_px
    front_axle_m = front_axle_px * scale_m_per_px
    rear_wheel_radius_m = _distance(
        rear_wheel,
        rear_wheel_edge,
    ) * scale_m_per_px
    front_wheel_radius_m = _distance(
        front_wheel,
        front_wheel_edge,
    ) * scale_m_per_px

    if not 0.0 < rear_axle_m < VEHICLE_LENGTH_M:
        raise CalibrationError(
            "Rear wheel projection must lie strictly inside the body length."
        )
    if not 0.0 < front_axle_m < VEHICLE_LENGTH_M:
        raise CalibrationError(
            "Front wheel projection must lie strictly inside the body length."
        )
    if front_axle_m <= rear_axle_m:
        raise CalibrationError(
            "Front wheel must project ahead of the rear wheel. "
            "Check the rear-to-front click order."
        )
    if rear_wheel_radius_m <= 0.0:
        raise CalibrationError(
            "Rear wheel center and tire edge must be distinct."
        )
    if front_wheel_radius_m <= 0.0:
        raise CalibrationError(
            "Front wheel center and tire edge must be distinct."
        )

    return VehicleGeometry(
        rear_axle_from_rear_m=rear_axle_m,
        front_axle_from_rear_m=front_axle_m,
        wheelbase_m=front_axle_m - rear_axle_m,
        rear_wheel_radius_m=rear_wheel_radius_m,
        front_wheel_radius_m=front_wheel_radius_m,
        wheel_radius_m=(
            rear_wheel_radius_m + front_wheel_radius_m
        ) / 2.0,
        scale_m_per_px=scale_m_per_px,
        body_length_px=body_length_px,
        rear_wheel_projection_px=projected_point_on_body_axis(
            rear_wheel,
            body_rear,
            body_front,
        ),
        front_wheel_projection_px=projected_point_on_body_axis(
            front_wheel,
            body_rear,
            body_front,
        ),
    )


def validated_output_path(output_path: str | Path) -> Path:
    """Resolve an output path and reject every target outside ``sim``."""

    candidate = Path(output_path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve(strict=False)
    sim_root = SIM_ROOT.resolve(strict=True)
    try:
        relative = candidate.relative_to(sim_root)
    except ValueError as exc:
        raise CalibrationError(
            f"Output must stay inside {sim_root}: {candidate}"
        ) from exc
    if not relative.parts:
        raise CalibrationError(
            "Output path must name a JSON file inside the sim directory."
        )
    if candidate.suffix.lower() != ".json":
        raise CalibrationError("Output path must use the .json extension.")
    if candidate.exists() and candidate.is_dir():
        raise CalibrationError(f"Output path is a directory: {candidate}")
    return candidate


def _source_path_for_payload(source_image: Path) -> str:
    source = source_image.resolve(strict=False)
    try:
        return source.relative_to(SIM_ROOT.resolve(strict=True)).as_posix()
    except ValueError:
        return str(source)


def build_vehicle_geometry_payload(
    measurements: VehicleMeasurements,
    geometry: VehicleGeometry,
    source_image: str | Path,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "visible_left_side_annotated",
        "vehicle_length_m": VEHICLE_LENGTH_M,
        "vehicle_geometry": geometry.config_values(),
        "calibration": {
            "method": "affine_projection_on_visible_left_side_body_axis",
            "source_image": _source_path_for_payload(Path(source_image)),
            "known_length_m": VEHICLE_LENGTH_M,
            "scale_m_per_px": geometry.scale_m_per_px,
            "body_length_px": geometry.body_length_px,
            "points_px": {
                "body_rear": list(measurements.body_rear),
                "body_front": list(measurements.body_front),
                "rear_left_wheel_center": list(
                    measurements.rear_left_wheel_center
                ),
                "rear_left_wheel_edge": list(
                    measurements.rear_left_wheel_edge
                ),
                "front_left_wheel_center": list(
                    measurements.front_left_wheel_center
                ),
                "front_left_wheel_edge": list(
                    measurements.front_left_wheel_edge
                ),
            },
            "wheel_radius_estimates_m": {
                "rear": geometry.rear_wheel_radius_m,
                "front": geometry.front_wheel_radius_m,
                "configured_mean": geometry.wheel_radius_m,
            },
            "projected_wheel_centers_px": {
                "rear": list(geometry.rear_wheel_projection_px),
                "front": list(geometry.front_wheel_projection_px),
            },
            "limitations": [
                (
                    "Longitudinal values use a 1D affine image scale. "
                    "Perspective depth cannot be solved from only two known "
                    "body endpoints in one oblique photograph."
                ),
                (
                    "The profile photo is treated as an orthographic-like side "
                    "view when converting wheel pixel radii with the body scale."
                ),
                (
                    "The hidden right-side wheels are not mirrored or guessed. "
                    "Front/rear track remain null."
                ),
            ],
            "unmeasured": {
                "front_track_m": "Hidden right-side wheel center is unavailable.",
                "rear_track_m": "Hidden right-side wheel center is unavailable.",
            },
            "wheel_material": "polyurethane",
        },
    }


def save_calibration(
    state: ClickCalibrationState,
    source_image: str | Path,
    output_path: str | Path,
) -> tuple[Path, VehicleGeometry]:
    measurements = state.to_measurements()
    geometry = calculate_vehicle_geometry(measurements)
    destination = validated_output_path(output_path)
    payload = build_vehicle_geometry_payload(
        measurements,
        geometry,
        source_image,
    )
    write_json_in_sim(destination, payload)
    return destination, geometry


def _as_int_point(point: Point) -> tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))


def _draw_marker(
    draw,
    point: Point,
    color: tuple[int, int, int],
    label: str,
    font,
) -> None:
    x, y = _as_int_point(point)
    draw.line((x - 7, y, x + 7, y), fill=color, width=3)
    draw.line((x, y - 7, x, y + 7), fill=color, width=3)
    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    draw.text((x + 9, y - 19), label, fill=color, font=font)


def _draw_measurements_pil(
    base_image,
    state: ClickCalibrationState,
    cursor: Point | None,
    status: str,
):
    from PIL import Image, ImageDraw, ImageFont

    image = base_image.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=13)
    clicks = state.clicks

    if clicks:
        _draw_marker(draw, clicks[0], (255, 225, 0), "BODY REAR", font)
    if len(clicks) >= 2:
        _draw_marker(draw, clicks[1], (255, 225, 0), "BODY FRONT", font)
        draw.line(
            (_as_int_point(clicks[0]), _as_int_point(clicks[1])),
            fill=(255, 225, 0),
            width=3,
        )
    elif len(clicks) == 1 and cursor is not None:
        draw.line(
            (_as_int_point(clicks[0]), _as_int_point(cursor)),
            fill=(255, 225, 0),
            width=2,
        )

    wheel_specs = (
        (2, 3, "VISIBLE LEFT REAR WHEEL", (0, 220, 80)),
        (4, 5, "VISIBLE LEFT FRONT WHEEL", (255, 145, 0)),
    )
    for center_index, edge_index, label, color in wheel_specs:
        if len(clicks) <= center_index:
            continue
        _draw_marker(draw, clicks[center_index], color, label, font)
        if len(clicks) >= 2:
            projection = projected_point_on_body_axis(
                clicks[center_index],
                clicks[0],
                clicks[1],
            )
            draw.line(
                (
                    _as_int_point(clicks[center_index]),
                    _as_int_point(projection),
                ),
                fill=color,
                width=2,
            )
            px, py = _as_int_point(projection)
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color)
        if len(clicks) > edge_index:
            _draw_marker(
                draw,
                clicks[edge_index],
                color,
                f"{label} EDGE",
                font,
            )
            draw.line(
                (
                    _as_int_point(clicks[center_index]),
                    _as_int_point(clicks[edge_index]),
                ),
                fill=color,
                width=3,
            )
        elif len(clicks) == edge_index and cursor is not None:
            draw.line(
                (
                    _as_int_point(clicks[center_index]),
                    _as_int_point(cursor),
                ),
                fill=color,
                width=2,
            )

    if len(clicks) >= 5:
        draw.line(
            (_as_int_point(clicks[2]), _as_int_point(clicks[4])),
            fill=(80, 190, 255),
            width=3,
        )

    panel_lines = [
        state.next_prompt(),
        "Click only points visible in this photo.",
        "Order: body REAR/FRONT -> rear center/edge -> front center/edge",
        "Known body length: 0.99 m",
        "Keys: R reset | U undo | S save | Q quit",
        f"Status: {status}",
    ]
    if state.complete:
        try:
            geometry = calculate_vehicle_geometry(state.to_measurements())
            panel_lines.extend(
                (
                    f"rear axle={geometry.rear_axle_from_rear_m:.4f} m  "
                    f"front axle={geometry.front_axle_from_rear_m:.4f} m",
                    f"photo-derived wheelbase={geometry.wheelbase_m:.4f} m",
                    f"wheel radius mean={geometry.wheel_radius_m:.4f} m",
                    "front/rear track=UNMEASURED (saved as null)",
                )
            )
        except CalibrationError as exc:
            panel_lines.append(f"INVALID: {exc}")

    line_height = 19
    panel_width = min(
        max(
            (draw.textbbox((0, 0), line, font=font)[2] for line in panel_lines),
            default=0,
        )
        + 20,
        image.width - 8,
    )
    panel_height = line_height * len(panel_lines) + 12
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        (4, 8, 4 + panel_width, 8 + panel_height),
        fill=(0, 0, 0, 180),
    )
    for index, line in enumerate(panel_lines):
        overlay_draw.text(
            (12, 14 + index * line_height),
            line,
            fill=(245, 245, 245, 255),
            font=font,
        )
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def run_gui(image_path: Path, output_path: Path) -> int:
    try:
        import tkinter as tk
        from PIL import Image, ImageTk
    except (ModuleNotFoundError, ImportError) as exc:
        raise RuntimeError(
            "The calibration GUI requires the standard tkinter module and Pillow."
        ) from exc

    source = image_path.resolve(strict=False)
    if not source.is_file():
        raise CalibrationError(f"Source image does not exist: {source}")
    try:
        image = Image.open(source).convert("RGB")
    except Exception as exc:
        raise CalibrationError(
            f"Pillow could not decode the source image: {source}"
        ) from exc

    destination = validated_output_path(output_path)
    state = ClickCalibrationState()
    cursor: Point | None = None
    status = "Click the requested visible points in order."
    window_name = "CTRL_ZERO visible-side vehicle calibration"
    print("Vehicle geometry calibration")
    print(
        "Clicks: body REAR end, body FRONT end, REAR wheel center/edge, "
        "FRONT wheel center/edge."
    )
    print("Keys: R reset | U undo | S save | Q quit")
    print(f"Source: {source}")
    print(f"Output: {destination}")

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError(f"Tk could not open a window: {exc}") from exc
    root.title(window_name)
    root.resizable(False, False)
    canvas = tk.Canvas(
        root,
        width=image.width,
        height=image.height,
        highlightthickness=0,
    )
    canvas.pack()
    rendered_photo = None

    def redraw() -> None:
        nonlocal rendered_photo
        rendered = _draw_measurements_pil(
            image,
            state,
            cursor,
            status,
        )
        rendered_photo = ImageTk.PhotoImage(rendered)
        canvas.delete("all")
        canvas.create_image(0, 0, image=rendered_photo, anchor=tk.NW)

    def on_motion(event) -> None:
        nonlocal cursor
        cursor = (float(event.x), float(event.y))
        redraw()

    def on_click(event) -> None:
        nonlocal cursor, status
        cursor = (float(event.x), float(event.y))
        if state.add_click(cursor):
            status = (
                f"Point {len(state.clicks)}/{REQUIRED_CLICK_COUNT} recorded."
            )
        else:
            status = "All points are present. Press S, U, or R."
        redraw()

    def on_key(event) -> None:
        nonlocal status
        key = event.keysym.lower()
        if key in {"q", "escape"}:
            root.destroy()
            return
        if key == "r":
            state.reset()
            status = "Reset complete."
        elif key == "u":
            status = (
                "Removed the last point."
                if state.undo()
                else "Nothing to undo."
            )
        elif key == "s":
            try:
                saved_path, geometry = save_calibration(
                    state,
                    source,
                    destination,
                )
            except CalibrationError as exc:
                status = f"NOT SAVED: {exc}"
                print(status)
            else:
                status = f"Saved: {saved_path.name}"
                print(f"Saved vehicle geometry to {saved_path}")
                print(json.dumps(geometry.config_values(), indent=2))
        redraw()

    canvas.bind("<Motion>", on_motion)
    canvas.bind("<Button-1>", on_click)
    root.bind("<Key>", on_key)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    redraw()
    root.focus_force()
    root.mainloop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Annotate body endpoints and each visible wheel's center/edge "
            "to derive longitudinal axle geometry and wheel radius."
        )
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help=f"Reference image. Default: {DEFAULT_IMAGE_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"JSON output inside sim/. Default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run_gui(args.image, args.output)
    except (CalibrationError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
