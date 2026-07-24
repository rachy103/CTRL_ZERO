from __future__ import annotations

import json
from pathlib import Path

import pytest

from sim.calibrate_vehicle import (
    DEFAULT_IMAGE_PATH,
    REQUIRED_CLICK_COUNT,
    SIM_ROOT,
    CalibrationError,
    ClickCalibrationState,
    IncompleteCalibrationError,
    VehicleMeasurements,
    _draw_measurements_pil,
    build_vehicle_geometry_payload,
    calculate_vehicle_geometry,
    project_distance_from_rear_px,
    save_calibration,
    validated_output_path,
)


def _valid_measurements() -> VehicleMeasurements:
    return VehicleMeasurements(
        body_rear=(0.0, 0.0),
        body_front=(200.0, 0.0),
        rear_left_wheel_center=(40.0, 50.0),
        rear_left_wheel_edge=(40.0, 30.0),
        front_left_wheel_center=(160.0, -40.0),
        front_left_wheel_edge=(160.0, -62.0),
    )


def test_visible_wheels_are_measured_from_body_scale() -> None:
    geometry = calculate_vehicle_geometry(_valid_measurements())

    assert geometry.rear_axle_from_rear_m == pytest.approx(0.198)
    assert geometry.front_axle_from_rear_m == pytest.approx(0.792)
    assert geometry.wheelbase_m == pytest.approx(0.594)
    assert geometry.rear_wheel_radius_m == pytest.approx(0.099)
    assert geometry.front_wheel_radius_m == pytest.approx(0.1089)
    assert geometry.wheel_radius_m == pytest.approx(0.10395)
    assert geometry.rear_wheel_projection_px == pytest.approx((40.0, 0.0))
    assert geometry.front_wheel_projection_px == pytest.approx((160.0, 0.0))
    assert geometry.config_values()["front_track_m"] is None
    assert geometry.config_values()["rear_track_m"] is None
    assert geometry.config_values()["wheel_radius_m"] == pytest.approx(
        0.10395
    )


def test_rotated_body_axis_ignores_perpendicular_offset() -> None:
    rear = (10.0, 20.0)
    front = (110.0, 120.0)
    point = (55.0, 25.0)

    distance = project_distance_from_rear_px(point, rear, front)

    assert distance == pytest.approx((2**0.5) * 25.0)


def test_click_state_requires_six_points_and_supports_undo_reset() -> None:
    state = ClickCalibrationState()
    for index in range(REQUIRED_CLICK_COUNT):
        assert state.add_click((index, index + 1))

    assert REQUIRED_CLICK_COUNT == 6
    assert state.complete
    assert not state.add_click((99, 99))
    measurements = state.to_measurements()
    assert measurements.body_rear == (0.0, 1.0)
    assert measurements.front_left_wheel_edge == (5.0, 6.0)

    assert state.undo()
    assert not state.complete
    with pytest.raises(IncompleteCalibrationError):
        state.to_measurements()

    state.reset()
    assert state.clicks == []
    assert not state.undo()


def test_every_partial_click_state_can_redraw_with_cursor() -> None:
    from PIL import Image

    ordered_points = list(_valid_measurements().points())
    base_image = Image.new("RGB", (800, 531), "white")

    for click_count in range(REQUIRED_CLICK_COUNT + 1):
        state = ClickCalibrationState(ordered_points[:click_count])
        rendered = _draw_measurements_pil(
            base_image,
            state,
            cursor=(420.0, 260.0),
            status=f"{click_count} points",
        )

        assert rendered.size == base_image.size
        assert rendered.mode == "RGB"


def test_incomplete_input_never_creates_output(tmp_path: Path) -> None:
    state = ClickCalibrationState()
    state.add_click((0, 0))
    outside_output = tmp_path / "must_not_exist.json"

    with pytest.raises(IncompleteCalibrationError):
        save_calibration(state, DEFAULT_IMAGE_PATH, outside_output)

    assert not outside_output.exists()


@pytest.mark.parametrize(
    ("measurements", "message"),
    [
        (
            VehicleMeasurements(
                (0, 0), (0, 0), (1, 1), (1, 2), (2, 2), (2, 3)
            ),
            "distinct endpoints",
        ),
        (
            VehicleMeasurements(
                (0, 0), (100, 0), (-1, 5), (-1, 6), (80, 5), (80, 6)
            ),
            "Rear wheel projection",
        ),
        (
            VehicleMeasurements(
                (0, 0), (100, 0), (20, 5), (20, 6), (101, 5), (101, 6)
            ),
            "Front wheel projection",
        ),
        (
            VehicleMeasurements(
                (0, 0), (100, 0), (80, 5), (80, 6), (20, 5), (20, 6)
            ),
            "Front wheel must project ahead",
        ),
        (
            VehicleMeasurements(
                (0, 0),
                (100, 0),
                (float("nan"), 5),
                (20, 6),
                (80, 5),
                (80, 6),
            ),
            "finite",
        ),
        (
            VehicleMeasurements(
                (0, 0), (100, 0), (20, 5), (20, 5), (80, 5), (80, 6)
            ),
            "Rear wheel center and tire edge",
        ),
        (
            VehicleMeasurements(
                (0, 0), (100, 0), (20, 5), (20, 6), (80, 5), (80, 5)
            ),
            "Front wheel center and tire edge",
        ),
    ],
)
def test_invalid_annotations_are_rejected(
    measurements: VehicleMeasurements,
    message: str,
) -> None:
    with pytest.raises(CalibrationError, match=message):
        calculate_vehicle_geometry(measurements)


def test_output_path_must_be_json_inside_sim() -> None:
    inside = SIM_ROOT / "config" / "test_geometry.json"
    assert validated_output_path(inside) == inside.resolve(strict=False)

    with pytest.raises(CalibrationError, match="inside"):
        validated_output_path(SIM_ROOT.parent / "outside.json")
    with pytest.raises(CalibrationError, match=r"\.json"):
        validated_output_path(SIM_ROOT / "config" / "vehicle_geometry.txt")


def test_payload_records_measurement_scope_and_unknown_values() -> None:
    measurements = _valid_measurements()
    geometry = calculate_vehicle_geometry(measurements)
    payload = build_vehicle_geometry_payload(
        measurements, geometry, DEFAULT_IMAGE_PATH
    )

    assert payload["schema_version"] == 1
    assert payload["status"] == "visible_left_side_annotated"
    assert payload["vehicle_length_m"] == 0.99
    measured = payload["vehicle_geometry"]
    assert measured["wheelbase_m"] == pytest.approx(0.594)
    assert measured["wheel_radius_m"] == pytest.approx(0.10395)
    assert measured["front_track_m"] is None
    assert measured["rear_track_m"] is None

    calibration = payload["calibration"]
    assert (
        calibration["method"]
        == "affine_projection_on_visible_left_side_body_axis"
    )
    assert calibration["source_image"] == (
        "assets/source/vehicle_side_profile.png"
    )
    assert set(calibration["points_px"]) == {
        "body_rear",
        "body_front",
        "rear_left_wheel_center",
        "rear_left_wheel_edge",
        "front_left_wheel_center",
        "front_left_wheel_edge",
    }
    assert set(calibration["unmeasured"]) == {
        "front_track_m",
        "rear_track_m",
    }
    assert calibration["wheel_material"] == "polyurethane"
    serialized = json.dumps(payload).lower()
    assert "perspective" in serialized
    assert "wheel_width_m" not in serialized


def test_complete_valid_state_saves_wrapped_json_inside_sim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measurements = _valid_measurements()
    state = ClickCalibrationState(list(measurements.points()))
    destination = SIM_ROOT / ".tmp" / "tests" / "vehicle-calibration-save.json"
    written: dict[str, object] = {}

    def fake_write_json(path: Path, payload: dict[str, object]) -> None:
        written["path"] = path
        written["payload"] = payload

    monkeypatch.setattr(
        "sim.calibrate_vehicle.write_json_in_sim", fake_write_json
    )
    saved_path, geometry = save_calibration(
        state, DEFAULT_IMAGE_PATH, destination
    )

    assert saved_path == destination.resolve(strict=False)
    assert written["path"] == saved_path
    assert written["payload"]["vehicle_geometry"]["front_track_m"] is None
    assert written["payload"]["vehicle_geometry"][
        "wheel_radius_m"
    ] == pytest.approx(0.10395)
    assert geometry.wheelbase_m == pytest.approx(0.594)
