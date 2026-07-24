from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from sim import build_map
from sim.config import ConfigurationError


def _rgb_source_with_artwork() -> tuple[np.ndarray, np.ndarray]:
    source = np.full((30, 44, 3), 255, dtype=np.uint8)
    y1, y2 = 7, 23
    x1, x2 = 9, 35
    yy, xx = np.indices((y2 - y1, x2 - x1))
    artwork = np.empty((y2 - y1, x2 - x1, 3), dtype=np.uint8)
    artwork[:, :, 0] = 30 + xx * 3
    artwork[:, :, 1] = 110 + yy * 4
    artwork[:, :, 2] = 20
    source[y1:y2, x1:x2] = artwork

    # A smaller colored object must not displace the main artwork component.
    source[1:3, 1:3] = (20, 180, 20)
    return source, artwork


def _plan_source_with_cyan_frame() -> np.ndarray:
    source = np.full((34, 46, 3), 255, dtype=np.uint8)
    x1, y1, x2, y2 = 6, 5, 38, 27
    cyan = (245, 245, 225)
    source[y1, x1:x2] = cyan
    source[y2 - 1, x1:x2] = cyan
    source[y1:y2, x1] = cyan
    source[y1:y2, x2 - 1] = cyan
    return source


def test_detect_rgb_artwork_bbox_selects_largest_colored_region() -> None:
    source, _ = _rgb_source_with_artwork()

    assert build_map.detect_rgb_artwork_bbox(source) == (9, 7, 35, 23)


def test_detect_rgb_artwork_bbox_rejects_blank_screenshot() -> None:
    source = np.full((40, 60, 3), 255, dtype=np.uint8)

    with pytest.raises(ConfigurationError, match="colored map region"):
        build_map.detect_rgb_artwork_bbox(source)


def test_detect_plan_frame_bbox_returns_exclusive_cyan_bounds() -> None:
    source = _plan_source_with_cyan_frame()

    assert build_map.detect_plan_frame_bbox(source) == (6, 5, 38, 27)


def test_detect_plan_frame_bbox_rejects_insufficient_cyan_pixels() -> None:
    source = np.full((20, 20, 3), 255, dtype=np.uint8)
    source[3, 4:14] = (245, 245, 225)

    with pytest.raises(ConfigurationError, match="cyan plan boundary"):
        build_map.detect_plan_frame_bbox(source)


@pytest.mark.parametrize(
    ("physical_width_m", "physical_height_m", "long_edge_px", "expected"),
    [
        (8.0, 6.0, 1024, (1024, 768)),
        (6.0, 8.0, 1024, (768, 1024)),
        (4.0, 4.0, 513, (513, 513)),
        (10.0, 3.0, 1001, (1001, 300)),
    ],
)
def test_target_texture_size_applies_physical_aspect_ratio(
    physical_width_m: float,
    physical_height_m: float,
    long_edge_px: int,
    expected: tuple[int, int],
) -> None:
    assert build_map.target_texture_size(
        physical_width_m,
        physical_height_m,
        long_edge_px,
    ) == expected


@pytest.mark.parametrize(
    ("physical_width_m", "physical_height_m", "long_edge_px", "message"),
    [
        (0.0, 1.0, 256, "positive"),
        (1.0, -1.0, 256, "positive"),
        (1.0, 1.0, 255, "at least 256"),
    ],
)
def test_target_texture_size_rejects_invalid_dimensions(
    physical_width_m: float,
    physical_height_m: float,
    long_edge_px: int,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        build_map.target_texture_size(
            physical_width_m,
            physical_height_m,
            long_edge_px,
        )


def test_build_map_assets_crops_and_resizes_with_lanczos_in_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rgb_source, rgb_artwork = _rgb_source_with_artwork()
    plan_source = _plan_source_with_cyan_frame()
    config = {
        "map": {
            "rgb_image": "assets/source/track_rgb.png",
            "plan_image": "assets/source/track_plan.png",
            "texture_long_edge_px": 256,
            "plan_rotation_clockwise_quarters": 0,
            "physical_width_m": 4.0,
            "physical_height_m": 2.0,
        }
    }

    original_resize = cv2.resize
    expected_texture = original_resize(
        rgb_artwork,
        (256, 128),
        interpolation=cv2.INTER_LANCZOS4,
    )
    resize_calls: list[tuple[np.ndarray, tuple[int, int], int]] = []

    def recording_resize(
        image: np.ndarray,
        size: tuple[int, int],
        *,
        interpolation: int,
    ) -> np.ndarray:
        resize_calls.append((image.copy(), size, interpolation))
        return original_resize(image, size, interpolation=interpolation)

    def read_image(_path: Path, label: str) -> np.ndarray:
        if label == "RGB map image":
            return rgb_source.copy()
        if label == "track plan image":
            return plan_source.copy()
        raise AssertionError(f"Unexpected image label: {label}")

    written_images: dict[str, np.ndarray] = {}

    def capture_image(path: str, image: np.ndarray) -> bool:
        written_images[Path(path).name] = image.copy()
        return True

    written_metadata: dict[str, object] = {}

    def capture_json(path: Path, value: dict[str, object]) -> Path:
        written_metadata["path"] = path
        written_metadata["value"] = value
        return path

    generated_dir = Mock()
    monkeypatch.setattr(build_map, "GENERATED_DIR", generated_dir)
    monkeypatch.setattr(build_map, "_read_image", read_image)
    monkeypatch.setattr(build_map, "_sha256", lambda _path: "in-memory-sha256")
    monkeypatch.setattr(build_map.cv2, "resize", recording_resize)
    monkeypatch.setattr(build_map.cv2, "imwrite", capture_image)
    monkeypatch.setattr(build_map, "write_json_in_sim", capture_json)

    outputs = build_map.build_map_assets(config)

    assert resize_calls[0][1:] == ((256, 128), cv2.INTER_LANCZOS4)
    np.testing.assert_array_equal(resize_calls[0][0], rgb_artwork)
    np.testing.assert_array_equal(
        written_images["track_texture.png"],
        expected_texture,
    )
    assert written_images["track_alignment_preview.png"].shape == (128, 256, 3)
    assert outputs["texture"].name == "track_texture.png"

    metadata = written_metadata["value"]
    assert isinstance(metadata, dict)
    assert metadata["rgb_crop_xyxy_px"] == [9, 7, 35, 23]
    assert metadata["plan_frame_xyxy_px"] == [6, 5, 38, 27]
    assert metadata["physical_size_m"] == [4.0, 2.0]
    assert metadata["texture_size_px"] == [256, 128]
    assert metadata["interpolation"] == "OpenCV INTER_LANCZOS4"
    generated_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_preview_uses_rotated_plan_aspect_not_rgb_screenshot_aspect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rgb_source, _ = _rgb_source_with_artwork()  # crop ratio 26:16
    plan_source = _plan_source_with_cyan_frame()  # crop ratio 32:22
    config = {
        "map": {
            "rgb_image": "assets/source/track_rgb.png",
            "plan_image": "assets/source/track_plan.png",
            "texture_long_edge_px": 320,
            "plan_rotation_clockwise_quarters": 0,
            "physical_width_m": None,
            "physical_height_m": None,
        }
    }
    written_images: dict[str, np.ndarray] = {}
    metadata: dict[str, object] = {}

    monkeypatch.setattr(
        build_map,
        "_read_image",
        lambda _path, label: (
            rgb_source.copy()
            if label == "RGB map image"
            else plan_source.copy()
        ),
    )
    monkeypatch.setattr(build_map, "_sha256", lambda _path: "sha")
    monkeypatch.setattr(
        build_map.cv2,
        "imwrite",
        lambda path, image: (
            written_images.__setitem__(Path(path).name, image.copy()) or True
        ),
    )
    monkeypatch.setattr(
        build_map,
        "write_json_in_sim",
        lambda path, value: (metadata.update(value) or path),
    )
    monkeypatch.setattr(build_map, "GENERATED_DIR", Mock())

    build_map.build_map_assets(config, preview_only=True)

    expected_height = round(320 * 22 / 32)
    texture = written_images["track_texture_unscaled_preview.png"]
    assert texture.shape[:2] == (expected_height, 320)
    assert metadata["texture_size_px"] == [320, expected_height]
