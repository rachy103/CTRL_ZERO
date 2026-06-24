from __future__ import annotations

import numpy as np

from ctrl_zero.vision.preprocess import BirdEyeConfig, LanePreprocessor, ROICropConfig


def test_roi_crop_maps_points_and_mask_to_original_coordinates():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    preprocessor = LanePreprocessor(
        roi=ROICropConfig(enabled=True, top_ratio=0.40, bottom_ratio=1.0, left_ratio=0.10, right_ratio=0.90),
        bird_eye=BirdEyeConfig(enabled=False),
    )

    cropped, transform = preprocessor.apply(frame)

    assert cropped.shape[:2] == (60, 160)
    assert transform.points_to_original([(0, 0), (159, 59)]) == [(20, 40), (179, 99)]

    mask = np.full((60, 160), 255, dtype=np.uint8)
    restored = transform.mask_to_original(mask)

    assert restored.shape == (100, 200)
    assert restored[40, 20] == 255
    assert restored[99, 179] == 255
    assert restored[39, 20] == 0
    assert restored[40, 19] == 0


def test_identity_bird_eye_preserves_points():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    preprocessor = LanePreprocessor(
        roi=ROICropConfig(enabled=False),
        bird_eye=BirdEyeConfig(
            enabled=True,
            src_bottom_left=(0.0, 1.0),
            src_bottom_right=(1.0, 1.0),
            src_top_right=(1.0, 0.0),
            src_top_left=(0.0, 0.0),
            dst_margin_ratio=0.0,
        ),
    )

    warped, transform = preprocessor.apply(frame)

    assert warped.shape[:2] == frame.shape[:2]
    assert transform.points_to_original([(10, 20), (119, 79)]) == [(10, 20), (119, 79)]
