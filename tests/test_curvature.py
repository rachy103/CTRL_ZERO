from __future__ import annotations

import numpy as np
import pytest

from ctrl_zero.vision.classical_lane import ClassicalLaneConfig, ClassicalLaneDetector
from ctrl_zero.vision.yolo_lane import YOLOLaneDetector


@pytest.mark.parametrize(
    ("fit", "expected_sign"),
    [
        (np.array([0.0, 0.25, 100.0]), 0),
        (np.array([0.002, -0.8, 100.0]), 1),
        (np.array([-0.002, 0.8, 500.0]), -1),
    ],
)
def test_signed_curvature_for_straight_right_and_left(fit, expected_sign):
    curvature = ClassicalLaneDetector._curvature_from_fit(fit, y_eval=200.0)
    if expected_sign == 0:
        assert curvature == pytest.approx(0.0, abs=1e-12)
    else:
        assert np.sign(curvature) == expected_sign
        assert abs(curvature) == pytest.approx(0.004, rel=0.01)


def test_linear_fallback_still_returns_three_coefficients():
    detector = ClassicalLaneDetector(ClassicalLaneConfig(min_samples_for_quad=6))
    samples = [(100, 100, 20.0), (120, 200, 20.0)]
    fit = detector._fit_x_as_function_of_y(samples)
    assert fit is not None
    assert len(fit) == 3
    assert fit[0] == pytest.approx(0.0)


def test_yolo_curvature_matches_classical_formula():
    fit = np.array([0.002, -0.8, 100.0])
    assert YOLOLaneDetector._curvature_from_fit(fit, 200.0) == pytest.approx(
        ClassicalLaneDetector._curvature_from_fit(fit, 200.0)
    )
