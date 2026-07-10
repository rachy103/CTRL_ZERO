from __future__ import annotations

import pytest

from main import build_parser, build_preprocessor, parse_ratio_point


def test_parse_ratio_point_accepts_x_y_pair():
    assert parse_ratio_point("0.38,0.35") == (0.38, 0.35)


def test_parse_ratio_point_rejects_out_of_range_values():
    with pytest.raises(Exception):
        parse_ratio_point("1.2,0.35")


def test_bird_eye_cli_builds_enabled_preprocessor():
    args = build_parser().parse_args(
        [
            "--bird-eye",
            "--bird-eye-src-bottom-left",
            "0.20,0.95",
            "--bird-eye-src-bottom-right",
            "0.80,0.95",
            "--bird-eye-src-top-right",
            "0.60,0.45",
            "--bird-eye-src-top-left",
            "0.40,0.45",
            "--bird-eye-dst-margin",
            "0.12",
            "--no-bird-eye-mask-source-polygon",
        ]
    )

    preprocessor = build_preprocessor(args)

    assert preprocessor.bird_eye.enabled
    assert preprocessor.bird_eye.src_bottom_left == (0.20, 0.95)
    assert preprocessor.bird_eye.src_bottom_right == (0.80, 0.95)
    assert preprocessor.bird_eye.src_top_right == (0.60, 0.45)
    assert preprocessor.bird_eye.src_top_left == (0.40, 0.45)
    assert preprocessor.bird_eye.dst_margin_ratio == 0.12
    assert not preprocessor.bird_eye.mask_source_polygon


def test_roi_cli_builds_enabled_crop_config():
    args = build_parser().parse_args(
        [
            "--roi",
            "--roi-top",
            "0.20",
            "--roi-bottom",
            "0.95",
            "--roi-left",
            "0.10",
            "--roi-right",
            "0.90",
        ]
    )

    preprocessor = build_preprocessor(args)

    assert preprocessor.roi.enabled
    assert preprocessor.roi.top_ratio == 0.20
    assert preprocessor.roi.bottom_ratio == 0.95
    assert preprocessor.roi.left_ratio == 0.10
    assert preprocessor.roi.right_ratio == 0.90
