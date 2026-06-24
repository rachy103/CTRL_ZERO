from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ctrl_zero.common import clamp
from ctrl_zero.vision.base import Point


RatioPoint = tuple[float, float]


@dataclass(frozen=True)
class ROICropConfig:
    enabled: bool = True
    top_ratio: float = 0.35
    bottom_ratio: float = 1.0
    left_ratio: float = 0.0
    right_ratio: float = 1.0


@dataclass(frozen=True)
class BirdEyeConfig:
    enabled: bool = False
    src_bottom_left: RatioPoint = (0.10, 0.98)
    src_bottom_right: RatioPoint = (0.90, 0.98)
    src_top_right: RatioPoint = (0.62, 0.35)
    src_top_left: RatioPoint = (0.38, 0.35)
    dst_margin_ratio: float = 0.18


@dataclass(frozen=True)
class FrameTransform:
    original_width: int
    original_height: int
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int
    perspective_matrix: np.ndarray | None = None
    inverse_perspective_matrix: np.ndarray | None = None
    bird_eye_source_points: np.ndarray | None = None

    def points_to_original(self, points: list[Point]) -> list[Point]:
        if not points:
            return []

        coords = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        if self.inverse_perspective_matrix is not None:
            coords = cv2.perspectiveTransform(coords, self.inverse_perspective_matrix)

        coords = coords.reshape(-1, 2)
        coords[:, 0] += self.crop_x
        coords[:, 1] += self.crop_y

        mapped: list[Point] = []
        for x, y in coords:
            mapped.append(
                (
                    int(round(clamp(float(x), 0.0, self.original_width - 1.0))),
                    int(round(clamp(float(y), 0.0, self.original_height - 1.0))),
                )
            )
        return mapped

    def mask_to_original(self, mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None:
            return None

        restored = mask
        if self.inverse_perspective_matrix is not None:
            restored = cv2.warpPerspective(
                restored,
                self.inverse_perspective_matrix,
                (self.crop_width, self.crop_height),
                flags=cv2.INTER_NEAREST,
            )

        output = np.zeros((self.original_height, self.original_width), dtype=restored.dtype)
        y2 = min(self.crop_y + self.crop_height, self.original_height)
        x2 = min(self.crop_x + self.crop_width, self.original_width)
        crop_h = max(0, y2 - self.crop_y)
        crop_w = max(0, x2 - self.crop_x)
        if crop_h > 0 and crop_w > 0:
            output[self.crop_y : y2, self.crop_x : x2] = restored[:crop_h, :crop_w]
        return output

    def draw_overlay(self, image: np.ndarray) -> None:
        roi_color = (255, 120, 0)
        cv2.rectangle(
            image,
            (self.crop_x, self.crop_y),
            (self.crop_x + self.crop_width - 1, self.crop_y + self.crop_height - 1),
            roi_color,
            2,
        )

        if self.bird_eye_source_points is None:
            return

        points = self.bird_eye_source_points.copy()
        points[:, 0] += self.crop_x
        points[:, 1] += self.crop_y
        cv2.polylines(image, [points.astype(np.int32)], isClosed=True, color=(255, 0, 255), thickness=2)


class LanePreprocessor:
    """ROI crop and optional bird-eye transform for lane detector input."""

    def __init__(self, roi: ROICropConfig | None = None, bird_eye: BirdEyeConfig | None = None):
        self.roi = roi or ROICropConfig(enabled=False)
        self.bird_eye = bird_eye or BirdEyeConfig(enabled=False)

    def apply(self, frame: np.ndarray) -> tuple[np.ndarray, FrameTransform]:
        original_h, original_w = frame.shape[:2]
        x1, y1, x2, y2 = self._crop_bounds(original_w, original_h)
        cropped = frame[y1:y2, x1:x2]
        crop_h, crop_w = cropped.shape[:2]

        matrix = None
        inverse_matrix = None
        source_points = None
        processed = cropped

        if self.bird_eye.enabled and crop_w >= 2 and crop_h >= 2:
            source_points = self._source_points(crop_w, crop_h)
            destination_points = self._destination_points(crop_w, crop_h)
            matrix = cv2.getPerspectiveTransform(source_points, destination_points)
            inverse_matrix = cv2.getPerspectiveTransform(destination_points, source_points)
            processed = cv2.warpPerspective(cropped, matrix, (crop_w, crop_h), flags=cv2.INTER_LINEAR)

        return processed, FrameTransform(
            original_width=original_w,
            original_height=original_h,
            crop_x=x1,
            crop_y=y1,
            crop_width=crop_w,
            crop_height=crop_h,
            perspective_matrix=matrix,
            inverse_perspective_matrix=inverse_matrix,
            bird_eye_source_points=source_points,
        )

    def _crop_bounds(self, width: int, height: int) -> tuple[int, int, int, int]:
        if not self.roi.enabled:
            return 0, 0, width, height

        left = clamp(self.roi.left_ratio, 0.0, 0.98)
        right = clamp(self.roi.right_ratio, left + 0.01, 1.0)
        top = clamp(self.roi.top_ratio, 0.0, 0.98)
        bottom = clamp(self.roi.bottom_ratio, top + 0.01, 1.0)

        x1 = int(round(width * left))
        x2 = int(round(width * right))
        y1 = int(round(height * top))
        y2 = int(round(height * bottom))

        x1 = int(clamp(x1, 0, width - 1))
        x2 = int(clamp(x2, x1 + 1, width))
        y1 = int(clamp(y1, 0, height - 1))
        y2 = int(clamp(y2, y1 + 1, height))
        return x1, y1, x2, y2

    def _source_points(self, width: int, height: int) -> np.ndarray:
        points = (
            self.bird_eye.src_bottom_left,
            self.bird_eye.src_bottom_right,
            self.bird_eye.src_top_right,
            self.bird_eye.src_top_left,
        )
        return self._ratio_points(points, width, height)

    def _destination_points(self, width: int, height: int) -> np.ndarray:
        margin = clamp(self.bird_eye.dst_margin_ratio, 0.0, 0.45) * (width - 1)
        return np.array(
            [
                [margin, height - 1.0],
                [width - 1.0 - margin, height - 1.0],
                [width - 1.0 - margin, 0.0],
                [margin, 0.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _ratio_points(points: tuple[RatioPoint, ...], width: int, height: int) -> np.ndarray:
        scaled = []
        for x_ratio, y_ratio in points:
            x = clamp(x_ratio, 0.0, 1.0) * (width - 1)
            y = clamp(y_ratio, 0.0, 1.0) * (height - 1)
            scaled.append([x, y])
        return np.array(scaled, dtype=np.float32)
