from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

Point = tuple[int, int]


@dataclass
class LaneDetection:
    lanes: Sequence[Sequence[Point]]
    left_fit: np.ndarray | None
    right_fit: np.ndarray | None
    lane_center_near_x: float | None
    lane_center_far_x: float | None
    frame_center_x: float
    offset_px: float | None
    offset_norm: float | None
    heading_deg: float | None
    lane_width_px: float | None
    confidence: float
    mask: np.ndarray | None
    annotated: np.ndarray
    curvature: float = 0.0

    @property
    def has_center(self) -> bool:
        return self.lane_center_near_x is not None
