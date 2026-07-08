from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ctrl_zero.perception import DetectedObject

Point = tuple[int, int]


@dataclass(frozen=True)
class LaneReference:
    name: str
    near_x: float
    far_x: float | None
    near_y: int
    far_y: int
    width_px: float | None = None
    fit: np.ndarray | None = None


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
    objects: Sequence[DetectedObject] = field(default_factory=tuple)
    lane_label: str = ""
    lane_references: Mapping[str, LaneReference] = field(default_factory=dict)
    curvature: float = 0.0
    lane_pair_label: str = ""
    traffic_light_state: str = "unknown"

    @property
    def has_center(self) -> bool:
        return self.lane_center_near_x is not None
