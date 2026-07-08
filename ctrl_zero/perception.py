from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def bottom_y(self) -> float:
        return self.y2

    def clipped(self, width: int, height: int) -> "BoundingBox":
        max_x = max(float(width - 1), 0.0)
        max_y = max(float(height - 1), 0.0)
        return BoundingBox(
            x1=min(max(self.x1, 0.0), max_x),
            y1=min(max(self.y1, 0.0), max_y),
            x2=min(max(self.x2, 0.0), max_x),
            y2=min(max(self.y2, 0.0), max_y),
        )


@dataclass(frozen=True)
class DetectedObject:
    class_name: str
    confidence: float
    bbox: BoundingBox
    class_id: int | None = None
    mask_area_px: float | None = None
    lane_label: str = ""
    lane_distance_px: float | None = None

    @property
    def compact_class_name(self) -> str:
        return compact_class_name(self.class_name)


def compact_class_name(class_name: str) -> str:
    return "".join(ch for ch in str(class_name).lower() if ch.isalnum())
