"""Parking-slot detector (``parking_v2_openvino_model``) for the rear reverse.

The model is a YOLO11n-pose net with one class, ``parking_slot``.  It also emits
four corner keypoints, but those are NOT used: the near corners fall outside the
rear view about half the time, where the model clamps them to the border and
flags them with ~0.02 confidence, and steering off them made the heading jump by
tens of degrees between consecutive frames.

Only the BBOX is used, as the search region for
:func:`~ctrl_zero.vision.parking_line.prepare_infinite_lane_frame`, which finds
the bay's side lines inside it and repaints them for the lane model.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlotDetection:
    """One parking-slot detection: the box and its score."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float


class ParkingSlotDetector:
    """Thin wrapper over the Ultralytics pose model (OpenVINO or .pt)."""

    def __init__(self, model_path, device: str = "cpu", image_size: int = 640,
                 confidence: float = 0.25, iou: float = 0.45):
        from ultralytics import YOLO

        self.model = YOLO(str(model_path), task="pose")
        self.device = device
        self.image_size = image_size
        self.confidence = confidence
        self.iou = iou

    def detect(self, frame_bgr) -> list[SlotDetection]:
        results = self.model.predict(
            frame_bgr, imgsz=self.image_size, conf=self.confidence,
            iou=self.iou, device=self.device, verbose=False,
        )
        if not results:
            return []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        return [
            SlotDetection(
                bbox=tuple(float(v) for v in boxes[i]),
                confidence=float(confs[i]),
            )
            for i in range(len(boxes))
        ]


def best_slot(dets: list[SlotDetection], frame_w: int) -> SlotDetection | None:
    """Pick the slot to drive into: the largest one nearest the frame centre."""
    if not dets:
        return None

    def score(d: SlotDetection) -> float:
        x1, y1, x2, y2 = d.bbox
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        centre_pen = abs((x1 + x2) / 2.0 - frame_w / 2.0) / max(frame_w, 1)
        return area * (1.0 - 0.5 * centre_pen)

    return max(dets, key=score)
