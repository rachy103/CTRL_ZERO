from __future__ import annotations

import cv2
import numpy as np

from ctrl_zero.vision.base import LaneDetection


class FrameSkippingLaneDetector:
    """Run an expensive detector every N frames and reuse the last lane estimate."""

    def __init__(self, detector, skip: int = 1, stale_confidence_scale: float = 0.85):
        self.detector = detector
        self.skip = max(1, int(skip))
        self.stale_confidence_scale = stale_confidence_scale
        self.frame_index = 0
        self.last_detection: LaneDetection | None = None

    def detect(self, frame):
        self.frame_index += 1
        if self.last_detection is None or (self.frame_index - 1) % self.skip == 0:
            self.last_detection = self.detector.detect(frame)
            return self.last_detection

        reused = reuse_detection_on_frame(frame, self.last_detection, self.stale_confidence_scale)
        self.last_detection = reused
        return reused


def reuse_detection_on_frame(frame, previous: LaneDetection, confidence_scale: float) -> LaneDetection:
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    near_y = h - 1
    far_y = int(h * 0.68)
    _draw_fit(annotated, previous.left_fit, near_y, far_y, (0, 220, 0))
    _draw_fit(annotated, previous.right_fit, near_y, far_y, (0, 220, 0))
    cv2.line(annotated, (w // 2, h - 1), (w // 2, int(h * 0.55)), (0, 0, 255), 1)
    if previous.lane_center_near_x is not None and previous.lane_center_far_x is not None:
        cv2.line(
            annotated,
            (int(previous.lane_center_near_x), near_y),
            (int(previous.lane_center_far_x), far_y),
            (255, 255, 0),
            2,
        )
        cv2.circle(annotated, (int(previous.lane_center_near_x), near_y), 6, (255, 255, 0), -1)
    cv2.putText(annotated, "cached lane", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(annotated, "cached lane", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    return LaneDetection(
        lanes=previous.lanes,
        left_fit=previous.left_fit,
        right_fit=previous.right_fit,
        lane_center_near_x=previous.lane_center_near_x,
        lane_center_far_x=previous.lane_center_far_x,
        frame_center_x=w / 2.0,
        offset_px=previous.offset_px,
        offset_norm=previous.offset_norm,
        heading_deg=previous.heading_deg,
        lane_width_px=previous.lane_width_px,
        confidence=max(0.0, previous.confidence * confidence_scale),
        mask=previous.mask,
        annotated=annotated,
        objects=previous.objects,
        curvature=previous.curvature,
        lane_pair_label=previous.lane_pair_label,
        traffic_light_state=previous.traffic_light_state,
    )


def _draw_fit(image, fit, near_y, far_y, color) -> None:
    if fit is None:
        return
    h, w = image.shape[:2]
    points = []
    for y in range(int(far_y), int(near_y) + 1, 6):
        x = int(max(0, min(w - 1, float(np.polyval(fit, y)))))
        points.append((x, y))
    if len(points) >= 2:
        cv2.polylines(image, [np.array(points, dtype=np.int32)], isClosed=False, color=color, thickness=3)
