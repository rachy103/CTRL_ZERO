"""Stop line + crosswalk detection (rotation tolerant, raw camera frame).

Pattern: a crosswalk is a group of >= N similar, elongated, roughly parallel
white blobs (stripes).  The stop line is a long elongated white blob BELOW the
stripes (closer to the car).  Nothing assumes perfectly vertical/horizontal
lines: blobs are measured with minAreaRect and the stop line with fitLine, so
the view can be rotated while the car turns.

Distance kept to the line = gap between the frame bottom (the car) and the
stop line, measured at the frame centre x along the fitted (tilted) line.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class StopLineConfig:
    search_top_ratio: float = 0.30        # ignore everything above this y ratio
    white_min_value: int = 160            # HSV V
    white_max_saturation: int = 120       # HSV S
    min_blob_area_ratio: float = 0.0005   # of frame area, noise filter
    # Crosswalk stripes
    stripe_min_aspect: float = 2.0        # long / short side of minAreaRect
    crosswalk_min_stripes: int = 3
    stripe_size_tolerance: float = 0.6    # |area - median| / median allowed
    stripe_angle_tolerance_deg: float = 25.0
    # Stop line (near line)
    line_min_length_ratio: float = 0.35   # long side / frame width
    line_min_aspect: float = 4.0
    require_crosswalk: bool = True


@dataclass(frozen=True)
class StopLineResult:
    detected: bool
    gap_ratio: float | None = None        # (H - line_y_at_centre) / H, 0 = touching the car
    line_y: int | None = None             # line y at frame centre x
    line_pts: tuple[tuple[int, int], tuple[int, int]] | None = None
    crosswalk: bool = False
    stripes: int = 0


def _angle_of(rect) -> float:
    (_, _), (w, h), ang = rect
    # Angle of the LONG side, in [0, 180).
    return (ang if w >= h else ang + 90.0) % 180.0


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def detect_stop_line(frame: np.ndarray | None, lane_references=None, config: StopLineConfig | None = None) -> StopLineResult:
    config = config or StopLineConfig()
    if frame is None or frame.size == 0:
        return StopLineResult(False)
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 2] >= config.white_min_value) & (hsv[:, :, 1] <= config.white_max_saturation)).astype(np.uint8)
    white[: int(h * config.search_top_ratio)] = 0
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_area = config.min_blob_area_ratio * w * h
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        rect = cv2.minAreaRect(c)
        long_side, short_side = max(rect[1]), max(min(rect[1]), 1.0)
        blobs.append((c, area, rect, long_side, long_side / short_side))

    # Stop-line candidates: long and thin.
    lines: list = []
    # Stripe candidates: elongated, not a stop line.
    stripes = [b for b in blobs if b[4] >= config.stripe_min_aspect and b[3] < 0.6 * w]

    # Crosswalk = >= N similar-size elongated blobs side by side (perspective
    # fans their angles out, so no parallel check; they must overlap in y).
    stripe_group: list = []
    if len(stripes) >= config.crosswalk_min_stripes:
        median_area = float(np.median([s[1] for s in stripes]))
        cand = [s for s in stripes if abs(s[1] - median_area) <= config.stripe_size_tolerance * median_area]
        boxes = [cv2.boundingRect(s[0]) for s in cand]
        best_group: list = []
        for i, (_, y, _, bh) in enumerate(boxes):
            group = [cand[j] for j, (_, y2, _, bh2) in enumerate(boxes) if min(y + bh, y2 + bh2) - max(y, y2) > 0.3 * min(bh, bh2)]
            if len(group) > len(best_group):
                best_group = group
        stripe_group = best_group
    crosswalk = len(stripe_group) >= config.crosswalk_min_stripes
    if not crosswalk:
        return StopLineResult(False, stripes=len(stripe_group))
    # Reference = leading (far / top) edge of the pattern, fitted along the
    # stripe tips so it follows the tilt.
    tips = []
    for s in stripe_group:
        pts = s[0].reshape(-1, 2)
        tips.append(pts[np.argmin(pts[:, 1])])
    tips = np.array(tips, dtype=np.float32)
    # Median tip y is robust to one stripe merging with the background.
    # Tilt from a fit, but only trusted within +/-20 deg.
    y_c = float(np.median(tips[:, 1]))
    slope = 0.0
    vx, vy, _, _ = cv2.fitLine(tips, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    if abs(vx) > 1e-3 and abs(vy / vx) <= np.tan(np.radians(20.0)):
        slope = float(vy / vx)
    pts = ((0, int(y_c - (w / 2.0) * slope)), (w, int(y_c + (w / 2.0) * slope)))
    y_c = min(max(y_c, 0.0), float(h))
    return StopLineResult(True, (h - y_c) / h, int(y_c), pts, True, len(stripe_group))


def stop_line_reached(result: StopLineResult, stop_gap_ratio: float) -> bool:
    """True when the gap between the car (frame bottom) and the line <= stop_gap_ratio."""
    return result.detected and result.gap_ratio is not None and result.gap_ratio <= stop_gap_ratio


def draw_stop_line(image: np.ndarray, result: StopLineResult, stop_gap_ratio: float) -> None:
    h, w = image.shape[:2]
    stop_y = int(h * (1.0 - stop_gap_ratio))
    cv2.line(image, (0, stop_y), (w, stop_y), (0, 165, 255), 1)
    if result.line_pts is not None:
        color = (0, 0, 255) if result.detected else (200, 200, 200)
        cv2.line(image, result.line_pts[0], result.line_pts[1], color, 2)
        cv2.putText(image, f"stopline gap={result.gap_ratio:.2f} cw={int(result.crosswalk)} s={result.stripes}",
                    (5, max(12, (result.line_y or 12) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
