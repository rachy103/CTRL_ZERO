"""Classic-CV heading guide for parking_sim's initial straight drive.

During SEARCH / APPROACH the car should drive dead straight, but open-loop
``steer = 0`` drifts.  This module looks at the LEFT HALF of the camera frame,
finds the near-vertical line-like features there (they are not real lane lines,
just two roughly vertical marks), takes the MIDLINE between the two, and returns
a steering command that keeps the car aligned with (parallel to, and centred on)
that midline.

No YOLO — plain Canny + probabilistic Hough, so it is cheap enough to run every
frame.  ``parking_guide_steer`` returns ``(steer | None, annotated_bgr)``; the
annotated image is a full debug view so you can verify, at a glance, that lines
are actually being extracted and that the resulting steer is what feeds control:

* yellow rectangle  = the ROI actually processed
* magenta pixels    = raw Canny edges inside the ROI (what the detector "sees")
* orange lines      = Hough segments ACCEPTED as near-vertical
* grey lines        = Hough segments REJECTED (too tilted)
* green line        = the midline whose heading/offset drives the steer
* HUD + bottom bar  = line counts, angle, offset, and the guide steer value

Everything is tunable via :class:`ParkingGuideConfig`; the sign of the two gains
is track-dependent, so flip them if the correction pushes the wrong way.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ParkingGuideConfig:
    roi_left_ratio: float = 0.5        # use the leftmost this fraction of the width
    roi_top_ratio: float = 0.30        # ignore the top this fraction (far/sky)
    canny_lo: int = 50
    canny_hi: int = 150
    hough_threshold: int = 40
    hough_min_line_frac: float = 0.25  # min line length as a fraction of ROI height
    hough_max_gap: int = 30
    max_vertical_dev_deg: float = 35.0  # keep only lines within this of vertical
    heading_gain: float = 2.0          # steer per deg of midline tilt (sign: tune)
    offset_gain: float = 40.0          # steer per normalized lateral offset (sign: tune)
    steer_limit: int = 60              # clamp on the guide steer


def parking_guide_steer(frame_bgr, config: ParkingGuideConfig | None = None):
    """Return ``(steer:int|None, annotated_bgr)`` from the left-half ROI.

    steer is ``None`` if fewer than two near-vertical lines are found (the caller
    should then fall back to the mission's own straight steer).
    """
    cfg = config or ParkingGuideConfig()
    if frame_bgr is None:
        return None, frame_bgr

    h, w = frame_bgr.shape[:2]
    x_right = max(1, int(w * cfg.roi_left_ratio))
    y_top = int(h * cfg.roi_top_ratio)
    roi = frame_bgr[y_top:h, 0:x_right]
    rh, rw = roi.shape[:2]

    annotated = frame_bgr.copy()

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, cfg.canny_lo, cfg.canny_hi)

    # Show exactly what the detector sees: paint the Canny edges into the ROI.
    roi_view = annotated[y_top:h, 0:x_right]
    roi_view[edges > 0] = (255, 0, 255)
    cv2.rectangle(annotated, (0, y_top), (x_right, h), (0, 255, 255), 1)

    min_len = max(10, int(rh * cfg.hough_min_line_frac))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, cfg.hough_threshold,
        minLineLength=min_len, maxLineGap=cfg.hough_max_gap,
    )

    verts = []  # (mid_x_in_roi, angle_from_vertical_deg) for accepted lines
    total = 0 if lines is None else len(lines)
    if lines is not None:
        for line in lines:
            lx1, ly1, lx2, ly2 = (int(v) for v in line[0])
            if ly2 < ly1:  # orient downward for a consistent angle sign
                lx1, ly1, lx2, ly2 = lx2, ly2, lx1, ly1
            dx = float(lx2 - lx1)
            dy = float(ly2 - ly1)
            ang = np.degrees(np.arctan2(dx, dy)) if dy >= 1e-3 else 90.0
            accepted = dy >= 1e-3 and abs(ang) <= cfg.max_vertical_dev_deg
            color = (0, 165, 255) if accepted else (120, 120, 120)
            cv2.line(annotated, (lx1, ly1 + y_top), (lx2, ly2 + y_top),
                     color, 2 if accepted else 1)
            if accepted:
                verts.append(((lx1 + lx2) / 2.0, float(ang)))

    steer = None
    mid_ang = 0.0
    offset_norm = 0.0
    if len(verts) >= 2:
        centre_x = rw / 2.0
        left = [v[0] for v in verts if v[0] < centre_x]
        right = [v[0] for v in verts if v[0] >= centre_x]
        if left and right:
            mid_x = (float(np.mean(left)) + float(np.mean(right))) / 2.0
        else:
            mid_x = float(np.mean([v[0] for v in verts]))
        mid_ang = float(np.mean([v[1] for v in verts]))
        offset_norm = (mid_x - centre_x) / (rw / 2.0)
        raw = cfg.heading_gain * mid_ang + cfg.offset_gain * offset_norm
        steer = int(max(-cfg.steer_limit, min(cfg.steer_limit, raw)))
        # ROI centre reference (grey) vs. the detected midline (green).
        cv2.line(annotated, (int(centre_x), y_top), (int(centre_x), h), (200, 200, 200), 1)
        cv2.line(annotated, (int(mid_x), y_top), (int(mid_x), h), (0, 255, 0), 3)

    _draw_hud(annotated, len(verts), total, mid_ang, offset_norm, steer, cfg.steer_limit)
    return steer, annotated


def _text(img, s, org, color):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def _draw_hud(img, n_vert, n_total, ang, off, steer, limit) -> None:
    ok = steer is not None
    col = (0, 255, 0) if ok else (0, 200, 255)
    _text(img, f"vertical lines: {n_vert} / {n_total} hough", (10, 22), col)
    _text(img, f"angle: {ang:+.1f} deg   offset: {off:+.2f}", (10, 46), col)
    _text(img, (f"GUIDE steer: {steer:+d}" if ok else "GUIDE steer: -- (no 2 lines -> fallback)"),
          (10, 70), col)
    # Steering bar: centre tick + a marker at the guide steer.
    h, w = img.shape[:2]
    bx0, bx1, by = 10, w - 10, h - 16
    cx = (bx0 + bx1) // 2
    cv2.line(img, (bx0, by), (bx1, by), (140, 140, 140), 2)
    cv2.line(img, (cx, by - 9), (cx, by + 9), (210, 210, 210), 1)
    if ok:
        px = int(cx + (steer / float(max(1, limit))) * ((bx1 - bx0) // 2))
        cv2.circle(img, (px, by), 7, (0, 255, 0), -1)
