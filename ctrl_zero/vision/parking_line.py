"""Rear-camera parking-line pre-processing for --mode parking_logic (and the
parking_sim camera guide).

The steering comes from the UNCHANGED lane model (final openvino) run on the
prepared frame — this module only fixes how a parking bay differs from a normal
lane before inference:

1. **Horizontal lines are ERASED** (bay end/entry paint confuses the model).
2. **Broken dashes are MERGED and EXTENDED** — but not one extension per Hough
   segment (that floods the frame: Hough yields several segments per dash and
   several dashes per bay line).  Instead, near-vertical segments are CLUSTERED
   by where their extension crosses the ROI bottom (+ similar angle); each
   cluster = one physical bay line.  Only the ``max_lines`` clusters with the
   largest total segment support are fitted (least squares through all their
   endpoints) and repainted as ONE solid extended line each.
3. The ROI crop: everything above ``roi_top_ratio`` (and right of
   ``keep_left_ratio``) is blacked out.

``prepare_parking_line_frame(frame)`` ->
``(model_frame, debug_vis, n_lines, fitted_lines, h_far_y_ratio)``, where
``h_far_y_ratio`` is the closest erased horizontal line as ``y / frame_height``
(``None`` if none seen) — the rear-camera stop signal when reversing into a bay.

* ``model_frame`` - what to feed the YOLO lane detector.
* ``debug_vis``   - ROI box (yellow), white mask (magenta), accepted dash
                    segments (orange), ERASED horizontals (red), and the final
                    fitted lines (green, thick) — exactly what gets painted.
* ``n_lines``     - number of fitted bay lines drawn (0 -> caller falls back).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ParkingLineConfig:
    roi_top_ratio: float = 0.55      # keep only the bottom (1 - ratio) of the frame
    keep_left_ratio: float = 1.0     # keep only the leftmost this fraction (1.0 = full width)
    white_thresh: int = 180          # gray >= this counts as painted line
    hough_threshold: int = 30
    min_line_frac: float = 0.12      # min dash length as a fraction of ROI height
    max_gap_px: int = 40
    # Segment classification, in degrees from VERTICAL:
    #   <= vertical_max_dev_deg      -> bay side-line dash: keep for clustering
    #   >= horizontal_min_dev_deg    -> horizontal paint: ERASE before inference
    #   in between                   -> ambiguous: leave untouched
    vertical_max_dev_deg: float = 45.0
    horizontal_min_dev_deg: float = 60.0
    # Horizontals below this height ratio are IGNORED for far_y.  The very bottom
    # of the rear view is floor trim / mat edge, not the bay-end line, and it was
    # reporting far_y = 1.00 and firing the stop immediately.  Capping the search
    # here means far_y can never exceed this value.
    horizontal_max_y_ratio: float = 0.9
    # A horizontal counts as the bay-end line only if it spans at least this
    # fraction of the frame width.  The bay-end line crosses the view; faint
    # brightness patches on the grey floor do not, yet a 156 px smudge in the
    # bottom-right corner was reporting far_y = 0.89 and firing the stop.
    horizontal_min_len_frac: float = 0.5
    # Clustering: segments whose bottom-crossing x and angle are this close
    # belong to the same physical bay line.
    merge_x_tol_px: int = 40
    merge_angle_tol_deg: float = 8.0
    # Only the strongest clusters (by total dash length) get painted.  A bay has
    # two side lines -> 2.
    max_lines: int = 2
    # Synthetic frame for the reverse (prepare_infinite_lane_frame): outside the
    # slot bbox is blacked out and the extended lines are painted white, but the
    # REAL road texture is kept inside the bbox.
    # fill_corridor paints road_bgr between the two extended lines so the drivable
    # area is continuous top-to-bottom.  It fills ONLY the black remainder outside
    # the bbox: overwriting the bbox's real road pixels with flat grey as well
    # drops the lane model from a 100% to a 10% fit rate.
    road_bgr: tuple[int, int, int] = (132, 130, 128)  # measured median of the captures
    fill_corridor: bool = True
    # Grain added to the filled grey.  A dead-flat fill is out of the lane model's
    # training distribution and costs a lot of fits; measured over the same 30
    # frames: flat 40%, sigma 6 -> 70%, sigma 12 -> 83%.
    fill_noise_sigma: float = 12.0
    extend_thickness_px: int = 7     # repainted solid line thickness
    erase_thickness_px: int = 15     # blackout stroke over horizontal segments


def prepare_parking_line_frame(frame_bgr, config: ParkingLineConfig | None = None):
    """Erase horizontals; cluster + fit + extend the bay lines. See module doc."""
    cfg = config or ParkingLineConfig()
    if frame_bgr is None:
        return None, None, 0

    h, w = frame_bgr.shape[:2]
    y_top = int(h * cfg.roi_top_ratio)
    x_right = max(1, int(w * cfg.keep_left_ratio))
    roi = frame_bgr[y_top:h, 0:x_right]
    rh, rw = roi.shape[:2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, cfg.white_thresh, 255)

    min_len = max(8, int(rh * cfg.min_line_frac))
    lines = cv2.HoughLinesP(
        mask, 1, np.pi / 180.0, cfg.hough_threshold,
        minLineLength=min_len, maxLineGap=cfg.max_gap_px,
    )

    debug = frame_bgr.copy()
    debug[y_top:h, 0:x_right][mask > 0] = (255, 0, 255)
    cv2.rectangle(debug, (0, y_top), (x_right - 1, h - 1), (0, 255, 255), 1)

    model_frame = frame_bgr.copy()
    model_frame[:y_top] = 0       # "crop": the model only sees the ROI
    model_frame[:, x_right:] = 0

    segs = []      # accepted near-vertical: (x1,y1,x2,y2, ang, length, x_bottom)
    n_erased = 0
    h_far_y_ratio = None  # closest horizontal paint, as y / frame_height
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = (int(v) for v in line[0])
            dx, dy = float(x2 - x1), float(y2 - y1)
            ang = abs(np.degrees(np.arctan2(dx, dy))) if (dx or dy) else 90.0
            ang = min(ang, 180.0 - ang)

            if ang >= cfg.horizontal_min_dev_deg:
                n_erased += 1
                cv2.line(model_frame, (x1, y1 + y_top), (x2, y2 + y_top),
                         (0, 0, 0), cfg.erase_thickness_px)
                cv2.line(debug, (x1, y1 + y_top), (x2, y2 + y_top), (0, 0, 255), 2)
                # Track the CLOSEST horizontal paint (largest y in the full frame),
                # ignoring anything below horizontal_max_y_ratio: the very bottom
                # of the rear view is floor trim, not the bay-end line.
                seg_far_y = (max(y1, y2) + y_top) / float(h)
                if seg_far_y <= cfg.horizontal_max_y_ratio and (
                    h_far_y_ratio is None or seg_far_y > h_far_y_ratio
                ):
                    h_far_y_ratio = seg_far_y
                continue
            if ang > cfg.vertical_max_dev_deg:
                continue

            length = float(np.hypot(dx, dy))
            # x where this segment's own line crosses the ROI bottom (y = rh-1):
            # the clustering key that groups dashes of the same bay line.
            if abs(dy) > 1e-6:
                x_bottom = x1 + (rh - 1.0 - y1) * dx / dy
            else:
                x_bottom = (x1 + x2) / 2.0
            # Signed angle for clustering (left-lean vs right-lean differ).
            signed_ang = float(np.degrees(np.arctan2(dx, dy)))
            segs.append((x1, y1, x2, y2, signed_ang, length, x_bottom))
            cv2.line(debug, (x1, y1 + y_top), (x2, y2 + y_top), (0, 165, 255), 2)

    # --- cluster the accepted segments into physical bay lines ---------------
    clusters: list[dict] = []  # {'x_bottom': mean, 'ang': mean, 'support': sum_len, 'segs': [...]}
    for seg in sorted(segs, key=lambda s: -s[5]):  # longest first seeds clusters
        placed = False
        for cl in clusters:
            if (
                abs(seg[6] - cl["x_bottom"]) <= cfg.merge_x_tol_px
                and abs(seg[4] - cl["ang"]) <= cfg.merge_angle_tol_deg
            ):
                n = len(cl["segs"])
                cl["x_bottom"] = (cl["x_bottom"] * n + seg[6]) / (n + 1)
                cl["ang"] = (cl["ang"] * n + seg[4]) / (n + 1)
                cl["support"] += seg[5]
                cl["segs"].append(seg)
                placed = True
                break
        if not placed:
            clusters.append(
                {"x_bottom": seg[6], "ang": seg[4], "support": seg[5], "segs": [seg]}
            )

    # Strongest clusters only: one painted line per physical bay line.
    clusters.sort(key=lambda c: -c["support"])
    n_lines = 0
    fitted_lines = []  # ((x1,y1),(x2,y2)) per painted line, in FULL-frame coords
    for cl in clusters[: max(0, cfg.max_lines)]:
        pts = []
        for (sx1, sy1, sx2, sy2, _a, _l, _xb) in cl["segs"]:
            pts.append((sx1, sy1))
            pts.append((sx2, sy2))
        arr = np.array(pts, dtype=np.float32)
        vx, vy, cx, cy = (float(v) for v in cv2.fitLine(arr, cv2.DIST_L2, 0, 0.01, 0.01).ravel())
        (ex1, ey1), (ex2, ey2) = _extend_to_rect(
            cx, cy, cx + vx * 50.0, cy + vy * 50.0, rw, rh
        )
        cv2.line(model_frame, (ex1, ey1 + y_top), (ex2, ey2 + y_top),
                 (255, 255, 255), cfg.extend_thickness_px)
        cv2.line(debug, (ex1, ey1 + y_top), (ex2, ey2 + y_top), (0, 255, 0), 2)
        fitted_lines.append(((ex1, ey1 + y_top), (ex2, ey2 + y_top)))
        n_lines += 1

    model_frame[:y_top] = 0       # re-black: thick strokes bleed past the ROI edges
    model_frame[:, x_right:] = 0
    _text(
        debug,
        f"dashes:{len(segs)} clusters:{len(clusters)} drawn:{n_lines} erasedH:{n_erased}"
        + ("" if h_far_y_ratio is None else f" hFarY:{h_far_y_ratio:.2f}"),
        (10, 24),
    )
    if h_far_y_ratio is not None:
        hy = int(h_far_y_ratio * h)
        cv2.line(debug, (0, hy), (w - 1, hy), (0, 0, 255), 1)
    return model_frame, debug, n_lines, fitted_lines, h_far_y_ratio


def vanishing_point(fitted_lines):
    """Image-space intersection of the first two fitted lines, or ``None``.

    The straight-drive invariant: when the car is PARALLEL to two (physically
    parallel) painted lines, their image intersection — the vanishing point —
    sits exactly at the camera's optical centre x, regardless of the car's
    lateral position.  Steering on (vp_x - cx) therefore holds heading with a
    single gain and zero calibration.
    """
    if fitted_lines is None or len(fitted_lines) < 2:
        return None
    (x1, y1), (x2, y2) = fitted_lines[0]
    (x3, y3), (x4, y4) = fitted_lines[1]
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-6:
        return None  # parallel in the image: already aligned (or degenerate)
    a = x1 * y2 - y1 * x2
    b = x3 * y4 - y3 * x4
    px = (a * (x3 - x4) - (x1 - x2) * b) / d
    py = (a * (y3 - y4) - (y1 - y2) * b) / d
    return float(px), float(py)


def _scan_far_y(frame_bgr, cfg: ParkingLineConfig, debug=None):
    """Nearest white horizontal in the WHOLE frame, as ``y / frame_height``.

    Deliberately independent of the slot bbox: the bay-end line the car is
    reversing toward is a fact about the surroundings, not about the bbox, and
    it often lies below the bbox.  Segments beyond ``horizontal_max_y_ratio``
    (the very bottom of the rear view = floor trim, not bay paint) are ignored,
    so the returned value can never exceed that cap.
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    mask = cv2.inRange(gray, cfg.white_thresh, 255)
    lines = cv2.HoughLinesP(
        mask, 1, np.pi / 180.0, cfg.hough_threshold,
        minLineLength=max(8, int(h * cfg.min_line_frac)), maxLineGap=cfg.max_gap_px,
    )
    if lines is None:
        return None
    best = None
    for ln in lines:
        x1, y1, x2, y2 = (int(v) for v in ln[0])
        dx, dy = float(x2 - x1), float(y2 - y1)
        ang = abs(np.degrees(np.arctan2(dx, dy))) if (dx or dy) else 90.0
        ang = min(ang, 180.0 - ang)
        if ang < cfg.horizontal_min_dev_deg:
            continue
        if abs(dx) < cfg.horizontal_min_len_frac * w:
            continue  # too short to be the bay-end line
        far_y = max(y1, y2) / float(h)
        if far_y > cfg.horizontal_max_y_ratio:
            continue
        if best is None or far_y > best:
            best = far_y
        if debug is not None:
            cv2.line(debug, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return best


def prepare_infinite_lane_frame(frame_bgr, bbox=None, config: ParkingLineConfig | None = None):
    """Build an "infinite lane" frame for the rear reverse, then let the lane model drive.

    Why: deep in the bay the horizontal bay-end line covers the drivable area.
    The lane model then has almost no vertical evidence left and forces a fit out
    of whatever remains, so the heading swings violently right at the end — the
    worst possible moment.

    The synthetic frame handed to the model is therefore rebuilt from scratch:

    * everything outside the slot bbox -> BLACK,
    * the bay's side lines, found while they are still visible, extended to span
      the FULL frame height — a lane that never ends,
    * the corridor BETWEEN those lines filled with the road colour, so the
      drivable area is explicit and continuous (a pure black canvas makes the
      lane model return nothing at all — measured conf 0.00 on every frame),
    * the white lines painted on top.

    Horizontal bay paint is never drawn, so it cannot hijack the fit; it is only
    MEASURED on the raw frame and reported as ``h_far_y_ratio`` for the
    far_y >= 0.9 stop.

    ``bbox`` (x1,y1,x2,y2) restricts WHERE lines are searched; the drawn lane
    still spans the whole frame.

    Returns ``(model_frame, debug_vis, n_lines, h_far_y_ratio)``.
    """
    cfg = config or ParkingLineConfig()
    if frame_bgr is None:
        return None, None, 0, None

    h, w = frame_shape = frame_bgr.shape[:2]
    x0, y0, x1b, y1b = 0, int(h * cfg.roi_top_ratio), w, h
    if bbox is not None:
        bx1, by1, bx2, by2 = (int(v) for v in bbox)
        x0, x1b = max(0, bx1), min(w, bx2)
        y0, y1b = max(0, by1), min(h, by2)
    if x1b - x0 < 5 or y1b - y0 < 5:
        return frame_bgr.copy(), frame_bgr.copy(), 0, None

    roi = frame_bgr[y0:y1b, x0:x1b]
    rh, rw = roi.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    mask = cv2.inRange(gray, cfg.white_thresh, 255)

    debug = frame_bgr.copy()
    debug[y0:y1b, x0:x1b][mask > 0] = (255, 0, 255)
    cv2.rectangle(debug, (x0, y0), (x1b - 1, y1b - 1), (0, 255, 255), 1)

    lines = cv2.HoughLinesP(
        mask, 1, np.pi / 180.0, cfg.hough_threshold,
        minLineLength=max(8, int(rh * cfg.min_line_frac)), maxLineGap=cfg.max_gap_px,
    )

    # far_y is a STOP signal about the car's real surroundings, so it is measured
    # over the WHOLE frame, not just the slot bbox: the approaching bay-end line
    # frequently sits below the bbox (29% of detected frames here), and cropping
    # the search to the bbox made it invisible right when it matters.
    h_far_y_ratio = _scan_far_y(frame_bgr, cfg, debug)

    # Black outside the slot bbox; the real road texture stays inside it (the
    # lane model needs real pixels — see ParkingLineConfig.fill_corridor).
    model_frame = np.zeros_like(frame_bgr)
    model_frame[y0:y1b, x0:x1b] = frame_bgr[y0:y1b, x0:x1b]

    segs = []
    if lines is not None:
        for ln in lines:
            sx1, sy1, sx2, sy2 = (int(v) for v in ln[0])
            dx, dy = float(sx2 - sx1), float(sy2 - sy1)
            ang = abs(np.degrees(np.arctan2(dx, dy))) if (dx or dy) else 90.0
            ang = min(ang, 180.0 - ang)
            if ang >= cfg.horizontal_min_dev_deg:
                # ERASED from the model's copy so it cannot hijack the fit once it
                # covers the drivable area.  far_y is NOT taken here — it is
                # measured over the whole frame by _scan_far_y above.
                cv2.line(model_frame, (sx1 + x0, sy1 + y0), (sx2 + x0, sy2 + y0),
                         (0, 0, 0), cfg.erase_thickness_px)
                continue
            if ang > cfg.vertical_max_dev_deg:
                continue
            length = float(np.hypot(dx, dy))
            x_bottom = (sx1 + (rh - 1.0 - sy1) * dx / dy) if abs(dy) > 1e-6 else (sx1 + sx2) / 2.0
            segs.append((sx1, sy1, sx2, sy2, float(np.degrees(np.arctan2(dx, dy))), length, x_bottom))
            cv2.line(debug, (sx1 + x0, sy1 + y0), (sx2 + x0, sy2 + y0), (0, 165, 255), 2)

    # Cluster dashes belonging to the same physical side line.
    clusters: list[dict] = []
    for seg in sorted(segs, key=lambda s: -s[5]):
        for cl in clusters:
            if (abs(seg[6] - cl["x_bottom"]) <= cfg.merge_x_tol_px
                    and abs(seg[4] - cl["ang"]) <= cfg.merge_angle_tol_deg):
                n = len(cl["segs"])
                cl["x_bottom"] = (cl["x_bottom"] * n + seg[6]) / (n + 1)
                cl["ang"] = (cl["ang"] * n + seg[4]) / (n + 1)
                cl["support"] += seg[5]
                cl["segs"].append(seg)
                break
        else:
            clusters.append({"x_bottom": seg[6], "ang": seg[4], "support": seg[5], "segs": [seg]})
    clusters.sort(key=lambda c: -c["support"])

    # Fit each side line and extend it to span the WHOLE frame height.
    spans = []  # (x_at_top, x_at_bottom)
    for cl in clusters[: max(0, cfg.max_lines)]:
        pts = []
        for (ax, ay, bx, by, _a, _l, _xb) in cl["segs"]:
            pts.append((ax + x0, ay + y0))
            pts.append((bx + x0, by + y0))
        vx, vy, cx, cy = (float(v) for v in
                          cv2.fitLine(np.array(pts, np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel())
        if abs(vy) < 1e-6:
            continue
        # Intersect with y = 0 and y = h-1 -> a line with no start and no end.
        spans.append((cx + vx * (0.0 - cy) / vy, cx + vx * ((h - 1.0) - cy) / vy))
    n_lines = len(spans)

    # Fill the corridor BETWEEN the two lines with the road colour so the drivable
    # area runs unbroken from top to bottom — but only over pixels that are still
    # black (i.e. outside the bbox).  Painting flat grey over the bbox's real road
    # pixels as well drops the lane model to a 10% fit rate; keeping them and only
    # filling the black remainder gives a continuous corridor the model still reads.
    if cfg.fill_corridor and n_lines >= 2:
        spans.sort(key=lambda s: s[0] + s[1])
        (lt, lb), (rt, rb) = spans[0], spans[-1]
        poly = np.array([[lt, 0], [rt, 0], [rb, h - 1], [lb, h - 1]], dtype=np.int32)
        corridor = np.zeros(model_frame.shape[:2], np.uint8)
        cv2.fillPoly(corridor, [poly], 255)
        model_frame[(corridor > 0) & (model_frame.max(axis=2) == 0)] = cfg.road_bgr
        # Grain over the WHOLE corridor — the filled grey and the real road pixels
        # alike.  Graining only the filled part leaves a visible seam between the
        # two and the model still balks (43% fit); graining everything inside the
        # corridor restores it to 83%.
        if cfg.fill_noise_sigma > 0.0:
            rng = np.random.default_rng(0)  # fixed seed: same frame -> same grain
            region = (corridor > 0) & (model_frame.max(axis=2) > 0) & (model_frame.min(axis=2) <= 200)
            if region.any():
                noisy = np.clip(
                    model_frame.astype(np.int16)
                    + rng.normal(0.0, cfg.fill_noise_sigma, model_frame.shape).astype(np.int16),
                    0, 255,
                ).astype(np.uint8)
                model_frame[region] = noisy[region]
    for (xt, xb) in spans:
        cv2.line(model_frame, (int(round(xt)), 0), (int(round(xb)), h - 1),
                 (255, 255, 255), cfg.extend_thickness_px)
        cv2.line(debug, (int(round(xt)), 0), (int(round(xb)), h - 1), (0, 255, 0), 2)

    _text(debug, f"dashes:{len(segs)} lanes:{n_lines}"
                 + ("" if h_far_y_ratio is None else f" hFarY:{h_far_y_ratio:.2f}"), (10, 24))
    if h_far_y_ratio is not None:
        hy = int(h_far_y_ratio * h)
        cv2.line(debug, (0, hy), (w - 1, hy), (0, 0, 255), 1)
    return model_frame, debug, n_lines, h_far_y_ratio


def _extend_to_rect(x1, y1, x2, y2, rw, rh):
    """Extend the line through (x1,y1)-(x2,y2) to the borders of a rw x rh rect."""
    dx, dy = float(x2 - x1), float(y2 - y1)
    pts = []
    if abs(dx) > 1e-6:
        for bx in (0.0, rw - 1.0):
            t = (bx - x1) / dx
            by = y1 + t * dy
            if -1.0 <= by <= rh:
                pts.append((bx, min(max(by, 0.0), rh - 1.0)))
    if abs(dy) > 1e-6:
        for by in (0.0, rh - 1.0):
            t = (by - y1) / dy
            bx = x1 + t * dx
            if -1.0 <= bx <= rw:
                pts.append((min(max(bx, 0.0), rw - 1.0), by))
    if len(pts) < 2:
        return (int(x1), int(y1)), (int(x2), int(y2))
    best, best_d = (pts[0], pts[1]), -1.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2
            if d > best_d:
                best_d = d
                best = (pts[i], pts[j])
    (ax, ay), (bx, by) = best
    return (int(round(ax)), int(round(ay))), (int(round(bx)), int(round(by)))


def _text(img, s, org):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
