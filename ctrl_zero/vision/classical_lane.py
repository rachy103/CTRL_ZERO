from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ctrl_zero.common import clamp
from ctrl_zero.vision.base import LaneDetection


@dataclass
class ClassicalLaneConfig:
    resize_width: int = 640
    roi_top_ratio: float = 0.58
    roi_top_half_width_ratio: float = 0.12
    canny_low: int = 50
    canny_high: int = 150
    hough_threshold: int = 35
    min_line_length: int = 35
    max_line_gap: int = 45
    min_abs_slope: float = 0.35
    max_abs_slope: float = 4.5
    default_lane_width_ratio: float = 0.48
    min_lane_width_ratio: float = 0.18
    max_lane_width_ratio: float = 0.90
    fit_smoothing: float = 0.35
    width_smoothing: float = 0.20
    max_missed_frames: int = 8
    curve_lookahead_ratio: float = 0.45
    min_samples_for_quad: int = 6


class ClassicalLaneDetector:
    """OpenCV color/edge lane detector used as a CPU fallback and calibration tool."""

    def __init__(self, config: ClassicalLaneConfig):
        self.config = config
        self.left_fit = None
        self.right_fit = None
        self.left_missed = 0
        self.right_missed = 0
        self.lane_width_px = None

    def detect(self, frame: np.ndarray) -> LaneDetection:
        original_h, original_w = frame.shape[:2]
        if self.config.resize_width > 0 and original_w != self.config.resize_width:
            scale = self.config.resize_width / float(original_w)
            frame = cv2.resize(frame, (self.config.resize_width, int(original_h * scale)))

        h, w = frame.shape[:2]
        y_near = h - 1
        y_far = int(h * self.config.roi_top_ratio)
        raw_mask = self._lane_color_mask(frame)
        edges = cv2.Canny(cv2.GaussianBlur(raw_mask, (5, 5), 0), self.config.canny_low, self.config.canny_high)
        roi_mask, roi_polygon = self._roi(edges)
        lines = cv2.HoughLinesP(
            roi_mask,
            rho=1,
            theta=np.pi / 180,
            threshold=self.config.hough_threshold,
            minLineLength=self.config.min_line_length,
            maxLineGap=self.config.max_line_gap,
        )

        left_samples, right_samples = self._split_line_samples(lines, w)
        left_fit, left_detected = self._smooth_fit("left", self._fit_x_as_function_of_y(left_samples))
        right_fit, right_detected = self._smooth_fit("right", self._fit_x_as_function_of_y(right_samples))

        left_near = self._x_at_y(left_fit, y_near)
        right_near = self._x_at_y(right_fit, y_near)
        left_far = self._x_at_y(left_fit, y_far)
        right_far = self._x_at_y(right_fit, y_far)
        width_px = self._update_lane_width(left_near, right_near, w)

        center_near = None
        center_far = None
        if left_near is not None and right_near is not None:
            center_near = (left_near + right_near) / 2.0
            center_far = (left_far + right_far) / 2.0 if left_far is not None and right_far is not None else None
        elif left_near is not None and width_px is not None:
            center_near = left_near + width_px / 2.0
            center_far = left_far + width_px / 2.0 if left_far is not None else None
        elif right_near is not None and width_px is not None:
            center_near = right_near - width_px / 2.0
            center_far = right_far - width_px / 2.0 if right_far is not None else None

        frame_center = w / 2.0
        offset_px = center_near - frame_center if center_near is not None else None
        offset_norm = offset_px / max(frame_center, 1.0) if offset_px is not None else None
        heading_deg = None
        if center_near is not None and center_far is not None:
            heading_deg = math.degrees(math.atan2(center_far - center_near, max(y_near - y_far, 1)))

        center_fit = self._center_fit(left_fit, right_fit)
        y_eval = y_near - self.config.curve_lookahead_ratio * (y_near - y_far)
        curvature = self._curvature_from_fit(center_fit, y_eval)

        confidence = self._confidence(left_fit, right_fit, left_detected, right_detected)
        annotated = self._annotate(frame, roi_polygon, left_fit, right_fit, center_near, center_far, y_near, y_far)
        lanes = self._fits_to_lanes((left_fit, right_fit), y_near, y_far)

        return LaneDetection(
            lanes=lanes,
            left_fit=left_fit,
            right_fit=right_fit,
            lane_center_near_x=center_near,
            lane_center_far_x=center_far,
            frame_center_x=frame_center,
            offset_px=offset_px,
            offset_norm=offset_norm,
            heading_deg=heading_deg,
            lane_width_px=width_px,
            confidence=confidence,
            mask=roi_mask,
            annotated=annotated,
            curvature=curvature,
        )

    def _lane_color_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        white_hls = cv2.inRange(hls, np.array([0, 150, 0]), np.array([180, 255, 95]))
        yellow_hsv = cv2.inRange(hsv, np.array([15, 70, 70]), np.array([42, 255, 255]))
        mask = cv2.bitwise_or(white_hls, yellow_hsv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    def _roi(self, image: np.ndarray):
        h, w = image.shape[:2]
        top_y = int(h * self.config.roi_top_ratio)
        half_top = int(w * self.config.roi_top_half_width_ratio)
        center = w // 2
        polygon = np.array(
            [[0, h - 1], [w - 1, h - 1], [center + half_top, top_y], [center - half_top, top_y]],
            dtype=np.int32,
        )
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [polygon], 255)
        return cv2.bitwise_and(image, mask), polygon

    def _split_line_samples(self, lines, width: int):
        left_samples = []
        right_samples = []
        if lines is None:
            return left_samples, right_samples

        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue
            slope = dy / float(dx)
            abs_slope = abs(slope)
            if abs_slope < self.config.min_abs_slope or abs_slope > self.config.max_abs_slope:
                continue
            length = math.hypot(dx, dy)
            sample = [(x1, y1, length), (x2, y2, length)]
            mid_x = (x1 + x2) / 2.0
            if slope < 0 and mid_x < width * 0.68:
                left_samples.extend(sample)
            elif slope > 0 and mid_x > width * 0.32:
                right_samples.extend(sample)
        return left_samples, right_samples

    def _fit_x_as_function_of_y(self, samples):
        if len(samples) < 2:
            return None
        xs = np.array([item[0] for item in samples], dtype=np.float64)
        ys = np.array([item[1] for item in samples], dtype=np.float64)
        weights = np.array([max(item[2], 1.0) for item in samples], dtype=np.float64)
        unique_y = len(np.unique(ys))
        if unique_y < 2:
            return None
        use_quad = unique_y >= 3 and len(samples) >= self.config.min_samples_for_quad
        degree = 2 if use_quad else 1
        try:
            fit = np.polyfit(ys, xs, deg=degree, w=weights)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(fit)):
            return None
        if degree == 1:
            fit = np.array([0.0, fit[0], fit[1]], dtype=np.float64)
        return fit

    def _smooth_fit(self, side: str, new_fit):
        previous = self.left_fit if side == "left" else self.right_fit
        missed = self.left_missed if side == "left" else self.right_missed

        detected = new_fit is not None
        if detected:
            alpha = self.config.fit_smoothing
            smoothed = new_fit if previous is None or len(previous) != len(new_fit) else alpha * new_fit + (1.0 - alpha) * previous
            missed = 0
        else:
            missed += 1
            smoothed = previous if missed <= self.config.max_missed_frames else None

        if side == "left":
            self.left_fit = smoothed
            self.left_missed = missed
        else:
            self.right_fit = smoothed
            self.right_missed = missed
        return smoothed, detected

    @staticmethod
    def _x_at_y(fit, y: float) -> float | None:
        if fit is None:
            return None
        return float(np.polyval(fit, y))

    @staticmethod
    def _center_fit(left_fit, right_fit):
        if left_fit is not None and right_fit is not None:
            return (left_fit + right_fit) / 2.0
        if left_fit is not None:
            return left_fit
        if right_fit is not None:
            return right_fit
        return None

    @staticmethod
    def _curvature_from_fit(fit, y_eval: float) -> float:
        if fit is None or len(fit) < 3:
            return 0.0
        a = float(fit[0])
        b = float(fit[1])
        xp = 2.0 * a * y_eval + b
        xpp = 2.0 * a
        denom = (1.0 + xp * xp) ** 1.5
        if denom < 1e-9:
            return 0.0
        kappa = xpp / denom
        return float(kappa) if math.isfinite(kappa) else 0.0

    def _update_lane_width(self, left_x, right_x, width: int):
        if self.lane_width_px is None:
            self.lane_width_px = width * self.config.default_lane_width_ratio
        if left_x is None or right_x is None:
            return self.lane_width_px

        new_width = right_x - left_x
        min_width = width * self.config.min_lane_width_ratio
        max_width = width * self.config.max_lane_width_ratio
        if min_width <= new_width <= max_width:
            alpha = self.config.width_smoothing
            self.lane_width_px = alpha * new_width + (1.0 - alpha) * self.lane_width_px
        return self.lane_width_px

    @staticmethod
    def _confidence(left_fit, right_fit, left_detected: bool, right_detected: bool) -> float:
        confidence = 0.0
        if left_fit is not None:
            confidence += 0.45 if left_detected else 0.25
        if right_fit is not None:
            confidence += 0.45 if right_detected else 0.25
        if left_fit is not None and right_fit is not None:
            confidence += 0.10
        return float(clamp(confidence, 0.0, 1.0))

    def _annotate(self, frame, polygon, left_fit, right_fit, center_near, center_far, y_near, y_far):
        annotated = frame.copy()
        cv2.polylines(annotated, [polygon], isClosed=True, color=(255, 100, 0), thickness=2)
        self._draw_fit(annotated, left_fit, y_near, y_far, (0, 255, 0))
        self._draw_fit(annotated, right_fit, y_near, y_far, (0, 255, 0))
        h, w = annotated.shape[:2]
        frame_center = int(w / 2)
        cv2.line(annotated, (frame_center, h - 1), (frame_center, int(h * 0.55)), (0, 0, 255), 1)
        if center_near is not None and center_far is not None:
            cv2.line(annotated, (int(center_near), y_near), (int(center_far), y_far), (255, 255, 0), 2)
            cv2.circle(annotated, (int(center_near), y_near), 6, (255, 255, 0), -1)
        return annotated

    def _draw_fit(self, image, fit, y_near, y_far, color) -> None:
        if fit is None:
            return
        h, w = image.shape[:2]
        points = []
        for y in np.linspace(y_far, y_near, 16):
            x = self._x_at_y(fit, y)
            if x is not None:
                points.append((int(clamp(x, 0, w - 1)), int(y)))
        if len(points) >= 2:
            cv2.polylines(image, [np.array(points, dtype=np.int32)], isClosed=False, color=color, thickness=3)

    def _fits_to_lanes(self, fits, y_near: int, y_far: int):
        lanes = []
        for fit in fits:
            if fit is None:
                continue
            lanes.append([(int(self._x_at_y(fit, y)), int(y)) for y in np.linspace(y_far, y_near, 12)])
        return lanes
