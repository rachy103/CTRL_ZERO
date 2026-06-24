from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from ctrl_zero.common import clamp
from ctrl_zero.vision.base import LaneDetection, Point
from ctrl_zero.vision.preprocess import LanePreprocessor


@dataclass
class YOLOLaneConfig:
    model_path: Path
    device: str = "cpu"
    image_size: int = 640
    confidence: float = 0.25
    iou: float = 0.45
    class_names: tuple[str, ...] = ("lane", "left_lane", "right_lane", "center_lane", "dashed_lane", "solid_lane", "road_line", "line")
    control_near_y_ratio: float = 0.95
    control_far_y_ratio: float = 0.68
    curve_lookahead_ratio: float = 0.45
    mask_sample_step_px: int = 6
    min_points_per_lane: int = 3
    min_valid_y_span_ratio: float = 0.08
    skeleton_enabled: bool = True
    skeleton_bridge_gap_px: int = 17
    dashed_merge_max_x_gap_ratio: float = 0.12
    curve_fit_degree: int = 2
    solidify_step_px: int = 6
    fit_outlier_rejection_px: float = 35.0
    default_lane_width_ratio: float = 0.50
    min_lane_width_ratio: float = 0.35
    max_lane_width_ratio: float = 0.65
    lane_width_smoothing: float = 0.18
    center_smoothing: float = 0.35
    max_missed_frames: int = 12
    preprocessor: LanePreprocessor | None = None


@dataclass
class LaneFragment:
    points: list[Point]
    fit: np.ndarray
    min_y: float
    max_y: float
    median_y: float


class YOLOLaneDetector:
    """Lane detector for Ultralytics-compatible YOLO lane models.

    A generic COCO YOLO model is not enough because COCO does not include lane
    markings. Use a lane-trained detection or segmentation model when driving.
    """

    def __init__(self, config: YOLOLaneConfig):
        self.config = config
        self.model = self._load_model(config.model_path)
        self.class_names = tuple(name.lower() for name in config.class_names)
        self.lane_width_px = None
        self.center_near_x = None
        self.center_far_x = None
        self.missed_frames = 0

    def detect(self, frame: np.ndarray) -> LaneDetection:
        h, w = frame.shape[:2]
        near_y = int(h * self.config.control_near_y_ratio)
        far_y = int(h * self.config.control_far_y_ratio)
        frame_center_x = w / 2.0

        model_frame = frame
        transform = None
        if self.config.preprocessor is not None:
            model_frame, transform = self.config.preprocessor.apply(frame)
        model_h, model_w = model_frame.shape[:2]

        results = self.model.predict(
            source=model_frame,
            imgsz=self.config.image_size,
            conf=self.config.confidence,
            iou=self.config.iou,
            device=self.config.device,
            verbose=False,
            retina_masks=True,
        )
        result = results[0]
        lane_fragments = self._extract_lanes(result, model_h, model_w)
        if transform is not None:
            lane_fragments = [transform.points_to_original(lane) for lane in lane_fragments]

        fits = self._fit_lane_groups(lane_fragments, h, w, near_y, frame_center_x)
        if not fits:
            fits = [self._fit_lane_x_of_y(lane, h) for lane in lane_fragments]
            fits = [fit for fit in fits if fit is not None]
        left_fit, right_fit = self._select_left_right_fits(fits, frame_center_x, near_y)
        lanes = self._solidified_lanes_from_fits((left_fit, right_fit), h, w, near_y, far_y)

        left_near = self._x_at_y(left_fit, near_y)
        right_near = self._x_at_y(right_fit, near_y)
        left_far = self._x_at_y(left_fit, far_y)
        right_far = self._x_at_y(right_fit, far_y)
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

        confidence = self._confidence(left_fit, right_fit, len(lane_fragments), result)
        if center_near is None:
            self.missed_frames += 1
            if self.missed_frames <= self.config.max_missed_frames and self.center_near_x is not None:
                center_near = self.center_near_x
                center_far = self.center_far_x
                confidence = min(confidence, 0.25)
        else:
            self.missed_frames = 0
            center_near = self._smooth("near", center_near)
            if center_far is not None:
                center_far = self._smooth("far", center_far)

        offset_px = center_near - frame_center_x if center_near is not None else None
        offset_norm = offset_px / max(frame_center_x, 1.0) if offset_px is not None else None
        heading_deg = None
        if center_near is not None and center_far is not None:
            heading_deg = math.degrees(math.atan2(center_far - center_near, max(near_y - far_y, 1)))

        center_fit = self._center_fit(left_fit, right_fit)
        y_eval = near_y - self.config.curve_lookahead_ratio * (near_y - far_y)
        curvature = self._curvature_from_fit(center_fit, y_eval)

        model_mask = self._combined_mask(result, model_h, model_w)
        mask = transform.mask_to_original(model_mask) if transform is not None else model_mask
        annotated = self._annotate(
            frame,
            lanes,
            lane_fragments,
            left_fit,
            right_fit,
            center_near,
            center_far,
            near_y,
            far_y,
            confidence,
            transform=transform,
        )
        return LaneDetection(
            lanes=lanes,
            left_fit=left_fit,
            right_fit=right_fit,
            lane_center_near_x=center_near,
            lane_center_far_x=center_far,
            frame_center_x=frame_center_x,
            offset_px=offset_px,
            offset_norm=offset_norm,
            heading_deg=heading_deg,
            lane_width_px=width_px,
            confidence=confidence,
            mask=mask,
            annotated=annotated,
            curvature=curvature,
        )

    @staticmethod
    def _load_model(model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO lane model not found: {model_path}\n"
                "Pass a lane-trained Ultralytics model path, such as .pt, .onnx, or an OpenVINO export directory.\n"
                "A generic COCO YOLO model will not detect lane markings."
            )
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise RuntimeError("ultralytics is required for YOLO backend. Run: python -m pip install -r requirements.txt") from exc
        return YOLO(str(model_path))

    def _extract_lanes(self, result, height: int, width: int) -> list[list[Point]]:
        lanes = self._extract_mask_lanes(result, height, width)
        if lanes:
            return lanes
        return self._extract_box_lanes(result)

    def _extract_mask_lanes(self, result, height: int, width: int) -> list[list[Point]]:
        if result.masks is None:
            return []
        masks = result.masks.data.detach().cpu().numpy()
        boxes = result.boxes
        lanes = []
        for index, mask in enumerate(masks):
            if boxes is not None and not self._class_allowed(int(boxes.cls[index].item()), result.names):
                continue
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
            points = self._centerline_from_mask(mask > 0.5)
            if len(points) >= self.config.min_points_per_lane:
                lanes.append(points)
        return lanes

    def _centerline_from_mask(self, mask: np.ndarray) -> list[Point]:
        center_mask = self._skeletonize_lane_mask(mask) if self.config.skeleton_enabled else mask
        ys, xs = np.where(center_mask)
        if len(xs) == 0:
            return []
        points = []
        step = max(1, self.config.mask_sample_step_px)
        for y in range(int(ys.min()), int(ys.max()) + 1, step):
            row_x = xs[ys == y]
            if len(row_x) == 0:
                continue
            points.append((int(np.median(row_x)), int(y)))
        return points

    def _skeletonize_lane_mask(self, mask: np.ndarray) -> np.ndarray:
        binary = (mask.astype(np.uint8) * 255)
        if self.config.skeleton_bridge_gap_px > 1:
            gap = max(3, int(self.config.skeleton_bridge_gap_px))
            if gap % 2 == 0:
                gap += 1
            bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, gap))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, bridge_kernel)

        skeleton = np.zeros(binary.shape, dtype=np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        working = binary.copy()
        max_iterations = max(binary.shape)
        for _ in range(max_iterations):
            eroded = cv2.erode(working, element)
            opened = cv2.dilate(eroded, element)
            skeleton = cv2.bitwise_or(skeleton, cv2.subtract(working, opened))
            working = eroded
            if cv2.countNonZero(working) == 0:
                break
        return skeleton > 0

    def _extract_box_lanes(self, result) -> list[list[Point]]:
        if result.boxes is None:
            return []
        lanes = []
        boxes = result.boxes
        xyxy = boxes.xyxy.detach().cpu().numpy()
        for index, box in enumerate(xyxy):
            if not self._class_allowed(int(boxes.cls[index].item()), result.names):
                continue
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2.0)
            lanes.append([(center_x, int(y1)), (center_x, int((y1 + y2) / 2.0)), (center_x, int(y2))])
        return lanes

    def _class_allowed(self, class_id: int, names) -> bool:
        if not self.class_names or "*" in self.class_names:
            return True
        if isinstance(names, dict):
            class_name = str(names.get(class_id, class_id)).lower()
        elif 0 <= class_id < len(names):
            class_name = str(names[class_id]).lower()
        else:
            class_name = str(class_id)
        return any(token in class_name for token in self.class_names)

    def _fit_lane_groups(self, lane_fragments: Sequence[Sequence[Point]], image_height: int, image_width: int, near_y: int, frame_center_x: float):
        fragments = self._build_lane_fragments(lane_fragments)
        if not fragments:
            return []

        groups: list[list[Point]] = []
        max_gap = max(18.0, image_width * self.config.dashed_merge_max_x_gap_ratio)
        for fragment in sorted(fragments, key=lambda item: self._x_at_y(item.fit, near_y) or frame_center_x):
            best_index = None
            best_distance = float("inf")
            fragment_x = self._x_at_y(fragment.fit, fragment.median_y)
            if fragment_x is None:
                continue

            for index, group_points in enumerate(groups):
                group_fit = self._fit_lane_x_of_y(group_points, image_height, min_y_span_ratio=0.0, degree=1)
                if group_fit is None:
                    continue
                group_x = self._x_at_y(group_fit, fragment.median_y)
                if group_x is None:
                    continue
                if (group_x < frame_center_x) != (fragment_x < frame_center_x):
                    continue
                distance = abs(group_x - fragment_x)
                if distance < best_distance:
                    best_distance = distance
                    best_index = index

            if best_index is None or best_distance > max_gap:
                groups.append(list(fragment.points))
            else:
                groups[best_index].extend(fragment.points)

        fits = []
        for group_points in groups:
            fit = self._fit_lane_x_of_y(group_points, image_height)
            if fit is not None:
                fits.append(fit)
        return fits

    def _build_lane_fragments(self, lane_fragments: Sequence[Sequence[Point]]) -> list[LaneFragment]:
        fragments: list[LaneFragment] = []
        for points in lane_fragments:
            fit = self._fit_lane_x_of_y(points, image_height=1, min_y_span_ratio=0.0, degree=1)
            if fit is None:
                continue
            ys = np.array([p[1] for p in points], dtype=np.float64)
            fragments.append(
                LaneFragment(
                    points=list(points),
                    fit=fit,
                    min_y=float(ys.min()),
                    max_y=float(ys.max()),
                    median_y=float(np.median(ys)),
                )
            )
        return fragments

    def _fit_lane_x_of_y(
        self,
        points: Sequence[Point],
        image_height: int,
        min_y_span_ratio: float | None = None,
        degree: int | None = None,
    ):
        if len(points) < self.config.min_points_per_lane:
            return None
        xs = np.array([p[0] for p in points], dtype=np.float64)
        ys = np.array([p[1] for p in points], dtype=np.float64)
        min_span = self.config.min_valid_y_span_ratio if min_y_span_ratio is None else min_y_span_ratio
        if image_height > 1 and ys.max() - ys.min() < image_height * min_span:
            return None
        unique_y = np.unique(ys)
        fit_degree = min(self.config.curve_fit_degree if degree is None else degree, len(unique_y) - 1)
        if fit_degree < 1:
            return None
        try:
            fit = np.polyfit(ys, xs, deg=fit_degree)
            if self.config.fit_outlier_rejection_px > 0 and len(xs) > fit_degree + 3:
                residual = np.abs(np.polyval(fit, ys) - xs)
                keep = residual <= self.config.fit_outlier_rejection_px
                if int(np.count_nonzero(keep)) >= max(self.config.min_points_per_lane, fit_degree + 1):
                    fit = np.polyfit(ys[keep], xs[keep], deg=fit_degree)
            return fit
        except np.linalg.LinAlgError:
            return None

    @staticmethod
    def _x_at_y(fit, y: float) -> float | None:
        if fit is None:
            return None
        return float(np.polyval(fit, y))

    @staticmethod
    def _center_fit(left_fit, right_fit):
        if left_fit is None and right_fit is None:
            return None
        if left_fit is None:
            return np.array(right_fit, dtype=np.float64)
        if right_fit is None:
            return np.array(left_fit, dtype=np.float64)

        left = np.array(left_fit, dtype=np.float64)
        right = np.array(right_fit, dtype=np.float64)
        max_len = max(len(left), len(right))
        left = np.pad(left, (max_len - len(left), 0))
        right = np.pad(right, (max_len - len(right), 0))
        return (left + right) / 2.0

    @staticmethod
    def _curvature_from_fit(fit, y_eval: float) -> float:
        if fit is None or len(fit) < 3:
            return 0.0
        first = np.polyder(fit, 1)
        second = np.polyder(fit, 2)
        xp = float(np.polyval(first, y_eval))
        xpp = float(np.polyval(second, y_eval))
        denom = (1.0 + xp * xp) ** 1.5
        if denom < 1e-9:
            return 0.0
        kappa = xpp / denom
        return float(kappa) if math.isfinite(kappa) else 0.0

    def _select_left_right_fits(self, fits, frame_center_x: float, near_y: int):
        left_candidates = []
        right_candidates = []
        for fit in fits:
            x_near = self._x_at_y(fit, near_y)
            if x_near is None:
                continue
            if x_near < frame_center_x:
                left_candidates.append((abs(frame_center_x - x_near), fit))
            else:
                right_candidates.append((abs(x_near - frame_center_x), fit))
        left_fit = min(left_candidates, default=(None, None), key=lambda item: item[0])[1]
        right_fit = min(right_candidates, default=(None, None), key=lambda item: item[0])[1]
        return left_fit, right_fit

    def _update_lane_width(self, left_x, right_x, image_width: int):
        if self.lane_width_px is None:
            self.lane_width_px = image_width * self.config.default_lane_width_ratio
        if left_x is None or right_x is None:
            return self.lane_width_px
        new_width = right_x - left_x
        if image_width * self.config.min_lane_width_ratio <= new_width <= image_width * self.config.max_lane_width_ratio:
            alpha = self.config.lane_width_smoothing
            self.lane_width_px = alpha * new_width + (1.0 - alpha) * self.lane_width_px
        return self.lane_width_px

    def _smooth(self, which: str, value: float) -> float:
        if which == "near":
            previous = self.center_near_x
            smoothed = value if previous is None else self.config.center_smoothing * value + (1.0 - self.config.center_smoothing) * previous
            self.center_near_x = smoothed
            return smoothed
        previous = self.center_far_x
        smoothed = value if previous is None else self.config.center_smoothing * value + (1.0 - self.config.center_smoothing) * previous
        self.center_far_x = smoothed
        return smoothed

    @staticmethod
    def _confidence(left_fit, right_fit, lane_count: int, result) -> float:
        confidence = 0.0
        if left_fit is not None:
            confidence += 0.42
        if right_fit is not None:
            confidence += 0.42
        confidence += min(lane_count, 4) * 0.04
        if result.boxes is not None and len(result.boxes) > 0 and result.boxes.conf is not None:
            box_conf = result.boxes.conf.detach().cpu().numpy()
            confidence = max(confidence, float(np.clip(np.mean(box_conf), 0.0, 1.0)))
        return float(clamp(confidence, 0.0, 1.0))

    def _combined_mask(self, result, height: int, width: int) -> np.ndarray | None:
        if result.masks is None:
            return None
        masks = result.masks.data.detach().cpu().numpy()
        if len(masks) == 0:
            return None
        combined = np.zeros((height, width), dtype=np.uint8)
        for mask in masks:
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
            combined[mask > 0.5] = 255
        return combined

    def _solidified_lanes_from_fits(self, fits, image_height: int, image_width: int, near_y: int, far_y: int) -> list[list[Point]]:
        lanes = []
        step = max(1, int(self.config.solidify_step_px))
        for fit in fits:
            if fit is None:
                continue
            lane = []
            for y in range(int(far_y), int(near_y) + 1, step):
                x = int(clamp(self._x_at_y(fit, y), 0, image_width - 1))
                lane.append((x, y))
            if lane and lane[-1][1] != near_y:
                x = int(clamp(self._x_at_y(fit, near_y), 0, image_width - 1))
                lane.append((x, near_y))
            lanes.append(lane)
        return lanes

    def _annotate(self, frame, lanes, lane_fragments, left_fit, right_fit, center_near, center_far, near_y, far_y, confidence, transform=None):
        vis = frame.copy()
        if transform is not None:
            transform.draw_overlay(vis)
        for lane in lane_fragments:
            for point in lane:
                cv2.circle(vis, point, 1, (0, 255, 255), -1)
        for lane in lanes:
            if len(lane) >= 2:
                cv2.polylines(vis, [np.array(lane, dtype=np.int32)], isClosed=False, color=(255, 255, 0), thickness=2)
        self._draw_fit(vis, left_fit, near_y, far_y, (0, 255, 0))
        self._draw_fit(vis, right_fit, near_y, far_y, (0, 255, 0))
        h, w = vis.shape[:2]
        cv2.line(vis, (w // 2, h - 1), (w // 2, int(h * 0.55)), (0, 0, 255), 1)
        if center_near is not None and center_far is not None:
            cv2.line(vis, (int(center_near), near_y), (int(center_far), far_y), (255, 255, 0), 2)
            cv2.circle(vis, (int(center_near), near_y), 6, (255, 255, 0), -1)
        cv2.putText(vis, f"yolo conf={confidence:.2f}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(vis, f"yolo conf={confidence:.2f}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        return vis

    def _draw_fit(self, image, fit, near_y, far_y, color) -> None:
        if fit is None:
            return
        h, w = image.shape[:2]
        x_near = int(clamp(self._x_at_y(fit, near_y), 0, w - 1))
        x_far = int(clamp(self._x_at_y(fit, far_y), 0, w - 1))
        points = []
        step = max(1, int(self.config.solidify_step_px))
        for y in range(int(far_y), int(near_y) + 1, step):
            x = int(clamp(self._x_at_y(fit, y), 0, w - 1))
            points.append((x, y))
        if len(points) >= 2:
            cv2.polylines(image, [np.array(points, dtype=np.int32)], isClosed=False, color=color, thickness=3)
        else:
            cv2.line(image, (x_near, near_y), (x_far, far_y), color, 3)
