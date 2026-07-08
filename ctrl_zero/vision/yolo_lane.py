from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from ctrl_zero.common import clamp
from ctrl_zero.perception import BoundingBox, DetectedObject
from ctrl_zero.traffic_light import traffic_light_state_from_objects
from ctrl_zero.vision.base import LaneDetection, LaneReference, Point
from ctrl_zero.vision.preprocess import LanePreprocessor


@dataclass
class YOLOLaneConfig:
    model_path: Path
    device: str = "cpu"
    image_size: int = 640
    confidence: float = 0.25
    iou: float = 0.45
    class_names: tuple[str, ...] = ("car", "obstacle", "lane1", "lane2", "traffic_light")
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
    segmentation_mode: str = "lane_area"
    drivable_min_row_width_ratio: float = 0.05
    drivable_edge_percentile: float = 2.0
    target_lane_pair: str = "right"
    target_path_mode: str = "closest_line"
    lane_pair_select_y_ratio: float = 0.78
    lane_pair_target_offset_ratio: float = 0.0
    default_lane_width_ratio: float = 0.50
    min_lane_width_ratio: float = 0.35
    max_lane_width_ratio: float = 0.65
    lane_width_smoothing: float = 0.18
    center_smoothing: float = 0.35
    max_missed_frames: int = 12
    display_bird_eye_view: bool = True
    preprocessor: LanePreprocessor | None = None


@dataclass
class LaneFragment:
    points: list[Point]
    fit: np.ndarray
    min_y: float
    max_y: float
    median_y: float


@dataclass
class LaneCandidate:
    index: int
    fit: np.ndarray
    x_ref: float


@dataclass
class LanePairCandidate:
    left: LaneCandidate
    right: LaneCandidate
    center_x: float
    width_px: float


@dataclass
class LanePairSelection:
    left_fit: np.ndarray | None
    right_fit: np.ndarray | None
    label: str
    candidates: list[LaneCandidate]


@dataclass
class LaneControlTarget:
    fit: np.ndarray | None
    near_x: float | None
    far_x: float | None
    label: str


@dataclass
class LaneAreaCandidate:
    name: str
    mask: np.ndarray
    points: list[Point]
    fit: np.ndarray
    near_x: float
    far_x: float
    width_px: float | None
    score: float


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

        predict_kwargs = {
            "source": model_frame,
            "imgsz": self.config.image_size,
            "conf": self.config.confidence,
            "iou": self.config.iou,
            "device": self.config.device,
            "verbose": False,
            "retina_masks": True,
        }
        allowed_class_ids = self._allowed_class_ids(getattr(self.model, "names", None))
        if allowed_class_ids is not None:
            predict_kwargs["classes"] = allowed_class_ids
        results = self.model.predict(**predict_kwargs)
        result = results[0]
        model_objects = self._extract_objects(result, model_h, model_w)
        segmentation_mode = self.config.segmentation_mode.lower().strip()
        if segmentation_mode == "lane_area":
            return self._detect_lane_area(
                model_frame,
                result,
                model_objects,
                traffic_light_state_from_objects(model_frame, model_objects),
            )

        model_mask = self._combined_mask(result, model_h, model_w)
        if segmentation_mode == "drivable_area":
            return self._detect_drivable_area(
                model_frame,
                model_mask,
                result,
                model_objects,
                traffic_light_state_from_objects(model_frame, model_objects),
            )

        lane_fragments = self._extract_lanes(result, model_h, model_w)
        if transform is not None:
            lane_fragments = [transform.points_to_original(lane) for lane in lane_fragments]
        objects = self._objects_to_original(model_objects, transform) if transform is not None else model_objects
        traffic_light_state = traffic_light_state_from_objects(frame, objects)

        fits = self._fit_lane_groups(lane_fragments, h, w, near_y, frame_center_x)
        if not fits:
            fits = [self._fit_lane_x_of_y(lane, h) for lane in lane_fragments]
            fits = [fit for fit in fits if fit is not None]
        selection_y = int(h * clamp(self.config.lane_pair_select_y_ratio, 0.05, 0.98))
        selection = self._select_left_right_fits(fits, frame_center_x, near_y, selection_y, w)
        left_fit, right_fit = selection.left_fit, selection.right_fit
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

        control_target = self._select_control_target(selection, fits, frame_center_x, near_y, far_y, w)
        target_fit = control_target.fit
        target_label = control_target.label
        if target_fit is not None:
            center_near = control_target.near_x
            center_far = control_target.far_x

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

        center_fit = target_fit if target_fit is not None else self._center_fit(left_fit, right_fit)
        y_eval = near_y - self.config.curve_lookahead_ratio * (near_y - far_y)
        curvature = self._curvature_from_fit(center_fit, y_eval)
        lane_label = selection.label if not target_label else f"{selection.label} {target_label}"

        display_in_bev = (
            self.config.display_bird_eye_view
            and transform is not None
            and transform.perspective_matrix is not None
        )
        mask = model_mask if display_in_bev else transform.mask_to_original(model_mask) if transform is not None else model_mask
        annotated_original = self._annotate(
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
            pair_label=lane_label,
            candidate_fits=selection.candidates,
            selection_y=selection_y,
            target_fit=target_fit,
        )
        self._draw_detected_objects(annotated_original, objects)
        annotated = (
            transform.image_to_processed(annotated_original, mask_source_polygon=True)
            if display_in_bev and transform is not None
            else annotated_original
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
            objects=objects,
            curvature=curvature,
            lane_pair_label=lane_label,
            traffic_light_state=traffic_light_state,
        )

    def _detect_drivable_area(
        self,
        frame: np.ndarray,
        mask: np.ndarray | None,
        result,
        objects: Sequence[DetectedObject],
        traffic_light_state: str,
    ) -> LaneDetection:
        h, w = frame.shape[:2]
        near_y = int(h * self.config.control_near_y_ratio)
        far_y = int(h * self.config.control_far_y_ratio)
        frame_center_x = w / 2.0

        center_points, row_widths = self._centerline_from_drivable_area_mask(mask)
        center_fit = self._fit_lane_x_of_y(center_points, h) if center_points else None
        lanes = self._solidified_lanes_from_fits((center_fit,), h, w, near_y, far_y)
        width_px = self._update_area_width(row_widths)

        center_near = self._x_at_y(center_fit, near_y)
        center_far = self._x_at_y(center_fit, far_y)
        confidence = self._confidence(center_fit, None, 1 if center_fit is not None else 0, result)

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

        y_eval = near_y - self.config.curve_lookahead_ratio * (near_y - far_y)
        curvature = self._curvature_from_fit(center_fit, y_eval)
        label = f"mode=drivable_area target=area_center points={len(center_points)}"
        if width_px is not None:
            label += f" width={width_px:.0f}"

        annotated = self._annotate_drivable_area(
            frame,
            mask,
            center_points,
            center_fit,
            lanes,
            center_near,
            center_far,
            near_y,
            far_y,
            confidence,
            label,
        )
        self._draw_detected_objects(annotated, objects)
        return LaneDetection(
            lanes=lanes,
            left_fit=center_fit,
            right_fit=None,
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
            objects=objects,
            curvature=curvature,
            lane_pair_label=label,
            traffic_light_state=traffic_light_state,
        )

    def _detect_lane_area(
        self,
        frame: np.ndarray,
        result,
        objects: Sequence[DetectedObject],
        traffic_light_state: str,
    ) -> LaneDetection:
        h, w = frame.shape[:2]
        near_y = int(h * self.config.control_near_y_ratio)
        far_y = int(h * self.config.control_far_y_ratio)
        frame_center_x = w / 2.0

        masks_by_name = self._class_masks(result, h, w, target_names=("lane1", "lane2"))
        combined_mask = self._combined_mask_from_named_masks(masks_by_name, h, w)
        candidates = self._lane_area_candidates_from_masks(masks_by_name, h, w, near_y, far_y, frame_center_x)
        lane_references = self._lane_references_from_candidates(candidates, near_y, far_y)
        objects = self._objects_with_lane_labels(objects, candidates, h, w)
        selected = candidates[0] if candidates else None

        center_fit = selected.fit if selected is not None else None
        lanes = self._solidified_lanes_from_fits((center_fit,), h, w, near_y, far_y)
        width_px = self._update_area_width([selected.width_px] if selected is not None and selected.width_px is not None else [])

        center_near = self._x_at_y(center_fit, near_y)
        center_far = self._x_at_y(center_fit, far_y)
        confidence = self._confidence(center_fit, None, len(candidates), result) if center_fit is not None else 0.0

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

        y_eval = near_y - self.config.curve_lookahead_ratio * (near_y - far_y)
        curvature = self._curvature_from_fit(center_fit, y_eval)
        if selected is None:
            label = "mode=lane_area target=none"
        else:
            vx = selected.near_x - frame_center_x
            label = f"mode=lane_area target={selected.name} points={len(selected.points)} vx={vx:+.0f}"
            if selected.width_px is not None:
                label += f" width={selected.width_px:.0f}"

        annotated = self._annotate_lane_area(
            frame,
            masks_by_name,
            candidates,
            selected,
            lanes,
            center_near,
            center_far,
            near_y,
            far_y,
            confidence,
            label,
        )
        self._draw_detected_objects(annotated, objects)
        return LaneDetection(
            lanes=lanes,
            left_fit=center_fit,
            right_fit=None,
            lane_center_near_x=center_near,
            lane_center_far_x=center_far,
            frame_center_x=frame_center_x,
            offset_px=offset_px,
            offset_norm=offset_norm,
            heading_deg=heading_deg,
            lane_width_px=width_px,
            confidence=confidence,
            mask=combined_mask,
            annotated=annotated,
            objects=objects,
            lane_label=selected.name if selected is not None else "",
            lane_references=lane_references,
            curvature=curvature,
            lane_pair_label=label,
            traffic_light_state=traffic_light_state,
        )

    def _lane_area_candidates_from_masks(
        self,
        masks_by_name: dict[str, np.ndarray],
        image_height: int,
        image_width: int,
        near_y: int,
        far_y: int,
        frame_center_x: float,
    ) -> list[LaneAreaCandidate]:
        target_x = frame_center_x + self.config.lane_pair_target_offset_ratio * image_width
        candidates: list[LaneAreaCandidate] = []
        for name in ("lane1", "lane2"):
            mask = masks_by_name.get(name)
            if mask is None:
                continue
            points, row_widths = self._centerline_from_drivable_area_mask(mask)
            fit = self._fit_lane_x_of_y(points, image_height) if points else None
            if fit is None:
                continue
            near_x = self._x_at_y(fit, near_y)
            far_x = self._x_at_y(fit, far_y)
            if near_x is None or far_x is None or not math.isfinite(near_x) or not math.isfinite(far_x):
                continue
            if not (-0.5 * image_width <= near_x <= 1.5 * image_width):
                continue
            width_px = float(np.median(np.array(row_widths, dtype=np.float64))) if row_widths else None
            candidates.append(
                LaneAreaCandidate(
                    name=name,
                    mask=mask,
                    points=points,
                    fit=fit,
                    near_x=float(near_x),
                    far_x=float(far_x),
                    width_px=width_px,
                    score=abs(float(near_x) - target_x),
                )
            )
        return sorted(candidates, key=lambda item: item.score)

    @staticmethod
    def _lane_references_from_candidates(
        candidates: Sequence[LaneAreaCandidate],
        near_y: int,
        far_y: int,
    ) -> dict[str, LaneReference]:
        return {
            candidate.name: LaneReference(
                name=candidate.name,
                near_x=candidate.near_x,
                far_x=candidate.far_x,
                near_y=near_y,
                far_y=far_y,
                width_px=candidate.width_px,
                fit=candidate.fit,
            )
            for candidate in candidates
        }

    def _objects_with_lane_labels(
        self,
        objects: Sequence[DetectedObject],
        candidates: Sequence[LaneAreaCandidate],
        image_height: int,
        image_width: int,
    ) -> tuple[DetectedObject, ...]:
        if not objects or not candidates:
            return tuple(objects)

        labeled: list[DetectedObject] = []
        for obj in objects:
            best_candidate = None
            best_distance = float("inf")
            y = float(clamp(obj.bbox.bottom_y, 0.0, image_height - 1.0))
            for candidate in candidates:
                x = self._x_at_y(candidate.fit, y)
                if x is None:
                    continue
                distance = abs(obj.bbox.center_x - float(x))
                if distance < best_distance:
                    best_candidate = candidate
                    best_distance = distance

            lane_label = ""
            lane_distance = None
            if best_candidate is not None:
                half_width = (
                    best_candidate.width_px / 2.0
                    if best_candidate.width_px is not None
                    else image_width * 0.16
                )
                max_distance = max(image_width * 0.10, half_width + image_width * 0.04)
                if best_distance <= max_distance:
                    lane_label = best_candidate.name
                    lane_distance = float(best_distance)

            labeled.append(
                DetectedObject(
                    class_name=obj.class_name,
                    confidence=obj.confidence,
                    bbox=obj.bbox,
                    class_id=obj.class_id,
                    mask_area_px=obj.mask_area_px,
                    lane_label=lane_label,
                    lane_distance_px=lane_distance,
                )
            )
        return tuple(labeled)

    def _class_masks(
        self,
        result,
        height: int,
        width: int,
        target_names: Sequence[str],
    ) -> dict[str, np.ndarray]:
        if result.masks is None or result.boxes is None or result.boxes.cls is None:
            return {}
        masks = result.masks.data.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        masks_by_name: dict[str, np.ndarray] = {}
        for index, mask in enumerate(masks):
            if index >= len(class_ids):
                continue
            class_name = self._class_name(int(class_ids[index]), result.names)
            if not self._class_allowed_name(class_name):
                continue
            target = self._matching_target_class(class_name, target_names)
            if target is None:
                continue
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
            target_mask = masks_by_name.setdefault(target, np.zeros((height, width), dtype=np.uint8))
            target_mask[mask > 0.5] = 255
        return {name: mask for name, mask in masks_by_name.items() if cv2.countNonZero(mask) > 0}

    @staticmethod
    def _combined_mask_from_named_masks(masks_by_name: dict[str, np.ndarray], height: int, width: int) -> np.ndarray | None:
        if not masks_by_name:
            return None
        combined = np.zeros((height, width), dtype=np.uint8)
        for mask in masks_by_name.values():
            combined[mask > 0] = 255
        return combined if cv2.countNonZero(combined) > 0 else None

    def _extract_objects(self, result, height: int, width: int) -> tuple[DetectedObject, ...]:
        if result.boxes is None or result.boxes.cls is None or getattr(result.boxes, "xyxy", None) is None:
            return ()

        xyxy = self._tensor_to_numpy(result.boxes.xyxy)
        class_ids = self._tensor_to_numpy(result.boxes.cls).astype(int)
        confidences = (
            self._tensor_to_numpy(result.boxes.conf).astype(float)
            if getattr(result.boxes, "conf", None) is not None
            else np.ones(len(class_ids), dtype=np.float64)
        )
        masks = self._tensor_to_numpy(result.masks.data) if getattr(result, "masks", None) is not None else None

        objects: list[DetectedObject] = []
        for index, box in enumerate(xyxy):
            if index >= len(class_ids):
                continue
            class_name = self._class_name(int(class_ids[index]), result.names)
            if not self._class_allowed_name(class_name) or not self._is_non_lane_object_name(class_name):
                continue
            x1, y1, x2, y2 = [float(value) for value in box]
            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2).clipped(width, height)
            if bbox.area <= 0.0:
                continue
            mask_area = None
            if masks is not None and index < len(masks):
                mask = masks[index]
                if mask.shape[:2] != (height, width):
                    mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
                mask_area = float(np.count_nonzero(mask > 0.5))
            objects.append(
                DetectedObject(
                    class_name=class_name,
                    confidence=float(confidences[index]) if index < len(confidences) else 1.0,
                    bbox=bbox,
                    class_id=int(class_ids[index]),
                    mask_area_px=mask_area,
                )
            )
        return tuple(objects)

    @staticmethod
    def _objects_to_original(objects: Sequence[DetectedObject], transform) -> tuple[DetectedObject, ...]:
        mapped: list[DetectedObject] = []
        for obj in objects:
            box = obj.bbox
            points = transform.points_to_original(
                [
                    (int(round(box.x1)), int(round(box.y1))),
                    (int(round(box.x2)), int(round(box.y1))),
                    (int(round(box.x2)), int(round(box.y2))),
                    (int(round(box.x1)), int(round(box.y2))),
                ]
            )
            if not points:
                continue
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            mapped.append(
                DetectedObject(
                    class_name=obj.class_name,
                    confidence=obj.confidence,
                    bbox=BoundingBox(float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))),
                    class_id=obj.class_id,
                    mask_area_px=obj.mask_area_px,
                    lane_label=obj.lane_label,
                    lane_distance_px=obj.lane_distance_px,
                )
            )
        return tuple(mapped)

    @staticmethod
    def _draw_detected_objects(image: np.ndarray, objects: Sequence[DetectedObject]) -> None:
        for obj in objects:
            box = obj.bbox.clipped(image.shape[1], image.shape[0])
            x1, y1 = int(round(box.x1)), int(round(box.y1))
            x2, y2 = int(round(box.x2)), int(round(box.y2))
            color = (0, 255, 255) if "traffic" in obj.compact_class_name else (0, 165, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            lane_suffix = f" {obj.lane_label}" if obj.lane_label else ""
            label = f"{obj.class_name} {obj.confidence:.2f}{lane_suffix}"
            cv2.putText(image, label, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(image, label, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    @staticmethod
    def _tensor_to_numpy(value) -> np.ndarray:
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

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
            if boxes is not None and not self._class_allowed_for_lane_geometry(int(boxes.cls[index].item()), result.names):
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

    def _centerline_from_drivable_area_mask(self, mask: np.ndarray | None) -> tuple[list[Point], list[float]]:
        if mask is None:
            return [], []

        binary = (mask > 0).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        ys, _ = np.where(binary > 0)
        if len(ys) == 0:
            return [], []

        h, w = binary.shape[:2]
        step = max(1, self.config.mask_sample_step_px)
        min_width_px = max(3.0, w * self.config.drivable_min_row_width_ratio)
        edge_pct = clamp(self.config.drivable_edge_percentile, 0.0, 20.0)
        points: list[Point] = []
        widths: list[float] = []

        for y in range(int(ys.min()), int(ys.max()) + 1, step):
            row_x = np.flatnonzero(binary[y] > 0)
            if len(row_x) < min_width_px:
                continue
            left = float(np.percentile(row_x, edge_pct))
            right = float(np.percentile(row_x, 100.0 - edge_pct))
            width = right - left
            if width < min_width_px:
                continue
            points.append((int(round((left + right) / 2.0)), int(y)))
            widths.append(width)

        return points, widths

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
            if not self._class_allowed_for_lane_geometry(int(boxes.cls[index].item()), result.names):
                continue
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2.0)
            lanes.append([(center_x, int(y1)), (center_x, int((y1 + y2) / 2.0)), (center_x, int(y2))])
        return lanes

    def _class_allowed(self, class_id: int, names) -> bool:
        return self._class_allowed_name(self._class_name(class_id, names))

    def _class_allowed_for_lane_geometry(self, class_id: int, names) -> bool:
        class_name = self._class_name(class_id, names)
        return self._class_allowed_name(class_name) and not self._is_non_lane_object_name(class_name)

    def _class_allowed_name(self, class_name: str) -> bool:
        if not self.class_names or "*" in self.class_names:
            return True
        normalized = str(class_name).lower()
        compact = self._compact_class_name(normalized)
        for token in self.class_names:
            token = str(token).lower()
            if self._compact_class_name(token) == compact or token in normalized:
                return True
        return False

    @staticmethod
    def _class_name(class_id: int, names) -> str:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id)).lower()
        if names is not None and 0 <= class_id < len(names):
            return str(names[class_id]).lower()
        return str(class_id)

    def _allowed_class_ids(self, names) -> list[int] | None:
        if not self.class_names or "*" in self.class_names or names is None:
            return None
        items = names.items() if isinstance(names, dict) else enumerate(names)
        class_ids: list[int] = []
        for class_id, class_name in items:
            try:
                class_id_int = int(class_id)
            except (TypeError, ValueError):
                continue
            if self._class_allowed_name(str(class_name).lower()):
                class_ids.append(class_id_int)
        return class_ids or None

    @staticmethod
    def _compact_class_name(class_name: str) -> str:
        return "".join(ch for ch in str(class_name).lower() if ch.isalnum())

    def _matching_target_class(self, class_name: str, target_names: Sequence[str]) -> str | None:
        compact = self._compact_class_name(class_name)
        for target in target_names:
            if compact == self._compact_class_name(target):
                return target
        return None

    def _is_non_lane_object_name(self, class_name: str) -> bool:
        compact = self._compact_class_name(class_name)
        object_names = {
            "car",
            "obstacle",
            "person",
            "pedestrian",
            "truck",
            "bus",
            "motorcycle",
            "bicycle",
            "cone",
            "barrier",
            "trafficlight",
            "signallight",
            "signal",
            "stoplight",
        }
        return compact in object_names or compact.startswith(("trafficlight", "signallight", "stoplight"))

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

    def _select_left_right_fits(
        self,
        fits,
        frame_center_x: float,
        near_y: int,
        selection_y: int,
        image_width: int,
    ) -> LanePairSelection:
        candidates = self._lane_candidates(fits, selection_y, image_width)
        if not candidates:
            return LanePairSelection(None, None, "pair=none", [])

        pairs = self._adjacent_lane_pairs(candidates, image_width)
        target_x = frame_center_x + self.config.lane_pair_target_offset_ratio * image_width
        mode = self.config.target_lane_pair.lower().strip()

        if pairs and mode in ("closest", "center"):
            pair = min(pairs, key=lambda item: self._pair_score(item, target_x, image_width))
            return self._selection_from_pair(pair, f"pair={mode} {pair.left.index}-{pair.right.index}", candidates)

        if pairs and mode == "left":
            pair = min(pairs, key=lambda item: item.center_x)
            return self._selection_from_pair(pair, f"pair=left {pair.left.index}-{pair.right.index}", candidates)

        if pairs and mode == "right":
            pair = self._right_lane_pair_or_none(pairs, target_x, image_width)
            if pair is not None:
                return self._selection_from_pair(pair, f"pair=right {pair.left.index}-{pair.right.index}", candidates)
            boundary = max(candidates, key=lambda item: item.x_ref)
            expected_width = self.lane_width_px or image_width * self.config.default_lane_width_ratio
            if boundary.x_ref > target_x + expected_width * 0.25:
                return LanePairSelection(None, boundary.fit, f"pair=right single-right {boundary.index}", candidates)
            return LanePairSelection(boundary.fit, None, f"pair=right single-left {boundary.index}", candidates)

        left_candidates = []
        right_candidates = []
        for candidate in candidates:
            x_near = self._x_at_y(candidate.fit, near_y)
            if x_near is None or not math.isfinite(x_near):
                continue
            if x_near < frame_center_x:
                left_candidates.append((abs(frame_center_x - x_near), candidate))
            else:
                right_candidates.append((abs(x_near - frame_center_x), candidate))
        left = min(left_candidates, default=(None, None), key=lambda item: item[0])[1]
        right = min(right_candidates, default=(None, None), key=lambda item: item[0])[1]
        label = "pair=split"
        if left is not None or right is not None:
            left_id = "-" if left is None else str(left.index)
            right_id = "-" if right is None else str(right.index)
            label = f"pair=split {left_id}-{right_id}"
        return LanePairSelection(
            left.fit if left is not None else None,
            right.fit if right is not None else None,
            label,
            candidates,
        )

    def _select_control_target(
        self,
        selection: LanePairSelection,
        fits,
        frame_center_x: float,
        near_y: int,
        far_y: int,
        image_width: int,
    ) -> LaneControlTarget:
        mode = self.config.target_path_mode.lower().strip()
        if mode in ("lane_center", "pair_center", "center"):
            return LaneControlTarget(None, None, None, "target=lane_center")

        candidates = selection.candidates or self._lane_candidates(fits, near_y, image_width)
        if not candidates:
            return LaneControlTarget(None, None, None, "target=none")

        target_x = frame_center_x + self.config.lane_pair_target_offset_ratio * image_width
        ranked = []
        for candidate in candidates:
            near_x = self._x_at_y(candidate.fit, near_y)
            far_x = self._x_at_y(candidate.fit, far_y)
            if near_x is None or far_x is None or not math.isfinite(near_x) or not math.isfinite(far_x):
                continue
            ranked.append((abs(near_x - target_x), candidate, near_x, far_x))

        if not ranked:
            return LaneControlTarget(None, None, None, "target=none")

        if mode == "left_line":
            _, candidate, near_x, far_x = min(ranked, key=lambda item: item[2])
        elif mode == "right_line":
            _, candidate, near_x, far_x = max(ranked, key=lambda item: item[2])
        else:
            _, candidate, near_x, far_x = min(ranked, key=lambda item: item[0])

        mode_label = mode if mode in ("closest_line", "left_line", "right_line") else "closest_line"
        return LaneControlTarget(candidate.fit, near_x, far_x, f"target={mode_label} {candidate.index} x={near_x:.0f}")

    def _lane_candidates(self, fits, selection_y: int, image_width: int) -> list[LaneCandidate]:
        candidates = []
        for fit in fits:
            x_ref = self._x_at_y(fit, selection_y)
            if x_ref is None or not math.isfinite(x_ref):
                continue
            if not (-0.5 * image_width <= x_ref <= 1.5 * image_width):
                continue
            candidates.append(LaneCandidate(index=0, fit=fit, x_ref=float(x_ref)))
        candidates.sort(key=lambda item: item.x_ref)
        for index, candidate in enumerate(candidates):
            candidate.index = index
        return candidates

    def _adjacent_lane_pairs(self, candidates: Sequence[LaneCandidate], image_width: int) -> list[LanePairCandidate]:
        pairs = []
        min_width = max(24.0, image_width * self.config.min_lane_width_ratio * 0.45)
        max_width = max(min_width + 1.0, image_width * self.config.max_lane_width_ratio * 1.75)
        for left, right in zip(candidates, candidates[1:]):
            width = right.x_ref - left.x_ref
            if min_width <= width <= max_width:
                pairs.append(
                    LanePairCandidate(
                        left=left,
                        right=right,
                        center_x=(left.x_ref + right.x_ref) / 2.0,
                        width_px=width,
                    )
                )
        return pairs

    def _right_lane_pair_or_none(
        self,
        pairs: Sequence[LanePairCandidate],
        target_x: float,
        image_width: int,
    ) -> LanePairCandidate | None:
        if not pairs:
            return None
        expected_width = self.lane_width_px or image_width * self.config.default_lane_width_ratio
        right_side_margin = expected_width * 0.35
        right_side_pairs = [pair for pair in pairs if pair.center_x >= target_x - right_side_margin]
        if right_side_pairs:
            return max(right_side_pairs, key=lambda item: item.center_x)
        if len(pairs) >= 2:
            return max(pairs, key=lambda item: item.center_x)
        only_pair = pairs[0]
        if only_pair.center_x < target_x - expected_width * 0.45:
            return None
        return only_pair

    def _pair_score(self, pair: LanePairCandidate, target_x: float, image_width: int) -> float:
        expected_width = self.lane_width_px or image_width * self.config.default_lane_width_ratio
        center_score = abs(pair.center_x - target_x)
        width_score = abs(pair.width_px - expected_width) * 0.35
        return center_score + width_score

    @staticmethod
    def _selection_from_pair(pair: LanePairCandidate, label: str, candidates: list[LaneCandidate]) -> LanePairSelection:
        return LanePairSelection(
            pair.left.fit,
            pair.right.fit,
            f"{label} w={pair.width_px:.0f} cx={pair.center_x:.0f}",
            candidates,
        )

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

    def _update_area_width(self, widths: Sequence[float]) -> float | None:
        if not widths:
            return self.lane_width_px
        new_width = float(np.median(np.array(widths, dtype=np.float64)))
        if self.lane_width_px is None:
            self.lane_width_px = new_width
        else:
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

    def _confidence(self, left_fit, right_fit, lane_count: int, result) -> float:
        confidence = 0.0
        if left_fit is not None:
            confidence += 0.42
        if right_fit is not None:
            confidence += 0.42
        confidence += min(lane_count, 4) * 0.04
        if result.boxes is not None and len(result.boxes) > 0 and result.boxes.conf is not None:
            box_conf = result.boxes.conf.detach().cpu().numpy()
            if result.boxes.cls is not None:
                class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
                keep = [
                    index
                    for index, class_id in enumerate(class_ids)
                    if index < len(box_conf) and self._class_allowed_for_lane_geometry(int(class_id), result.names)
                ]
                box_conf = box_conf[keep] if keep else np.array([], dtype=np.float64)
            if len(box_conf) > 0:
                confidence = max(confidence, float(np.clip(np.mean(box_conf), 0.0, 1.0)))
        return float(clamp(confidence, 0.0, 1.0))

    def _combined_mask(self, result, height: int, width: int) -> np.ndarray | None:
        if result.masks is None:
            return None
        masks = result.masks.data.detach().cpu().numpy()
        if len(masks) == 0:
            return None
        boxes = result.boxes
        combined = np.zeros((height, width), dtype=np.uint8)
        for index, mask in enumerate(masks):
            if boxes is not None and not self._class_allowed_for_lane_geometry(int(boxes.cls[index].item()), result.names):
                continue
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
            combined[mask > 0.5] = 255
        return combined if cv2.countNonZero(combined) > 0 else None

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

    def _annotate(
        self,
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
        transform=None,
        pair_label: str = "",
        candidate_fits: Sequence[LaneCandidate] | None = None,
        selection_y: int | None = None,
        target_fit=None,
    ):
        vis = frame.copy()
        if transform is not None:
            transform.draw_overlay(vis)
        for lane in lane_fragments:
            for point in lane:
                cv2.circle(vis, point, 1, (0, 255, 255), -1)
        if candidate_fits and selection_y is not None:
            for candidate in candidate_fits:
                x = int(clamp(candidate.x_ref, 0, vis.shape[1] - 1))
                is_selected = candidate.fit is left_fit or candidate.fit is right_fit
                is_target = target_fit is not None and candidate.fit is target_fit
                color = (255, 255, 0) if is_target else (0, 255, 0) if is_selected else (180, 180, 180)
                cv2.circle(vis, (x, selection_y), 5, color, -1)
                cv2.putText(vis, str(candidate.index), (x + 7, selection_y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(vis, str(candidate.index), (x + 7, selection_y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.line(vis, (0, selection_y), (vis.shape[1] - 1, selection_y), (120, 120, 120), 1)
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

    def _annotate_drivable_area(
        self,
        frame,
        mask,
        center_points,
        center_fit,
        lanes,
        center_near,
        center_far,
        near_y,
        far_y,
        confidence,
        label,
    ):
        vis = frame.copy()
        if mask is not None:
            overlay = vis.copy()
            overlay[mask > 0] = (0, 120, 0)
            vis = cv2.addWeighted(overlay, 0.28, vis, 0.72, 0)

        for point in center_points:
            cv2.circle(vis, point, 1, (0, 255, 255), -1)
        for lane in lanes:
            if len(lane) >= 2:
                cv2.polylines(vis, [np.array(lane, dtype=np.int32)], isClosed=False, color=(255, 255, 0), thickness=2)
        self._draw_fit(vis, center_fit, near_y, far_y, (255, 255, 0))

        h, w = vis.shape[:2]
        cv2.line(vis, (w // 2, h - 1), (w // 2, int(h * 0.55)), (0, 0, 255), 1)
        if center_near is not None and center_far is not None:
            cv2.line(vis, (int(center_near), near_y), (int(center_far), far_y), (255, 255, 0), 2)
            cv2.circle(vis, (int(center_near), near_y), 6, (255, 255, 0), -1)

        cv2.putText(vis, f"yolo conf={confidence:.2f}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(vis, f"yolo conf={confidence:.2f}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(vis, label, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(vis, label, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return vis

    def _annotate_lane_area(
        self,
        frame,
        masks_by_name: dict[str, np.ndarray],
        candidates: Sequence[LaneAreaCandidate],
        selected: LaneAreaCandidate | None,
        lanes,
        center_near,
        center_far,
        near_y,
        far_y,
        confidence,
        label,
    ):
        vis = frame.copy()
        if masks_by_name:
            overlay = vis.copy()
            colors = {
                "lane1": (0, 150, 0),
                "lane2": (180, 60, 0),
            }
            for name, mask in masks_by_name.items():
                overlay[mask > 0] = colors.get(name, (0, 120, 0))
            vis = cv2.addWeighted(overlay, 0.28, vis, 0.72, 0)

        for candidate in candidates:
            is_selected = candidate is selected
            fit_color = (255, 255, 0) if is_selected else (150, 150, 150)
            point_color = (0, 255, 255) if is_selected else (110, 110, 110)
            for point in candidate.points:
                cv2.circle(vis, point, 1, point_color, -1)
            self._draw_fit(vis, candidate.fit, near_y, far_y, fit_color)
            label_x = int(clamp(candidate.near_x, 0, vis.shape[1] - 1))
            cv2.putText(vis, candidate.name, (label_x + 6, near_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(vis, candidate.name, (label_x + 6, near_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fit_color, 1)

        for lane in lanes:
            if len(lane) >= 2:
                cv2.polylines(vis, [np.array(lane, dtype=np.int32)], isClosed=False, color=(255, 255, 0), thickness=2)

        h, w = vis.shape[:2]
        cv2.line(vis, (w // 2, h - 1), (w // 2, int(h * 0.55)), (0, 0, 255), 1)
        if center_near is not None and center_far is not None:
            cv2.line(vis, (int(center_near), near_y), (int(center_far), far_y), (255, 255, 0), 2)
            cv2.circle(vis, (int(center_near), near_y), 6, (255, 255, 0), -1)

        cv2.putText(vis, f"yolo conf={confidence:.2f}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(vis, f"yolo conf={confidence:.2f}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(vis, label, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(vis, label, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
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
