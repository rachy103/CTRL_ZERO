from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from ctrl_zero.control import DriveCommand
from ctrl_zero.lidar import ObstacleDecision
from ctrl_zero.perception import BoundingBox, DetectedObject, compact_class_name
from ctrl_zero.safety import SafetyDecision
from ctrl_zero.vision.base import LaneDetection, LaneReference


@dataclass
class LogConfig:
    enabled: bool = False
    directory: Path = Path("Log")
    save_every_n_frames: int = 5
    run_name: str = ""


class DriveLogger:
    def __init__(self, config: LogConfig):
        self.config = config
        self.index = 0
        self.run_dir: Path | None = None
        self.csv_file = None
        self.objects_file = None
        self.lane_refs_file = None
        self.jsonl_file = None
        self.writer = None
        self.objects_writer = None
        self.lane_refs_writer = None
        self.frame_dir = config.directory / "frames"
        self.annotated_dir = config.directory / "annotated"
        self.mask_dir = config.directory / "masks"

    def open(self) -> None:
        if not self.config.enabled:
            return
        if self.writer is not None:
            return
        self.run_dir = self._create_run_dir()
        self.frame_dir = self.run_dir / "frames"
        self.annotated_dir = self.run_dir / "annotated"
        self.mask_dir = self.run_dir / "masks"
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        self.csv_file = (self.run_dir / "drive_log.csv").open("w", newline="", encoding="utf-8")
        self.objects_file = (self.run_dir / "objects.csv").open("w", newline="", encoding="utf-8")
        self.lane_refs_file = (self.run_dir / "lane_references.csv").open("w", newline="", encoding="utf-8")
        self.jsonl_file = (self.run_dir / "frames.jsonl").open("w", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "timestamp",
                "timestamp_iso",
                "frame",
                "mode",
                "backend",
                "steer",
                "speed",
                "reason",
                "confidence",
                "offset_px",
                "offset_norm",
                "heading_deg",
                "curvature_1_per_px",
                "lane_width_px",
                "lane_pair",
                "current_lane",
                "lane_center_near_x",
                "lane_center_far_x",
                "frame_center_x",
                "lane_count",
                "lane_reference_count",
                "traffic_light_state",
                "traffic_light_area_ratio",
                "object_count",
                "car_count",
                "traffic_light_count",
                "safety_reason",
                "safety_should_stop",
                "safety_speed_scale",
                "vision_obstacle_class",
                "vision_obstacle_lane",
                "vision_obstacle_confidence",
                "vision_obstacle_bbox",
                "vision_obstacle_area_ratio",
                "avoidance_steer",
                "avoidance_current_lane",
                "avoidance_target_lane",
                "nearest_front_mm",
                "lidar_speed_scale",
                "lidar_front_points",
                "image_file",
                "annotated_file",
                "mask_file",
            ],
        )
        self.writer.writeheader()
        self.objects_writer = csv.DictWriter(
            self.objects_file,
            fieldnames=[
                "timestamp",
                "timestamp_iso",
                "frame",
                "object_index",
                "class_name",
                "compact_class_name",
                "class_id",
                "confidence",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "bbox_width",
                "bbox_height",
                "bbox_area",
                "bbox_area_ratio",
                "bbox_center_x",
                "bbox_center_y",
                "bbox_bottom_y",
                "bbox_bottom_ratio",
                "lane_label",
                "lane_distance_px",
                "mask_area_px",
                "is_vision_obstacle",
            ],
        )
        self.objects_writer.writeheader()
        self.lane_refs_writer = csv.DictWriter(
            self.lane_refs_file,
            fieldnames=[
                "timestamp",
                "timestamp_iso",
                "frame",
                "name",
                "near_x",
                "far_x",
                "near_y",
                "far_y",
                "width_px",
                "fit",
            ],
        )
        self.lane_refs_writer.writeheader()
        self._write_metadata()
        print(f"Logging enabled: {self.run_dir}")

    def log(
        self,
        frame,
        mode: str,
        backend: str,
        lane: LaneDetection,
        obstacle: ObstacleDecision | SafetyDecision | None,
        command: DriveCommand,
    ) -> None:
        if not self.config.enabled or self.writer is None:
            return
        self.index += 1
        timestamp = time.time()
        timestamp_iso = datetime.fromtimestamp(timestamp).isoformat(timespec="milliseconds")
        frame_height = lane.annotated.shape[0]
        frame_area = max(float(frame_height * lane.annotated.shape[1]), 1.0)
        image_file = ""
        annotated_file = ""
        mask_file = ""
        if self.index % max(self.config.save_every_n_frames, 1) == 0:
            image_file = f"frame_{self.index:06d}.jpg"
            annotated_file = f"annotated_{self.index:06d}.jpg"
            cv2.imwrite(str(self.frame_dir / image_file), frame)
            cv2.imwrite(str(self.annotated_dir / annotated_file), lane.annotated)
            image_file = f"frames/{image_file}"
            annotated_file = f"annotated/{annotated_file}"
            if lane.mask is not None:
                mask_file = f"mask_{self.index:06d}.png"
                cv2.imwrite(str(self.mask_dir / Path(mask_file).name), lane.mask)
                mask_file = f"masks/{mask_file}"

        vision_obstacle = obstacle.vision_obstacle if isinstance(obstacle, SafetyDecision) else None
        lidar = _lidar_decision(obstacle)
        objects = list(lane.objects)
        car_count = sum(1 for obj in objects if compact_class_name(obj.class_name) == "car")
        traffic_light_count = sum(1 for obj in objects if compact_class_name(obj.class_name) in {"trafficlight", "trafficlightred", "trafficlightyellow", "trafficlightgreen"})
        self.writer.writerow(
            {
                "timestamp": f"{timestamp:.6f}",
                "timestamp_iso": timestamp_iso,
                "frame": self.index,
                "mode": mode,
                "backend": backend,
                "steer": command.steer,
                "speed": command.speed,
                "reason": command.reason,
                "confidence": f"{lane.confidence:.3f}",
                "offset_px": "" if lane.offset_px is None else f"{lane.offset_px:.3f}",
                "offset_norm": "" if lane.offset_norm is None else f"{lane.offset_norm:.5f}",
                "heading_deg": "" if lane.heading_deg is None else f"{lane.heading_deg:.3f}",
                "curvature_1_per_px": f"{lane.curvature:.8f}",
                "lane_width_px": "" if lane.lane_width_px is None else f"{lane.lane_width_px:.3f}",
                "lane_pair": lane.lane_pair_label,
                "current_lane": lane.lane_label,
                "lane_center_near_x": _fmt(lane.lane_center_near_x),
                "lane_center_far_x": _fmt(lane.lane_center_far_x),
                "frame_center_x": f"{lane.frame_center_x:.3f}",
                "lane_count": len(lane.lanes),
                "lane_reference_count": len(lane.lane_references),
                "traffic_light_state": lane.traffic_light_state,
                "traffic_light_area_ratio": "" if obstacle is None or getattr(obstacle, "traffic_light_area_ratio", None) is None else f"{obstacle.traffic_light_area_ratio:.5f}",
                "object_count": len(objects),
                "car_count": car_count,
                "traffic_light_count": traffic_light_count,
                "safety_reason": _obstacle_reason(obstacle),
                "safety_should_stop": "" if obstacle is None else bool(getattr(obstacle, "should_stop", False)),
                "safety_speed_scale": "" if obstacle is None else f"{getattr(obstacle, 'speed_scale', 1.0):.3f}",
                "vision_obstacle_class": "" if vision_obstacle is None else vision_obstacle.class_name,
                "vision_obstacle_lane": "" if vision_obstacle is None else vision_obstacle.lane_label,
                "vision_obstacle_confidence": "" if vision_obstacle is None else f"{vision_obstacle.confidence:.3f}",
                "vision_obstacle_bbox": "" if vision_obstacle is None else json.dumps(_bbox_to_dict(vision_obstacle.bbox), separators=(",", ":")),
                "vision_obstacle_area_ratio": "" if vision_obstacle is None else f"{vision_obstacle.bbox.area / frame_area:.5f}",
                "avoidance_steer": "" if obstacle is None else f"{getattr(obstacle, 'avoidance_steer', 0.0):.3f}",
                "avoidance_current_lane": "" if obstacle is None else getattr(obstacle, "current_lane_label", ""),
                "avoidance_target_lane": "" if obstacle is None else getattr(obstacle, "target_lane_label", ""),
                "nearest_front_mm": "" if lidar is None or lidar.nearest_front_mm is None else f"{lidar.nearest_front_mm:.1f}",
                "lidar_speed_scale": "" if lidar is None else f"{lidar.speed_scale:.3f}",
                "lidar_front_points": "" if lidar is None else lidar.front_points,
                "image_file": image_file,
                "annotated_file": annotated_file,
                "mask_file": mask_file,
            }
        )
        self._write_objects(timestamp, timestamp_iso, objects, frame_area, frame_height, vision_obstacle)
        self._write_lane_references(timestamp, timestamp_iso, lane)
        self._write_jsonl(timestamp, timestamp_iso, mode, backend, lane, obstacle, command, image_file, annotated_file, mask_file)
        self._flush()

    def close(self) -> None:
        for attr in ("csv_file", "objects_file", "lane_refs_file", "jsonl_file"):
            file_obj = getattr(self, attr)
            if file_obj is not None:
                file_obj.close()
                setattr(self, attr, None)
        self.writer = None
        self.objects_writer = None
        self.lane_refs_writer = None

    def set_enabled(self, enabled: bool) -> None:
        self.config.enabled = enabled
        if enabled:
            self.open()
        else:
            self.close()

    def _create_run_dir(self) -> Path:
        root = self.config.directory
        root.mkdir(parents=True, exist_ok=True)
        run_name = self.config.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = root / run_name
        suffix = 1
        while candidate.exists():
            candidate = root / f"{run_name}_{suffix:02d}"
            suffix += 1
        candidate.mkdir(parents=True)
        return candidate

    def _write_metadata(self) -> None:
        if self.run_dir is None:
            return
        metadata = {
            "started_at": datetime.now().isoformat(timespec="milliseconds"),
            "root_directory": str(self.config.directory),
            "run_directory": str(self.run_dir),
            "save_every_n_frames": self.config.save_every_n_frames,
            "files": {
                "frames": "drive_log.csv",
                "objects": "objects.csv",
                "lane_references": "lane_references.csv",
                "jsonl": "frames.jsonl",
            },
        }
        (self.run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _write_objects(
        self,
        timestamp: float,
        timestamp_iso: str,
        objects: list[DetectedObject],
        frame_area: float,
        frame_height: int,
        vision_obstacle: DetectedObject | None,
    ) -> None:
        if self.objects_writer is None:
            return
        for idx, obj in enumerate(objects):
            bbox = obj.bbox
            self.objects_writer.writerow(
                {
                    "timestamp": f"{timestamp:.6f}",
                    "timestamp_iso": timestamp_iso,
                    "frame": self.index,
                    "object_index": idx,
                    "class_name": obj.class_name,
                    "compact_class_name": obj.compact_class_name,
                    "class_id": "" if obj.class_id is None else obj.class_id,
                    "confidence": f"{obj.confidence:.3f}",
                    "bbox_x1": f"{bbox.x1:.3f}",
                    "bbox_y1": f"{bbox.y1:.3f}",
                    "bbox_x2": f"{bbox.x2:.3f}",
                    "bbox_y2": f"{bbox.y2:.3f}",
                    "bbox_width": f"{bbox.width:.3f}",
                    "bbox_height": f"{bbox.height:.3f}",
                    "bbox_area": f"{bbox.area:.3f}",
                    "bbox_area_ratio": f"{bbox.area / frame_area:.5f}",
                    "bbox_center_x": f"{bbox.center_x:.3f}",
                    "bbox_center_y": f"{bbox.center_y:.3f}",
                    "bbox_bottom_y": f"{bbox.bottom_y:.3f}",
                    "bbox_bottom_ratio": f"{bbox.bottom_y / max(float(frame_height), 1.0):.5f}",
                    "lane_label": obj.lane_label,
                    "lane_distance_px": _fmt(obj.lane_distance_px),
                    "mask_area_px": _fmt(obj.mask_area_px),
                    "is_vision_obstacle": obj == vision_obstacle,
                }
            )

    def _write_lane_references(self, timestamp: float, timestamp_iso: str, lane: LaneDetection) -> None:
        if self.lane_refs_writer is None:
            return
        for name, reference in lane.lane_references.items():
            self.lane_refs_writer.writerow(
                {
                    "timestamp": f"{timestamp:.6f}",
                    "timestamp_iso": timestamp_iso,
                    "frame": self.index,
                    "name": name,
                    "near_x": f"{reference.near_x:.3f}",
                    "far_x": _fmt(reference.far_x),
                    "near_y": reference.near_y,
                    "far_y": reference.far_y,
                    "width_px": _fmt(reference.width_px),
                    "fit": json.dumps(_array_to_list(reference.fit), separators=(",", ":")),
                }
            )

    def _write_jsonl(
        self,
        timestamp: float,
        timestamp_iso: str,
        mode: str,
        backend: str,
        lane: LaneDetection,
        obstacle: ObstacleDecision | SafetyDecision | None,
        command: DriveCommand,
        image_file: str,
        annotated_file: str,
        mask_file: str,
    ) -> None:
        if self.jsonl_file is None:
            return
        record = {
            "timestamp": timestamp,
            "timestamp_iso": timestamp_iso,
            "frame": self.index,
            "mode": mode,
            "backend": backend,
            "lane": _lane_to_dict(lane),
            "objects": [_object_to_dict(obj, lane.annotated.shape) for obj in lane.objects],
            "safety": _safety_to_dict(obstacle),
            "command": {"steer": command.steer, "speed": command.speed, "reason": command.reason},
            "files": {"image": image_file, "annotated": annotated_file, "mask": mask_file},
        }
        self.jsonl_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=_json_default) + "\n")

    def _flush(self) -> None:
        for file_obj in (self.csv_file, self.objects_file, self.lane_refs_file, self.jsonl_file):
            if file_obj is not None:
                file_obj.flush()


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _bbox_to_dict(bbox: BoundingBox) -> dict[str, float]:
    return {
        "x1": float(bbox.x1),
        "y1": float(bbox.y1),
        "x2": float(bbox.x2),
        "y2": float(bbox.y2),
        "width": float(bbox.width),
        "height": float(bbox.height),
        "area": float(bbox.area),
        "center_x": float(bbox.center_x),
        "center_y": float(bbox.center_y),
        "bottom_y": float(bbox.bottom_y),
    }


def _object_to_dict(obj: DetectedObject, frame_shape) -> dict[str, Any]:
    height, width = frame_shape[:2]
    frame_area = max(float(height * width), 1.0)
    return {
        "class_name": obj.class_name,
        "compact_class_name": obj.compact_class_name,
        "class_id": obj.class_id,
        "confidence": float(obj.confidence),
        "bbox": _bbox_to_dict(obj.bbox),
        "bbox_area_ratio": obj.bbox.area / frame_area,
        "bbox_bottom_ratio": obj.bbox.bottom_y / max(float(height), 1.0),
        "lane_label": obj.lane_label,
        "lane_distance_px": obj.lane_distance_px,
        "mask_area_px": obj.mask_area_px,
    }


def _lane_to_dict(lane: LaneDetection) -> dict[str, Any]:
    return {
        "confidence": float(lane.confidence),
        "offset_px": lane.offset_px,
        "offset_norm": lane.offset_norm,
        "heading_deg": lane.heading_deg,
        "curvature_1_per_px": lane.curvature,
        "lane_width_px": lane.lane_width_px,
        "lane_pair": lane.lane_pair_label,
        "current_lane": lane.lane_label,
        "lane_center_near_x": lane.lane_center_near_x,
        "lane_center_far_x": lane.lane_center_far_x,
        "frame_center_x": lane.frame_center_x,
        "lane_count": len(lane.lanes),
        "lane_references": {name: _lane_reference_to_dict(ref) for name, ref in lane.lane_references.items()},
        "traffic_light_state": lane.traffic_light_state,
    }


def _lane_reference_to_dict(reference: LaneReference) -> dict[str, Any]:
    return {
        "name": reference.name,
        "near_x": reference.near_x,
        "far_x": reference.far_x,
        "near_y": reference.near_y,
        "far_y": reference.far_y,
        "width_px": reference.width_px,
        "fit": _array_to_list(reference.fit),
    }


def _safety_to_dict(obstacle: ObstacleDecision | SafetyDecision | None) -> dict[str, Any]:
    if obstacle is None:
        return {"reason": ""}
    lidar = _lidar_decision(obstacle)
    vision_obstacle = obstacle.vision_obstacle if isinstance(obstacle, SafetyDecision) else None
    return {
        "reason": _obstacle_reason(obstacle),
        "should_stop": bool(getattr(obstacle, "should_stop", False)),
        "speed_scale": float(getattr(obstacle, "speed_scale", 1.0)),
        "nearest_front_mm": None if lidar is None else lidar.nearest_front_mm,
        "lidar_speed_scale": None if lidar is None else lidar.speed_scale,
        "lidar_front_points": None if lidar is None else lidar.front_points,
        "traffic_light_state": getattr(obstacle, "traffic_light_state", ""),
        "traffic_light_area_ratio": getattr(obstacle, "traffic_light_area_ratio", None),
        "vision_obstacle": None if vision_obstacle is None else _object_to_dict(vision_obstacle, (1, 1, 3)),
        "avoidance_steer": float(getattr(obstacle, "avoidance_steer", 0.0)),
        "current_lane_label": getattr(obstacle, "current_lane_label", ""),
        "target_lane_label": getattr(obstacle, "target_lane_label", ""),
    }


def _array_to_list(value) -> list[float] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _lidar_decision(obstacle: ObstacleDecision | SafetyDecision | None) -> ObstacleDecision | None:
    if isinstance(obstacle, SafetyDecision):
        return obstacle.lidar
    if isinstance(obstacle, ObstacleDecision):
        return obstacle
    return None


def _obstacle_reason(obstacle: ObstacleDecision | SafetyDecision | None) -> str:
    if obstacle is None:
        return ""
    if isinstance(obstacle, SafetyDecision):
        return obstacle.reason
    if obstacle.should_stop:
        return "lidar_stop"
    if obstacle.speed_scale < 1.0:
        return "lidar_slow"
    return "clear"


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
