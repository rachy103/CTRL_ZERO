from __future__ import annotations

import csv
import json

import numpy as np

from ctrl_zero.control import DriveCommand
from ctrl_zero.logger import DriveLogger, LogConfig
from ctrl_zero.perception import BoundingBox, DetectedObject
from ctrl_zero.safety import SafetyDecision
from ctrl_zero.vision.base import LaneDetection, LaneReference


def test_drive_logger_creates_run_folder_and_per_frame_detection_logs(tmp_path):
    car = DetectedObject("car", 0.91, BoundingBox(4, 5, 14, 18), lane_label="lane1")
    traffic_light = DetectedObject("traffic_light", 0.88, BoundingBox(1, 1, 5, 6))
    lane = LaneDetection(
        lanes=[[(1, 2), (3, 4)]],
        left_fit=None,
        right_fit=None,
        lane_center_near_x=100.0,
        lane_center_far_x=110.0,
        frame_center_x=100.0,
        offset_px=0.0,
        offset_norm=0.0,
        heading_deg=1.5,
        lane_width_px=80.0,
        confidence=0.92,
        mask=None,
        annotated=np.zeros((20, 20, 3), dtype=np.uint8),
        objects=[car, traffic_light],
        lane_label="lane1",
        lane_references={
            "lane1": LaneReference("lane1", near_x=60.0, far_x=70.0, near_y=19, far_y=10, width_px=50.0),
            "lane2": LaneReference("lane2", near_x=140.0, far_x=130.0, near_y=19, far_y=10, width_px=50.0),
        },
        lane_pair_label="target=lane1",
        traffic_light_state="red",
    )
    safety = SafetyDecision(
        speed_scale=1.0,
        should_stop=False,
        reason="vision_obstacle_lane_change_lane1_to_lane2",
        vision_obstacle=car,
        avoidance_steer=80.0,
        current_lane_label="lane1",
        target_lane_label="lane2",
    )
    logger = DriveLogger(LogConfig(enabled=True, directory=tmp_path, run_name="run", save_every_n_frames=99))

    logger.open()
    logger.log(
        np.zeros((20, 20, 3), dtype=np.uint8),
        "auto",
        "yolo",
        lane,
        safety,
        DriveCommand(steer=80, speed=255, reason="vision_obstacle_lane_change_lane1_to_lane2"),
    )
    logger.close()

    run_dir = tmp_path / "run"
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "drive_log.csv").exists()
    assert (run_dir / "objects.csv").exists()
    assert (run_dir / "lane_references.csv").exists()
    assert (run_dir / "frames.jsonl").exists()

    frame_rows = list(csv.DictReader((run_dir / "drive_log.csv").open(encoding="utf-8")))
    assert frame_rows[0]["object_count"] == "2"
    assert frame_rows[0]["car_count"] == "1"
    assert frame_rows[0]["current_lane"] == "lane1"
    assert frame_rows[0]["safety_reason"] == "vision_obstacle_lane_change_lane1_to_lane2"
    assert frame_rows[0]["avoidance_target_lane"] == "lane2"

    object_rows = list(csv.DictReader((run_dir / "objects.csv").open(encoding="utf-8")))
    assert [row["class_name"] for row in object_rows] == ["car", "traffic_light"]
    assert object_rows[0]["lane_label"] == "lane1"
    assert object_rows[0]["is_vision_obstacle"] == "True"

    lane_ref_rows = list(csv.DictReader((run_dir / "lane_references.csv").open(encoding="utf-8")))
    assert [row["name"] for row in lane_ref_rows] == ["lane1", "lane2"]

    jsonl_record = json.loads((run_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert jsonl_record["lane"]["current_lane"] == "lane1"
    assert jsonl_record["objects"][0]["class_name"] == "car"
    assert jsonl_record["command"]["steer"] == 80
