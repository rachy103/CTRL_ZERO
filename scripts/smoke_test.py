from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ctrl_zero.control import DriveConfig, DriveController
from ctrl_zero.lidar import LidarConfig, analyze_obstacles
from ctrl_zero.vision.classical_lane import ClassicalLaneConfig, ClassicalLaneDetector


def synthetic_lane_frame() -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(frame, (210, 479), (300, 275), (255, 255, 255), 12)
    cv2.line(frame, (430, 479), (340, 275), (255, 255, 255), 12)
    return frame


def main() -> int:
    detector = ClassicalLaneDetector(ClassicalLaneConfig(resize_width=640))
    lane = detector.detect(synthetic_lane_frame())
    controller = DriveController(DriveConfig(base_speed=45, max_speed=80, min_confidence=0.30))
    command = controller.compute(lane, None, "auto")

    scan = np.array([[350.0, 800.0], [10.0, 700.0], [180.0, 250.0]], dtype=np.float32)
    obstacle = analyze_obstacles(scan, LidarConfig(stop_distance_mm=450, slow_distance_mm=900))

    print(f"lane_confidence={lane.confidence:.3f}")
    print(f"lane_offset_norm={lane.offset_norm}")
    print(f"command={command}")
    print(f"lidar={obstacle}")

    if lane.confidence < 0.30 or command.speed <= 0:
        return 1
    if obstacle.should_stop:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
