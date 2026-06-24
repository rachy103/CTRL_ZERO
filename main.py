from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from ctrl_zero.arduino import ArduinoConfig, ArduinoMotorController, format_serial_ports
from ctrl_zero.camera import CameraConfig, CameraReader
from ctrl_zero.common import clamp
from ctrl_zero.control import DriveConfig, DriveController
from ctrl_zero.lidar import LidarConfig, LidarReader, ObstacleDecision, analyze_obstacles
from ctrl_zero.logger import DriveLogger, LogConfig
from ctrl_zero.ui import draw_status
from ctrl_zero.vision.classical_lane import ClassicalLaneConfig, ClassicalLaneDetector


# =============================================================================
# USER TUNING PARAMETERS
# 이 영역만 수정해도 카메라, 라이다, Arduino, 모델, 조향/속도 튜닝을 바꿀 수 있게 둡니다.
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

# Runtime mode: "vision"은 모터 미출력, "manual"은 키보드 수동, "auto"는 차선+라이다 자동 주행입니다.
RUN_MODE = "auto"
LANE_BACKEND = "ufldv2"  # "ufldv2" 또는 "opencv"

# Camera
CAMERA_INDEX = 1
CAMERA_BACKEND = "dshow"  # Windows: "dshow" 권장. 필요 시 "msmf" 또는 "any".
CAMERA_WIDTH = 0
CAMERA_HEIGHT = 0
CAMERA_FPS = 0

# Arduino / motor
USE_ARDUINO = True
ARDUINO_PORT = "auto"
ARDUINO_BAUDRATE = 9600
DRIVE_MAX_PWM = 160

# LiDAR
USE_LIDAR = False
LIDAR_PORT = None  # 예: "COM5"
LIDAR_POLL_EVERY_N_FRAMES = 2
LIDAR_FRONT_MIN_ANGLE_DEG = 330.0
LIDAR_FRONT_MAX_ANGLE_DEG = 30.0
LIDAR_STOP_DISTANCE_MM = 450.0
LIDAR_SLOW_DISTANCE_MM = 900.0
LIDAR_MIN_SPEED_SCALE = 0.35

# Lane model: 기존 Tusimple ResNet18 대신 CULane ResNet34를 기본값으로 사용합니다.
UFLDV2_REPO_DIR = BASE_DIR / "third_party" / "ultra-fast-lane-detection-v2"
UFLDV2_CONFIG_PATH = UFLDV2_REPO_DIR / "configs" / "culane_res34.py"
UFLDV2_MODEL_PATH = BASE_DIR / "models" / "ufldv2" / "culane_res34.pth"
UFLDV2_DEVICE = "cpu"
UFLDV2_TORCH_THREADS = 4
UFLDV2_SHOW_RAW_POINTS = True

# OpenCV fallback lane detector
OPENCV_RESIZE_WIDTH = 640
OPENCV_ROI_TOP_RATIO = 0.58
OPENCV_CANNY_LOW = 50
OPENCV_CANNY_HIGH = 150
OPENCV_HOUGH_THRESHOLD = 35

# Shared lane geometry
DEFAULT_LANE_WIDTH_RATIO = 0.48
MIN_LANE_WIDTH_RATIO = 0.22
MAX_LANE_WIDTH_RATIO = 0.88

# Driving controller. Positive steer means right.
BASE_SPEED = 45
MAX_SPEED = 80
MIN_LANE_CONFIDENCE_TO_DRIVE = 0.45
KP_OFFSET = 78.0
KP_HEADING = 32.0
KD_OFFSET = 18.0
HEADING_NORM_DEG = 28.0
MAX_STEER = 100
REVERSE_STEER = False
CURVE_SPEED_REDUCTION = 0.55

# Manual mode
MANUAL_SPEED_STEP = 10
MANUAL_STEER_POWER = 80
MANUAL_STEER_HOLD_MS = 180

# Display/logging
DISPLAY_ENABLED = True
SHOW_MASK_WINDOW = False
LOG_ENABLED = False
LOG_DIR = BASE_DIR / "lane_logs"
SAVE_EVERY_N_FRAMES = 5
PRINT_EVERY_N_FRAMES = 15


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CTRL_ZERO camera/LiDAR/Arduino autonomous driving runtime.")
    parser.add_argument("--mode", choices=("vision", "manual", "auto"), default=RUN_MODE)
    parser.add_argument("--backend", choices=("ufldv2", "opencv"), default=LANE_BACKEND)
    parser.add_argument("--camera-index", type=int, default=CAMERA_INDEX)
    parser.add_argument("--camera-backend", choices=("dshow", "msmf", "any"), default=CAMERA_BACKEND)
    parser.add_argument("--camera-width", type=int, default=CAMERA_WIDTH)
    parser.add_argument("--camera-height", type=int, default=CAMERA_HEIGHT)
    parser.add_argument("--camera-fps", type=int, default=CAMERA_FPS)
    parser.add_argument("--arduino-port", default=ARDUINO_PORT)
    parser.add_argument("--no-arduino", action="store_true")
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--log", action="store_true")
    return parser


def build_lane_detector(backend: str):
    if backend == "opencv":
        return ClassicalLaneDetector(
            ClassicalLaneConfig(
                resize_width=OPENCV_RESIZE_WIDTH,
                roi_top_ratio=OPENCV_ROI_TOP_RATIO,
                canny_low=OPENCV_CANNY_LOW,
                canny_high=OPENCV_CANNY_HIGH,
                hough_threshold=OPENCV_HOUGH_THRESHOLD,
                default_lane_width_ratio=DEFAULT_LANE_WIDTH_RATIO,
                min_lane_width_ratio=MIN_LANE_WIDTH_RATIO,
                max_lane_width_ratio=MAX_LANE_WIDTH_RATIO,
            )
        )

    from ctrl_zero.vision.ufldv2 import UFLDv2DetectorConfig, UFLDv2LaneDetector

    return UFLDv2LaneDetector(
        UFLDv2DetectorConfig(
            repo_dir=UFLDV2_REPO_DIR,
            config_path=UFLDV2_CONFIG_PATH,
            model_path=UFLDV2_MODEL_PATH,
            device=UFLDV2_DEVICE,
            torch_num_threads=UFLDV2_TORCH_THREADS,
            default_lane_width_ratio=DEFAULT_LANE_WIDTH_RATIO,
            min_lane_width_ratio=MIN_LANE_WIDTH_RATIO,
            max_lane_width_ratio=MAX_LANE_WIDTH_RATIO,
            show_raw_points=UFLDV2_SHOW_RAW_POINTS,
        )
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.list_ports:
        print(format_serial_ports())
        return

    display_enabled = DISPLAY_ENABLED and not args.no_display
    motor_enabled = USE_ARDUINO and not args.no_arduino and args.mode != "vision"
    arduino_port = args.arduino_port
    if motor_enabled and arduino_port in (None, "", "auto"):
        arduino_port = "auto"

    camera = CameraReader(
        CameraConfig(
            index=args.camera_index,
            backend=args.camera_backend,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
        )
    )
    motor = ArduinoMotorController(
        ArduinoConfig(port=arduino_port, baudrate=ARDUINO_BAUDRATE, drive_max_pwm=DRIVE_MAX_PWM),
        dry_run=not motor_enabled,
    )
    controller = DriveController(
        DriveConfig(
            base_speed=BASE_SPEED,
            max_speed=MAX_SPEED,
            min_confidence=MIN_LANE_CONFIDENCE_TO_DRIVE,
            kp_offset=KP_OFFSET,
            kp_heading=KP_HEADING,
            kd_offset=KD_OFFSET,
            heading_norm_deg=HEADING_NORM_DEG,
            max_steer=MAX_STEER,
            reverse_steer=REVERSE_STEER,
            curve_speed_reduction=CURVE_SPEED_REDUCTION,
        )
    )

    lane_detector = build_lane_detector(args.backend)
    lidar_config = LidarConfig(
        port=LIDAR_PORT,
        front_min_angle_deg=LIDAR_FRONT_MIN_ANGLE_DEG,
        front_max_angle_deg=LIDAR_FRONT_MAX_ANGLE_DEG,
        stop_distance_mm=LIDAR_STOP_DISTANCE_MM,
        slow_distance_mm=LIDAR_SLOW_DISTANCE_MM,
        min_speed_scale=LIDAR_MIN_SPEED_SCALE,
    )
    lidar = LidarReader(lidar_config) if USE_LIDAR else None
    last_obstacle = ObstacleDecision.clear() if USE_LIDAR else None
    logger = DriveLogger(LogConfig(enabled=LOG_ENABLED or args.log, directory=LOG_DIR, save_every_n_frames=SAVE_EVERY_N_FRAMES))

    manual_steer = 0
    manual_steer_until = 0.0
    manual_speed = 0
    frame_count = 0
    fps = 0.0
    last_time = time.perf_counter()

    try:
        motor.open()
        camera.open()
        if lidar is not None:
            lidar.open()
        logger.open()

        print("Keys: q quit, space stop, +/- base speed, l toggle log.")
        print("Manual mode keys: w/s speed, a/d steer pulse, c center steer.")
        print(f"Runtime: mode={args.mode}, backend={args.backend}, motor={'on' if motor_enabled else 'dry'}")

        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                print("Camera frame read failed.")
                break

            frame_count += 1
            lane = lane_detector.detect(frame)

            if lidar is not None and frame_count % max(LIDAR_POLL_EVERY_N_FRAMES, 1) == 0:
                try:
                    last_obstacle = analyze_obstacles(lidar.read_scan(), lidar_config)
                except Exception as exc:  # pragma: no cover - hardware dependent
                    print(f"LiDAR read failed: {exc}")
                    last_obstacle = ObstacleDecision.clear()

            if args.mode == "manual" and time.time() > manual_steer_until:
                manual_steer = 0

            command = controller.compute(lane, last_obstacle, args.mode, manual_steer=manual_steer, manual_speed=manual_speed)
            motor.send(command.steer, command.speed if motor_enabled else 0)

            now = time.perf_counter()
            elapsed = now - last_time
            last_time = now
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if elapsed > 0 else fps

            if frame_count % PRINT_EVERY_N_FRAMES == 0:
                print(
                    f"frame={frame_count} fps={fps:.1f} conf={lane.confidence:.2f} "
                    f"steer={command.steer:+d} speed={command.speed:+d} reason={command.reason}",
                    flush=True,
                )

            logger.log(frame, args.mode, args.backend, lane, last_obstacle, command)

            key = -1
            if display_enabled:
                display = lane.annotated
                draw_status(display, lane, last_obstacle, command, args.mode, args.backend, fps, motor_enabled)
                cv2.imshow("CTRL_ZERO", display)
                if SHOW_MASK_WINDOW and lane.mask is not None:
                    cv2.imshow("CTRL_ZERO lane mask", lane.mask)
                key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord(" "):
                manual_speed = 0
                manual_steer = 0
                manual_steer_until = 0.0
                motor.stop()
            elif key in (ord("+"), ord("=")):
                controller.config.base_speed = int(clamp(controller.config.base_speed + 5, 0, MAX_SPEED))
            elif key in (ord("-"), ord("_")):
                controller.config.base_speed = int(clamp(controller.config.base_speed - 5, 0, MAX_SPEED))
            elif key == ord("l"):
                logger.set_enabled(not logger.config.enabled)
                print(f"Logging {'enabled' if logger.config.enabled else 'disabled'}")

            if args.mode == "manual":
                if key == ord("w"):
                    manual_speed = int(clamp(manual_speed + MANUAL_SPEED_STEP, -MAX_SPEED, MAX_SPEED))
                elif key == ord("s"):
                    manual_speed = int(clamp(manual_speed - MANUAL_SPEED_STEP, -MAX_SPEED, MAX_SPEED))
                elif key == ord("a"):
                    manual_steer = -int(clamp(MANUAL_STEER_POWER, 0, MAX_STEER))
                    manual_steer_until = time.time() + MANUAL_STEER_HOLD_MS / 1000.0
                elif key == ord("d"):
                    manual_steer = int(clamp(MANUAL_STEER_POWER, 0, MAX_STEER))
                    manual_steer_until = time.time() + MANUAL_STEER_HOLD_MS / 1000.0
                elif key == ord("c"):
                    manual_steer = 0
                    manual_steer_until = 0.0
    finally:
        motor.close()
        camera.release()
        if lidar is not None:
            lidar.close()
        logger.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
