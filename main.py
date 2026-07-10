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
from ctrl_zero.obstacles import LaneChangeState, VisionObstacleConfig, analyze_vision_obstacles, apply_lane_change_for_obstacle
from ctrl_zero.safety import build_safety_decision
from ctrl_zero.traffic_light import traffic_light_object
from ctrl_zero.ui import draw_status
from ctrl_zero.vision.preprocess import BirdEyeConfig, LanePreprocessor, ROICropConfig


# =============================================================================
# USER TUNING PARAMETERS
# 이 영역만 수정해도 카메라, 라이다, Arduino, 모델, 조향/속도 튜닝을 바꿀 수 있게 둡니다.
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

# Runtime mode: "vision"은 모터 미출력, "manual"은 키보드 수동, "auto"는 차선+라이다 자동 주행입니다.
RUN_MODE = "auto"
LANE_BACKEND = "yolo"

# Camera
CAMERA_INDEX = 0
CAMERA_BACKEND = "dshow"  # Windows: "dshow" 권장. 필요 시 "msmf" 또는 "any".
CAMERA_WIDTH = 0
CAMERA_HEIGHT = 0
CAMERA_FPS = 0

# Arduino / motor
USE_ARDUINO = True
ARDUINO_PORT = "auto"
ARDUINO_BAUDRATE = 9600
DRIVE_MAX_PWM = 255

# LiDAR
USE_LIDAR = False
LIDAR_PORT = None  # 예: "COM5"
LIDAR_POLL_EVERY_N_FRAMES = 2
LIDAR_FRONT_MIN_ANGLE_DEG = 210.0
LIDAR_FRONT_MAX_ANGLE_DEG = 150.0
LIDAR_STOP_DISTANCE_MM = 450.0
LIDAR_SLOW_DISTANCE_MM = 900.0
LIDAR_MIN_SPEED_SCALE = 0.35

# Vision obstacles
VISION_OBSTACLE_ENABLED = True
VISION_OBSTACLE_MIN_CONFIDENCE = 0.4
VISION_OBSTACLE_LANE_CHANGE_AREA_RATIO = 0.0
VISION_OBSTACLE_AVOIDANCE_STEER_WEIGHT = 15.0
VISION_OBSTACLE_AVOIDANCE_STEER_LIMIT = 80.0
VISION_OBSTACLE_LANE_CHANGE_COMPLETE_OFFSET_NORM = 0.10
VISION_OBSTACLE_LANE_CHANGE_COMPLETE_FRAMES = 2

# Traffic light stop gating. Stop only when red/yellow bbox area reaches this frame-area ratio.
TRAFFIC_LIGHT_STOP_AREA_RATIO = 0.058

# YOLO lane model. Use a lane-trained Ultralytics .pt file, not a generic COCO model.
YOLO_MODEL_PATH = BASE_DIR / "models" / "yolo" / "final.pt"
YOLO_DEVICE = "cpu"
YOLO_IMAGE_SIZE = 640
YOLO_CONFIDENCE = 0.25
YOLO_IOU = 0.45
YOLO_CLASS_NAMES = ("car", "obstacle", "lane1", "lane2", "traffic_light")
YOLO_FRAME_SKIP = 1
YOLO_MIN_POINTS_PER_LANE = 3
YOLO_MIN_VALID_Y_SPAN_RATIO = 0.08
YOLO_MASK_SAMPLE_STEP_PX = 6
YOLO_SKELETON_ENABLED = True
YOLO_SKELETON_BRIDGE_GAP_PX = 17
YOLO_DASHED_MERGE_MAX_X_GAP_RATIO = 0.12
YOLO_CURVE_FIT_DEGREE = 2
YOLO_SOLIDIFY_STEP_PX = 6
YOLO_FIT_OUTLIER_REJECTION_PX = 35.0
YOLO_SEGMENTATION_MODE = "lane_area"  # "lane_area" follows lane1/lane2 area centers. "drivable_area" follows one road mask.
YOLO_DRIVABLE_MIN_ROW_WIDTH_RATIO = 0.05
YOLO_DRIVABLE_EDGE_PERCENTILE = 2.0
YOLO_TARGET_LANE_PAIR = "right"  # "right"=right lane, "left"=left lane, "closest"=nearest pair.
YOLO_TARGET_PATH_MODE = "closest_line"  # "closest_line" follows one line. "lane_center" follows the lane pair center.
YOLO_LANE_PAIR_SELECT_Y_RATIO = 0.78
YOLO_LANE_PAIR_TARGET_OFFSET_RATIO = -0.03
YOLO_DISPLAY_BIRD_EYE_VIEW = True

# Camera/IPM preprocessing. Bird-eye source points are image ratios in this order:
# bottom-left, bottom-right, top-right, top-left.
ROI_ENABLED = False
ROI_TOP_RATIO = 0.0
ROI_BOTTOM_RATIO = 1.0
ROI_LEFT_RATIO = 0.0
ROI_RIGHT_RATIO = 1.0
BIRD_EYE_ENABLED = False
BIRD_EYE_SRC_BOTTOM_LEFT = (0.10, 0.98)
BIRD_EYE_SRC_BOTTOM_RIGHT = (0.90, 0.98)
BIRD_EYE_SRC_TOP_RIGHT = (0.62, 0.35)
BIRD_EYE_SRC_TOP_LEFT = (0.38, 0.35)
BIRD_EYE_DST_MARGIN_RATIO = 0.18
BIRD_EYE_MASK_SOURCE_POLYGON = True

# Shared lane geometry
DEFAULT_LANE_WIDTH_RATIO = 0.50
MIN_LANE_WIDTH_RATIO = 0.35
MAX_LANE_WIDTH_RATIO = 0.65

# Driving controller. Positive steer means right.
CONTROL_MODE = "contest"  # "contest" uses angle/position weights.
MIN_SPEED = 230
MAX_SPEED = 255
MIN_LANE_CONFIDENCE_TO_DRIVE = 0.45
CONTEST_ANGLE_WEIGHT = 0.45
CONTEST_POSITION_WEIGHT = 0.10
CONTEST_STEERING_ANGLE_NORM_DEG = 50.0
CONTEST_STEER_LIMIT = 10.0

# Common steering limits & safety
MAX_STEER = 80
REVERSE_STEER = False
MAX_HOLD_FRAMES = 6
HOLD_DECEL_STEP = 6

# Manual mode
MANUAL_SPEED_STEP = 10
MANUAL_STEER_POWER = 80
MANUAL_STEER_HOLD_MS = 180

# Display/logging
DISPLAY_ENABLED = True
SHOW_MASK_WINDOW = False
LOG_ENABLED = False
LOG_DIR = BASE_DIR / "Log"
SAVE_EVERY_N_FRAMES = 5
PRINT_EVERY_N_FRAMES = 15


def parse_ratio_point(value: str) -> tuple[float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected x,y ratio pair, for example 0.38,0.35")
    try:
        x_ratio = float(parts[0])
        y_ratio = float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ratio point values must be numbers") from exc
    if not 0.0 <= x_ratio <= 1.0 or not 0.0 <= y_ratio <= 1.0:
        raise argparse.ArgumentTypeError("ratio point values must be between 0.0 and 1.0")
    return x_ratio, y_ratio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CTRL_ZERO camera/LiDAR/Arduino autonomous driving runtime.")
    parser.add_argument("--mode", choices=("vision", "manual", "auto"), default=RUN_MODE)
    parser.add_argument("--backend", choices=("yolo",), default=LANE_BACKEND)
    parser.add_argument("--yolo-model", type=Path, default=YOLO_MODEL_PATH)
    parser.add_argument("--yolo-frame-skip", type=int, default=YOLO_FRAME_SKIP)
    parser.add_argument("--yolo-imgsz", type=int, default=YOLO_IMAGE_SIZE)
    parser.add_argument("--yolo-conf", type=float, default=YOLO_CONFIDENCE)
    parser.add_argument("--yolo-iou", type=float, default=YOLO_IOU)
    parser.add_argument("--yolo-classes", default=",".join(YOLO_CLASS_NAMES))
    parser.add_argument("--yolo-min-points", type=int, default=YOLO_MIN_POINTS_PER_LANE)
    parser.add_argument("--yolo-min-y-span", type=float, default=YOLO_MIN_VALID_Y_SPAN_RATIO)
    parser.add_argument("--yolo-mask-step", type=int, default=YOLO_MASK_SAMPLE_STEP_PX)
    parser.add_argument("--yolo-skeleton", action=argparse.BooleanOptionalAction, default=YOLO_SKELETON_ENABLED)
    parser.add_argument("--yolo-bridge-gap", type=int, default=YOLO_SKELETON_BRIDGE_GAP_PX)
    parser.add_argument("--yolo-merge-gap-ratio", type=float, default=YOLO_DASHED_MERGE_MAX_X_GAP_RATIO)
    parser.add_argument("--yolo-curve-degree", type=int, default=YOLO_CURVE_FIT_DEGREE)
    parser.add_argument("--yolo-solid-step", type=int, default=YOLO_SOLIDIFY_STEP_PX)
    parser.add_argument("--yolo-fit-outlier-px", type=float, default=YOLO_FIT_OUTLIER_REJECTION_PX)
    parser.add_argument("--yolo-segmentation-mode", choices=("lane_area", "drivable_area", "lane_lines"), default=YOLO_SEGMENTATION_MODE)
    parser.add_argument("--yolo-drivable-min-row-width", type=float, default=YOLO_DRIVABLE_MIN_ROW_WIDTH_RATIO)
    parser.add_argument("--yolo-drivable-edge-percentile", type=float, default=YOLO_DRIVABLE_EDGE_PERCENTILE)
    parser.add_argument("--yolo-target-lane-pair", choices=("right", "left", "closest", "center", "split"), default=YOLO_TARGET_LANE_PAIR)
    parser.add_argument("--yolo-target-path", choices=("closest_line", "left_line", "right_line", "lane_center"), default=YOLO_TARGET_PATH_MODE)
    parser.add_argument("--yolo-pair-select-y", type=float, default=YOLO_LANE_PAIR_SELECT_Y_RATIO)
    parser.add_argument("--yolo-pair-target-offset", type=float, default=YOLO_LANE_PAIR_TARGET_OFFSET_RATIO)
    parser.add_argument("--yolo-display-bird-eye", action=argparse.BooleanOptionalAction, default=YOLO_DISPLAY_BIRD_EYE_VIEW)
    parser.add_argument("--roi", action=argparse.BooleanOptionalAction, default=ROI_ENABLED)
    parser.add_argument("--roi-top", type=float, default=ROI_TOP_RATIO)
    parser.add_argument("--roi-bottom", type=float, default=ROI_BOTTOM_RATIO)
    parser.add_argument("--roi-left", type=float, default=ROI_LEFT_RATIO)
    parser.add_argument("--roi-right", type=float, default=ROI_RIGHT_RATIO)
    parser.add_argument("--bird-eye", action=argparse.BooleanOptionalAction, default=BIRD_EYE_ENABLED)
    parser.add_argument("--bird-eye-src-bottom-left", type=parse_ratio_point, default=BIRD_EYE_SRC_BOTTOM_LEFT)
    parser.add_argument("--bird-eye-src-bottom-right", type=parse_ratio_point, default=BIRD_EYE_SRC_BOTTOM_RIGHT)
    parser.add_argument("--bird-eye-src-top-right", type=parse_ratio_point, default=BIRD_EYE_SRC_TOP_RIGHT)
    parser.add_argument("--bird-eye-src-top-left", type=parse_ratio_point, default=BIRD_EYE_SRC_TOP_LEFT)
    parser.add_argument("--bird-eye-dst-margin", type=float, default=BIRD_EYE_DST_MARGIN_RATIO)
    parser.add_argument("--bird-eye-mask-source-polygon", action=argparse.BooleanOptionalAction, default=BIRD_EYE_MASK_SOURCE_POLYGON)
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
    parser.add_argument("--no-log", action="store_true")
    return parser


def build_preprocessor(args) -> LanePreprocessor:
    return LanePreprocessor(
        roi=ROICropConfig(
            enabled=args.roi,
            top_ratio=args.roi_top,
            bottom_ratio=args.roi_bottom,
            left_ratio=args.roi_left,
            right_ratio=args.roi_right,
        ),
        bird_eye=BirdEyeConfig(
            enabled=args.bird_eye,
            src_bottom_left=args.bird_eye_src_bottom_left,
            src_bottom_right=args.bird_eye_src_bottom_right,
            src_top_right=args.bird_eye_src_top_right,
            src_top_left=args.bird_eye_src_top_left,
            dst_margin_ratio=args.bird_eye_dst_margin,
            mask_source_polygon=args.bird_eye_mask_source_polygon,
        ),
    )


def build_lane_detector(args):
    from ctrl_zero.vision.yolo_lane import YOLOLaneConfig, YOLOLaneDetector
    from ctrl_zero.vision.cache import FrameSkippingLaneDetector

    yolo_classes = tuple(item.strip() for item in args.yolo_classes.split(",") if item.strip())
    detector = YOLOLaneDetector(
        YOLOLaneConfig(
            model_path=args.yolo_model,
            device=YOLO_DEVICE,
            image_size=args.yolo_imgsz,
            confidence=args.yolo_conf,
            iou=args.yolo_iou,
            class_names=yolo_classes,
            min_points_per_lane=args.yolo_min_points,
            min_valid_y_span_ratio=args.yolo_min_y_span,
            mask_sample_step_px=args.yolo_mask_step,
            skeleton_enabled=args.yolo_skeleton,
            skeleton_bridge_gap_px=args.yolo_bridge_gap,
            dashed_merge_max_x_gap_ratio=args.yolo_merge_gap_ratio,
            curve_fit_degree=args.yolo_curve_degree,
            solidify_step_px=args.yolo_solid_step,
            fit_outlier_rejection_px=args.yolo_fit_outlier_px,
            segmentation_mode=args.yolo_segmentation_mode,
            drivable_min_row_width_ratio=args.yolo_drivable_min_row_width,
            drivable_edge_percentile=args.yolo_drivable_edge_percentile,
            target_lane_pair=args.yolo_target_lane_pair,
            target_path_mode=args.yolo_target_path,
            lane_pair_select_y_ratio=args.yolo_pair_select_y,
            lane_pair_target_offset_ratio=args.yolo_pair_target_offset,
            default_lane_width_ratio=DEFAULT_LANE_WIDTH_RATIO,
            min_lane_width_ratio=MIN_LANE_WIDTH_RATIO,
            max_lane_width_ratio=MAX_LANE_WIDTH_RATIO,
            display_bird_eye_view=args.yolo_display_bird_eye,
            preprocessor=build_preprocessor(args),
        )
    )
    return FrameSkippingLaneDetector(detector, skip=args.yolo_frame_skip) if args.yolo_frame_skip > 1 else detector


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
            control_mode=CONTROL_MODE,
            min_speed=MIN_SPEED,
            max_speed=MAX_SPEED,
            contest_angle_weight=CONTEST_ANGLE_WEIGHT,
            contest_position_weight=CONTEST_POSITION_WEIGHT,
            contest_angle_norm_deg=CONTEST_STEERING_ANGLE_NORM_DEG,
            contest_steer_limit=CONTEST_STEER_LIMIT,
            min_confidence=MIN_LANE_CONFIDENCE_TO_DRIVE,
            max_steer=MAX_STEER,
            reverse_steer=REVERSE_STEER,
            max_hold_frames=MAX_HOLD_FRAMES,
            hold_decel_step=HOLD_DECEL_STEP,
        )
    )

    lane_detector = build_lane_detector(args)
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
    vision_obstacle_config = VisionObstacleConfig(
        enabled=VISION_OBSTACLE_ENABLED,
        min_confidence=VISION_OBSTACLE_MIN_CONFIDENCE,
        lane_change_area_ratio=VISION_OBSTACLE_LANE_CHANGE_AREA_RATIO,
        avoidance_steer_weight=VISION_OBSTACLE_AVOIDANCE_STEER_WEIGHT,
        avoidance_steer_limit=VISION_OBSTACLE_AVOIDANCE_STEER_LIMIT,
        lane_change_complete_offset_norm=VISION_OBSTACLE_LANE_CHANGE_COMPLETE_OFFSET_NORM,
        lane_change_complete_frames=VISION_OBSTACLE_LANE_CHANGE_COMPLETE_FRAMES,
    )
    lane_change_state = LaneChangeState()
    logger = DriveLogger(
        LogConfig(
            enabled=(LOG_ENABLED or args.log) and not args.no_log,
            directory=LOG_DIR,
            save_every_n_frames=SAVE_EVERY_N_FRAMES,
        )
    )

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

        print("Keys: q quit, space stop, +/- max speed, l toggle log.")
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

            vision_obstacle = analyze_vision_obstacles(lane, vision_obstacle_config)
            lane_for_control, vision_obstacle = apply_lane_change_for_obstacle(
                lane,
                vision_obstacle,
                vision_obstacle_config,
                lane_change_state,
            )
            traffic_obj = traffic_light_object(lane_for_control.objects)
            frame_area = float(max(lane_for_control.annotated.shape[0] * lane_for_control.annotated.shape[1], 1))
            safety = build_safety_decision(
                lidar=last_obstacle,
                traffic_light_state=lane_for_control.traffic_light_state,
                traffic_light_object=traffic_obj,
                traffic_light_frame_area=frame_area,
                traffic_light_min_stop_area_ratio=TRAFFIC_LIGHT_STOP_AREA_RATIO,
                vision_obstacle_decision=vision_obstacle,
            )
            command = controller.compute(lane_for_control, safety, args.mode, manual_steer=manual_steer, manual_speed=manual_speed)
            motor.send(command.steer, command.speed if motor_enabled else 0)

            now = time.perf_counter()
            elapsed = now - last_time
            last_time = now
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if elapsed > 0 else fps

            if frame_count % PRINT_EVERY_N_FRAMES == 0:
                print(
                    f"frame={frame_count} fps={fps:.1f} conf={lane_for_control.confidence:.2f} "
                    f"steer={command.steer:+d} speed={command.speed:+d} reason={command.reason}",
                    flush=True,
                )

            logger.log(frame, args.mode, args.backend, lane_for_control, safety, command)

            key = -1
            if display_enabled:
                display = lane_for_control.annotated
                draw_status(display, lane_for_control, safety, command, args.mode, args.backend, fps, motor_enabled)
                cv2.imshow("CTRL_ZERO", display)
                if SHOW_MASK_WINDOW and lane_for_control.mask is not None:
                    cv2.imshow("CTRL_ZERO lane mask", lane_for_control.mask)
                key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord(" "):
                manual_speed = 0
                manual_steer = 0
                manual_steer_until = 0.0
                motor.stop()
            elif key in (ord("+"), ord("=")):
                controller.config.max_speed = int(clamp(controller.config.max_speed + 5, 0, MAX_SPEED))
            elif key in (ord("-"), ord("_")):
                controller.config.max_speed = int(clamp(controller.config.max_speed - 5, MIN_SPEED, MAX_SPEED))
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
