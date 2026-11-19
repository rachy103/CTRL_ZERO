from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

import main as runtime
from ctrl_zero.arduino import ArduinoConfig, ArduinoMotorController, format_serial_ports
from ctrl_zero.camera import CameraConfig, CameraReader
from ctrl_zero.common import clamp
from ctrl_zero.control import DriveCommand, DriveConfig, DriveController
from ctrl_zero.lidar import LidarConfig, LidarReader, ObstacleDecision, analyze_obstacles
from ctrl_zero.logger import DriveLogger, LogConfig
from ctrl_zero.ui import draw_status
from ctrl_zero.vision.base import LaneDetection


# =============================================================================
# DATASET RECORDING PARAMETERS
# =============================================================================

RECORD_DIR = runtime.BASE_DIR / "captures" / "lane_dataset"
RECORD_INTERVAL_SEC = 0.5
RECORD_IMAGE_QUALITY = 95
RECORD_ANNOTATED = False


@dataclass
class DatasetRecordConfig:
    enabled: bool = True
    directory: Path = RECORD_DIR
    interval_sec: float = RECORD_INTERVAL_SEC
    image_quality: int = RECORD_IMAGE_QUALITY
    save_annotated: bool = RECORD_ANNOTATED
    session_name: str | None = None


class DatasetRecorder:
    def __init__(self, config: DatasetRecordConfig):
        self.config = config
        self.session_dir: Path | None = None
        self.image_dir: Path | None = None
        self.annotated_dir: Path | None = None
        self.csv_file = None
        self.writer = None
        self.saved_count = 0
        self.last_saved_at: float | None = None

    def open(self) -> None:
        if not self.config.enabled:
            return

        session_name = self.config.session_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.config.directory / session_name
        self.image_dir = self.session_dir / "images"
        self.annotated_dir = self.session_dir / "annotated"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_annotated:
            self.annotated_dir.mkdir(parents=True, exist_ok=True)

        self.csv_file = (self.session_dir / "dataset.csv").open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "timestamp",
                "epoch_sec",
                "frame",
                "image_file",
                "annotated_file",
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
                "nearest_front_mm",
                "lidar_speed_scale",
                "lidar_front_points",
            ],
        )
        self.writer.writeheader()
        print(f"Dataset recording enabled: {self.session_dir}")

    def maybe_record(
        self,
        frame,
        frame_count: int,
        mode: str,
        backend: str,
        lane: LaneDetection,
        obstacle: ObstacleDecision | None,
        command: DriveCommand,
    ) -> None:
        if not self.config.enabled or self.writer is None or self.image_dir is None:
            return

        now = time.monotonic()
        interval_sec = max(self.config.interval_sec, 0.001)
        if self.last_saved_at is not None and now - self.last_saved_at < interval_sec:
            return

        timestamp = time.time()
        stamp = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S_%f")[:-3]
        image_file = f"{stamp}_frame_{frame_count:06d}.jpg"
        image_path = self.image_dir / image_file
        quality = int(clamp(self.config.image_quality, 1, 100))
        image_ok = cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not image_ok:
            print(f"Dataset image save failed: {image_path}")
            return

        annotated_file = ""
        if self.config.save_annotated and self.annotated_dir is not None:
            annotated_file = f"{stamp}_frame_{frame_count:06d}_annotated.jpg"
            annotated_path = self.annotated_dir / annotated_file
            annotated_ok = cv2.imwrite(str(annotated_path), lane.annotated, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not annotated_ok:
                print(f"Dataset annotated image save failed: {annotated_path}")
                annotated_file = ""

        self.saved_count += 1
        self.last_saved_at = now
        self.writer.writerow(
            {
                "timestamp": datetime.fromtimestamp(timestamp).isoformat(timespec="milliseconds"),
                "epoch_sec": f"{timestamp:.6f}",
                "frame": frame_count,
                "image_file": f"images/{image_file}",
                "annotated_file": "" if not annotated_file else f"annotated/{annotated_file}",
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
                "nearest_front_mm": "" if obstacle is None or obstacle.nearest_front_mm is None else f"{obstacle.nearest_front_mm:.1f}",
                "lidar_speed_scale": "" if obstacle is None else f"{obstacle.speed_scale:.3f}",
                "lidar_front_points": "" if obstacle is None else obstacle.front_points,
            }
        )
        self.csv_file.flush()

    def set_enabled(self, enabled: bool) -> None:
        if enabled and self.writer is None:
            self.config.enabled = True
            self.open()
            return
        self.config.enabled = enabled

    def close(self) -> None:
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.writer = None


def build_parser() -> argparse.ArgumentParser:
    parser = runtime.build_parser()
    parser.description = "CTRL_ZERO autonomous driving runtime with 1 Hz lane dataset recording."
    parser.add_argument("--record-dir", type=Path, default=RECORD_DIR, help="Base directory for recorded dataset sessions.")
    parser.add_argument("--record-session", default=None, help="Session folder name. Defaults to YYYYMMDD_HHMMSS.")
    parser.add_argument("--record-interval", type=float, default=RECORD_INTERVAL_SEC, help="Seconds between saved frames.")
    parser.add_argument("--record-quality", type=int, default=RECORD_IMAGE_QUALITY, help="JPEG quality from 1 to 100.")
    parser.add_argument("--record-annotated", action=argparse.BooleanOptionalAction, default=RECORD_ANNOTATED)
    parser.add_argument("--no-record", action="store_true", help="Run without saving dataset images.")
    return parser


def resolve_record_dir(path: Path) -> Path:
    return path if path.is_absolute() else runtime.BASE_DIR / path


def main() -> None:
    args = build_parser().parse_args()
    if args.list_ports:
        print(format_serial_ports())
        return

    display_enabled = runtime.DISPLAY_ENABLED and not args.no_display
    motor_enabled = runtime.USE_ARDUINO and not args.no_arduino and args.mode != "vision"
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
        ArduinoConfig(port=arduino_port, baudrate=runtime.ARDUINO_BAUDRATE, drive_max_pwm=runtime.DRIVE_MAX_PWM),
        dry_run=not motor_enabled,
    )
    controller = DriveController(
        DriveConfig(
            control_mode=runtime.CONTROL_MODE,
            min_speed=runtime.MIN_SPEED,
            max_speed=runtime.MAX_SPEED,
            contest_angle_weight=runtime.CONTEST_ANGLE_WEIGHT,
            contest_position_weight=runtime.CONTEST_POSITION_WEIGHT,
            contest_angle_norm_deg=runtime.CONTEST_STEERING_ANGLE_NORM_DEG,
            contest_steer_limit=runtime.CONTEST_STEER_LIMIT,
            min_confidence=runtime.MIN_LANE_CONFIDENCE_TO_DRIVE,
            max_steer=runtime.MAX_STEER,
            reverse_steer=runtime.REVERSE_STEER,
            max_hold_frames=runtime.MAX_HOLD_FRAMES,
            hold_decel_step=runtime.HOLD_DECEL_STEP,
        )
    )

    lane_detector = runtime.build_lane_detector(args)
    lidar_config = LidarConfig(
        port=runtime.LIDAR_PORT,
        front_min_angle_deg=runtime.LIDAR_FRONT_MIN_ANGLE_DEG,
        front_max_angle_deg=runtime.LIDAR_FRONT_MAX_ANGLE_DEG,
        stop_distance_mm=runtime.LIDAR_STOP_DISTANCE_MM,
        slow_distance_mm=runtime.LIDAR_SLOW_DISTANCE_MM,
        min_speed_scale=runtime.LIDAR_MIN_SPEED_SCALE,
    )
    lidar = LidarReader(lidar_config) if runtime.USE_LIDAR else None
    last_obstacle = ObstacleDecision.clear() if runtime.USE_LIDAR else None
    logger = DriveLogger(
        LogConfig(
            enabled=runtime.LOG_ENABLED or args.log,
            directory=runtime.LOG_DIR,
            save_every_n_frames=runtime.SAVE_EVERY_N_FRAMES,
        )
    )
    recorder = DatasetRecorder(
        DatasetRecordConfig(
            enabled=not args.no_record,
            directory=resolve_record_dir(args.record_dir),
            interval_sec=args.record_interval,
            image_quality=args.record_quality,
            save_annotated=args.record_annotated,
            session_name=args.record_session,
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
        recorder.open()

        print("Keys: q quit, space stop, +/- max speed, l toggle log, r toggle dataset record.")
        print("Manual mode keys: w/s speed, a/d steer pulse, c center steer.")
        print(f"Runtime: mode={args.mode}, backend={args.backend}, motor={'on' if motor_enabled else 'dry'}")

        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                print("Camera frame read failed.")
                break

            frame_count += 1
            lane = lane_detector.detect(frame)

            if lidar is not None and frame_count % max(runtime.LIDAR_POLL_EVERY_N_FRAMES, 1) == 0:
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

            if frame_count % runtime.PRINT_EVERY_N_FRAMES == 0:
                print(
                    f"frame={frame_count} fps={fps:.1f} conf={lane.confidence:.2f} "
                    f"steer={command.steer:+d} speed={command.speed:+d} reason={command.reason} "
                    f"recorded={recorder.saved_count}",
                    flush=True,
                )

            logger.log(frame, args.mode, args.backend, lane, last_obstacle, command)
            recorder.maybe_record(frame, frame_count, args.mode, args.backend, lane, last_obstacle, command)

            key = -1
            if display_enabled:
                display = lane.annotated
                draw_status(display, lane, last_obstacle, command, args.mode, args.backend, fps, motor_enabled)
                cv2.imshow("CTRL_ZERO", display)
                if runtime.SHOW_MASK_WINDOW and lane.mask is not None:
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
                controller.config.max_speed = int(clamp(controller.config.max_speed + 5, 0, runtime.MAX_SPEED))
            elif key in (ord("-"), ord("_")):
                controller.config.max_speed = int(clamp(controller.config.max_speed - 5, runtime.MIN_SPEED, runtime.MAX_SPEED))
            elif key == ord("l"):
                logger.set_enabled(not logger.config.enabled)
                print(f"Logging {'enabled' if logger.config.enabled else 'disabled'}")
            elif key == ord("r"):
                recorder.set_enabled(not recorder.config.enabled)
                print(f"Dataset recording {'enabled' if recorder.config.enabled else 'disabled'}")

            if args.mode == "manual":
                if key == ord("w"):
                    manual_speed = int(clamp(manual_speed + runtime.MANUAL_SPEED_STEP, -runtime.MAX_SPEED, runtime.MAX_SPEED))
                elif key == ord("s"):
                    manual_speed = int(clamp(manual_speed - runtime.MANUAL_SPEED_STEP, -runtime.MAX_SPEED, runtime.MAX_SPEED))
                elif key == ord("a"):
                    manual_steer = -int(clamp(runtime.MANUAL_STEER_POWER, 0, runtime.MAX_STEER))
                    manual_steer_until = time.time() + runtime.MANUAL_STEER_HOLD_MS / 1000.0
                elif key == ord("d"):
                    manual_steer = int(clamp(runtime.MANUAL_STEER_POWER, 0, runtime.MAX_STEER))
                    manual_steer_until = time.time() + runtime.MANUAL_STEER_HOLD_MS / 1000.0
                elif key == ord("c"):
                    manual_steer = 0
                    manual_steer_until = 0.0
    finally:
        motor.close()
        camera.release()
        if lidar is not None:
            lidar.close()
        logger.close()
        recorder.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
