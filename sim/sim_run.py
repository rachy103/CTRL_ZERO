from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Importing the real controller must not create ctrl_zero/__pycache__ outside sim.
sys.dont_write_bytecode = True

SIM_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SIM_DIR.parent
RUNTIME_CACHE_DIR = SIM_DIR / ".cache"
RUNTIME_TEMP_DIR = SIM_DIR / ".tmp"
for directory in (RUNTIME_CACHE_DIR, RUNTIME_TEMP_DIR):
    directory.mkdir(parents=True, exist_ok=True)
for environment_name, directory in {
    "PYTHONPYCACHEPREFIX": RUNTIME_CACHE_DIR / "pycache",
    "PIP_CACHE_DIR": RUNTIME_CACHE_DIR / "pip",
    "XDG_CACHE_HOME": RUNTIME_CACHE_DIR,
    "XDG_CONFIG_HOME": RUNTIME_CACHE_DIR / "config",
    "YOLO_CONFIG_DIR": RUNTIME_CACHE_DIR / "ultralytics",
    "TORCH_HOME": RUNTIME_CACHE_DIR / "torch",
    "MPLCONFIGDIR": RUNTIME_CACHE_DIR / "matplotlib",
    "TEMP": RUNTIME_TEMP_DIR,
    "TMP": RUNTIME_TEMP_DIR,
}.items():
    os.environ[environment_name] = str(directory)
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

try:
    from .build_map import build_map_assets
    from .config import (
        AUTO_CAMERA_REQUIRED,
        DEFAULT_CONFIG_PATH,
        KINEMATIC_HARDWARE_REQUIRED,
        ConfigurationError,
        MAP_REQUIRED,
        VEHICLE_GEOMETRY_REQUIRED,
        get_value,
        is_known,
        load_config,
        missing_parameters,
        readable_path,
        set_value,
        sim_path,
    )
    from .model_builder import build_models
    from .vehicle_kinematics import KinematicCalibration, KinematicVehicle
    from .lidar_sim import simulate_scan
    from . import parking_scenario
except ImportError:  # Direct execution: python -B sim/sim_run.py
    from build_map import build_map_assets
    from config import (
        AUTO_CAMERA_REQUIRED,
        DEFAULT_CONFIG_PATH,
        KINEMATIC_HARDWARE_REQUIRED,
        ConfigurationError,
        MAP_REQUIRED,
        VEHICLE_GEOMETRY_REQUIRED,
        get_value,
        is_known,
        load_config,
        missing_parameters,
        readable_path,
        set_value,
        sim_path,
    )
    from model_builder import build_models
    from vehicle_kinematics import KinematicCalibration, KinematicVehicle
    from lidar_sim import simulate_scan
    import parking_scenario


def build_real_drive_controller():
    """Construct the same DriveController and tuning values used by main.py."""

    import main as real_main
    from ctrl_zero.control import DriveConfig, DriveController

    return DriveController(
        DriveConfig(
            control_mode=real_main.CONTROL_MODE,
            min_speed=real_main.MIN_SPEED,
            max_speed=real_main.MAX_SPEED,
            contest_angle_weight=real_main.CONTEST_ANGLE_WEIGHT,
            contest_position_weight=real_main.CONTEST_POSITION_WEIGHT,
            contest_angle_norm_deg=real_main.CONTEST_STEERING_ANGLE_NORM_DEG,
            contest_steer_limit=real_main.CONTEST_STEER_LIMIT,
            min_confidence=real_main.MIN_LANE_CONFIDENCE_TO_DRIVE,
            max_steer=real_main.MAX_STEER,
            reverse_steer=real_main.REVERSE_STEER,
            max_hold_frames=real_main.MAX_HOLD_FRAMES,
            hold_decel_step=real_main.HOLD_DECEL_STEP,
        )
    )


def _finite_config_number(config: dict[str, Any], dotted_key: str) -> float:
    value = get_value(config, dotted_key)
    if not is_known(value):
        raise ConfigurationError(
            f"{dotted_key} requires a measured value; null is not allowed."
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(
            f"{dotted_key} must be a finite measured number."
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"{dotted_key} must be a finite measured number."
        ) from exc
    if not math.isfinite(number):
        raise ConfigurationError(
            f"{dotted_key} must be a finite measured number."
        )
    return number


def _validate_auto_camera_config(config: dict[str, Any]) -> None:
    width = _finite_config_number(config, "hardware.camera.width_px")
    height = _finite_config_number(config, "hardware.camera.height_px")
    fps = _finite_config_number(config, "hardware.camera.fps")
    horizontal_fov = _finite_config_number(
        config, "hardware.camera.horizontal_fov_deg"
    )
    _finite_config_number(config, "hardware.camera.pitch_deg")
    _finite_config_number(config, "hardware.camera.yaw_deg")

    if not width.is_integer() or not height.is_integer():
        raise ConfigurationError(
            "Measured camera width_px and height_px must be whole pixels."
        )
    if width <= 0 or height <= 0 or fps <= 0:
        raise ConfigurationError(
            "Measured camera width, height, and FPS must be positive."
        )
    if not 0 < horizontal_fov < 180:
        raise ConfigurationError(
            "Measured camera horizontal FOV must lie between 0 and 180 degrees."
        )

    position_key = "hardware.camera.position_from_rear_axle_m"
    position = get_value(config, position_key)
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise ConfigurationError(
            f"{position_key} must contain exactly three measured values [x, y, z]."
        )
    for index, value in enumerate(position):
        try:
            coordinate = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"{position_key}[{index}] must be a finite measured number."
            ) from exc
        if not math.isfinite(coordinate):
            raise ConfigurationError(
                f"{position_key}[{index}] must be a finite measured number."
            )


def _validate_spawn_config(config: dict[str, Any]) -> None:
    spawn_x = _finite_config_number(config, "map.spawn.rear_axle_x_m")
    spawn_y = _finite_config_number(config, "map.spawn.rear_axle_y_m")
    yaw = math.radians(_finite_config_number(config, "map.spawn.yaw_deg"))
    map_width = _finite_config_number(config, "map.physical_width_m")
    map_height = _finite_config_number(config, "map.physical_height_m")
    body_length = _finite_config_number(config, "vehicle.body_length_m")
    body_width = _finite_config_number(config, "vehicle.body_width_m")
    rear_axle_offset = _finite_config_number(
        config,
        "vehicle.rear_axle_from_rear_m",
    )
    if map_width <= 0 or map_height <= 0:
        raise ConfigurationError("Measured map dimensions must be positive.")
    if body_length <= 0 or body_width <= 0:
        raise ConfigurationError("Measured vehicle dimensions must be positive.")
    if not 0 < rear_axle_offset < body_length:
        raise ConfigurationError(
            "vehicle.rear_axle_from_rear_m must lie inside the measured body length."
        )

    local_x_values = (-rear_axle_offset, body_length - rear_axle_offset)
    local_y_values = (-body_width / 2.0, body_width / 2.0)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    half_map_width = map_width / 2.0
    half_map_height = map_height / 2.0
    outside: list[tuple[float, float]] = []
    for local_x in local_x_values:
        for local_y in local_y_values:
            world_x = spawn_x + cosine * local_x - sine * local_y
            world_y = spawn_y + sine * local_x + cosine * local_y
            if (
                world_x < -half_map_width
                or world_x > half_map_width
                or world_y < -half_map_height
                or world_y > half_map_height
            ):
                outside.append((world_x, world_y))
    if outside:
        rendered = ", ".join(
            f"({world_x:.3f}, {world_y:.3f})"
            for world_x, world_y in outside
        )
        raise ConfigurationError(
            "The measured vehicle footprint does not fit on the map at the "
            f"configured spawn pose. Outside corner(s): {rendered}"
        )


class ManualControl:
    """Map collision-free arrow keys onto main.py's speed and steering rules."""

    KEY_ESCAPE = 256
    KEY_ENTER = 257
    KEY_RIGHT = 262
    KEY_LEFT = 263
    KEY_DOWN = 264
    KEY_UP = 265

    def __init__(self, controller: Any):
        import main as real_main

        self.controller = controller
        self.speed_step = int(real_main.MANUAL_SPEED_STEP)
        self.steer_power = int(real_main.MANUAL_STEER_POWER)
        self.steer_hold_s = float(real_main.MANUAL_STEER_HOLD_MS) / 1000.0
        self.min_speed = int(real_main.MIN_SPEED)
        self.max_speed = int(real_main.MAX_SPEED)
        self.max_steer = int(real_main.MAX_STEER)
        self.manual_speed = 0
        self.manual_steer = 0
        self.manual_steer_until = 0.0
        self.drive_started = False
        self.manual_takeover = False
        self.quit_requested = False
        self._lock = threading.Lock()

    @staticmethod
    def _clamp_int(value: int, low: int, high: int) -> int:
        return min(max(value, low), high)

    def on_key(self, keycode: int) -> None:
        # GLFW reports letter keys as uppercase ASCII regardless of Shift.
        with self._lock:
            now = time.monotonic()
            if keycode in (ord("Q"), self.KEY_ESCAPE):
                self.quit_requested = True
                return
            if keycode == 32:  # Space: stop and keep manual mode armed.
                self.drive_started = True
                self.manual_takeover = True
                self.manual_speed = 0
                self.manual_steer = 0
                self.manual_steer_until = 0.0
                return
            if not self.drive_started:
                if keycode == self.KEY_ENTER:
                    self.drive_started = True
                    return
                if keycode in (
                    self.KEY_UP,
                    self.KEY_DOWN,
                    self.KEY_LEFT,
                    self.KEY_RIGHT,
                ):
                    self.drive_started = True
                else:
                    return
            if keycode in (ord("="), 334):  # Main keyboard +/= or keypad add.
                self.controller.config.max_speed = self._clamp_int(
                    int(self.controller.config.max_speed) + 5,
                    0,
                    self.max_speed,
                )
            elif keycode in (ord("-"), 333):  # Main keyboard - or keypad subtract.
                self.controller.config.max_speed = self._clamp_int(
                    int(self.controller.config.max_speed) - 5,
                    self.min_speed,
                    self.max_speed,
                )
            elif keycode == self.KEY_UP:
                self.manual_speed = self._clamp_int(
                    self.manual_speed + self.speed_step,
                    -self.max_speed,
                    self.max_speed,
                )
            elif keycode == self.KEY_DOWN:
                self.manual_speed = self._clamp_int(
                    self.manual_speed - self.speed_step,
                    -self.max_speed,
                    self.max_speed,
                )
            elif keycode == self.KEY_LEFT:
                self.manual_steer = -self._clamp_int(
                    self.steer_power, 0, self.max_steer
                )
                self.manual_steer_until = now + self.steer_hold_s
            elif keycode == self.KEY_RIGHT:
                self.manual_steer = self._clamp_int(
                    self.steer_power, 0, self.max_steer
                )
                self.manual_steer_until = now + self.steer_hold_s

    def arm_with_explicit_command(self, steer: int, pwm: int) -> None:
        with self._lock:
            self.drive_started = True
            self.manual_steer = steer
            self.manual_speed = pwm
            self.manual_steer_until = float("inf")

    def arm_auto(self) -> None:
        with self._lock:
            self.drive_started = True
            self.manual_takeover = False

    def command(self):
        from ctrl_zero.control import DriveCommand

        with self._lock:
            if (
                self.manual_steer_until
                and time.monotonic() > self.manual_steer_until
            ):
                self.manual_steer = 0
                self.manual_steer_until = 0.0
            if not self.drive_started:
                return DriveCommand(steer=0, speed=0, reason="waiting_for_start")
            return self.controller.compute(
                None,
                None,
                "manual",
                manual_steer=self.manual_steer,
                manual_speed=self.manual_speed,
            )


class AutoLaneControl:
    """Render the onboard camera and feed it through main.py's lane controller."""

    def __init__(self, config: dict[str, Any], model: Any, controller: Any):
        import cv2
        import main as real_main
        import mujoco

        from ctrl_zero.obstacles import (
            ContestObstacleMission,
            ContestObstacleMissionConfig,
        )
        from ctrl_zero.safety import build_safety_decision
        from ctrl_zero.traffic_light import traffic_light_object

        _validate_auto_camera_config(config)
        self.cv2 = cv2
        self.real_main = real_main
        self.build_safety_decision = build_safety_decision
        self.traffic_light_object = traffic_light_object
        self.controller = controller
        self.width = int(get_value(config, "hardware.camera.width_px"))
        self.height = int(get_value(config, "hardware.camera.height_px"))
        self.fps = float(get_value(config, "hardware.camera.fps"))
        self.obstacle_mission = ContestObstacleMission(
            ContestObstacleMissionConfig(
                enabled=real_main.OBSTACLE_MISSION_ENABLED,
                use_lidar=real_main.USE_LIDAR,
                camera_trigger_far_y_ratio=(
                    real_main.OBSTACLE_CAMERA_TRIGGER_FAR_Y_RATIO
                ),
                raw_angle_for_ros_zero_deg=(
                    real_main.LIDAR_RAW_ANGLE_FOR_ROS_ZERO_DEG
                ),
                lidar_obstacle_distance_mm=real_main.OBSTACLE_DISTANCE_MM,
                trigger_frames=real_main.OBSTACLE_TRIGGER_FRAMES,
                cruise_speed=real_main.MAX_SPEED,
                lane2_shift_steer=real_main.OBSTACLE_LANE2_SHIFT_STEER,
                lane1_shift_steer=real_main.OBSTACLE_LANE1_SHIFT_STEER,
                shift_out_duration_s=real_main.OBSTACLE_SHIFT_OUT_SECONDS,
                shift_out_steer_weight=(
                    real_main.OBSTACLE_SHIFT_OUT_STEER_WEIGHT
                ),
                retrigger_cooldown_s=(
                    real_main.OBSTACLE_RETRIGGER_COOLDOWN_SECONDS
                ),
                lidar_min_ros_deg=real_main.OBSTACLE_LIDAR_MIN_ROS_DEG,
                lidar_max_ros_deg=real_main.OBSTACLE_LIDAR_MAX_ROS_DEG,
            )
        )
        if (
            real_main.OBSTACLE_MISSION_ENABLED
            and self.obstacle_mission.config.use_lidar
        ):
            print(
                "Automatic obstacle mission is using main.py's LiDAR gate. "
                "The simulator has no LiDAR scan, so lidar_scan=None keeps the "
                "mission safely waiting instead of estimating a distance.",
                flush=True,
            )

        model_value = str(get_value(config, "auto.yolo_model"))
        model_path = readable_path(model_value)
        if not model_path.exists():
            raise ConfigurationError(
                f"Automatic mode requires the trained model, but it is absent: {model_path}"
            )
        real_args = real_main.build_parser().parse_args(
            [
                "--mode",
                "auto",
                "--no-arduino",
                "--no-display",
                "--yolo-model",
                str(model_path),
            ]
        )
        try:
            self.detector = real_main.build_lane_detector(real_args)
        except ModuleNotFoundError as exc:
            raise ConfigurationError(
                f"Automatic mode dependency is missing: {exc.name}"
            ) from exc
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        self.last_update_time = -float("inf")
        self.last_command = None
        self.last_base_command = None
        self.last_mission_command = None
        self.last_obstacle_status = None
        self.last_lane = None

    def close(self) -> None:
        self.renderer.close()

    def command(self, data: Any):
        now = float(data.time)
        if self.last_command is not None and now - self.last_update_time < 1.0 / self.fps:
            return self.last_command
        self.renderer.update_scene(data, camera="onboard")
        rgb_frame = self.renderer.render()
        frame = self.cv2.cvtColor(rgb_frame, self.cv2.COLOR_RGB2BGR)
        lane = self.detector.detect(frame)
        traffic_object = self.traffic_light_object(lane.objects)
        frame_area = float(max(frame.shape[0] * frame.shape[1], 1))
        safety = self.build_safety_decision(
            lidar=None,
            traffic_light_state=lane.traffic_light_state,
            traffic_light_object=traffic_object,
            traffic_light_frame_area=frame_area,
            traffic_light_min_stop_area_ratio=self.real_main.TRAFFIC_LIGHT_STOP_AREA_RATIO,
        )
        base_command = self.controller.compute(lane, safety, "auto")
        mission_command = self.obstacle_mission.step(
            objects=lane.objects,
            current_lane=lane.lane_label or lane.lane_pair_label,
            frame_width=lane.annotated.shape[1],
            frame_height=lane.annotated.shape[0],
            lidar_scan=None,
            heading_deg=lane.heading_deg,
            offset_norm=lane.offset_norm,
            lane_follow_steer=base_command.steer,
            cruise_speed=self.controller.config.max_speed,
        )
        self.last_command = self.real_main.apply_obstacle_mission_override(
            base_command,
            mission_command,
            safety,
        )
        self.last_base_command = base_command
        self.last_mission_command = mission_command
        self.last_obstacle_status = self.obstacle_mission.debug_status()
        self.last_lane = lane
        self.last_update_time = now
        return self.last_command


PARKING_LIDAR_REQUIRED = (
    "hardware.lidar.position_from_rear_axle_m",
    "hardware.lidar.raw_angle_for_ros_zero_deg",
    "hardware.lidar.range_max_mm",
    "hardware.lidar.angular_step_deg",
)


class ParkingControl:
    """Drive main.py's ParkingHardMission from a simulated LiDAR scan."""

    def __init__(
        self,
        config: dict[str, Any],
        case: int,
        *,
        reverse_steer_override: int | None = None,
        search_overrun_s: float = 0.0,
        side_detect_mm_override: float | None = None,
    ):
        import main as real_main
        from ctrl_zero.parking_hard import (
            ParkingHardConfig,
            ParkingHardMission,
        )

        # Tuning experiments only touch the simulation; main.py stays intact.
        self.search_overrun_s = float(search_overrun_s)
        self.search_cleared_time: float | None = None
        self.reverse_straight_max_s: float | None = None
        self.last_command = None
        self.last_scan_time = -float("inf")

        for dotted_key in PARKING_LIDAR_REQUIRED:
            if not is_known(get_value(config, dotted_key)):
                raise ConfigurationError(
                    f"{dotted_key} requires a measured value; null is not allowed."
                )
        position = get_value(config, "hardware.lidar.position_from_rear_axle_m")
        if not isinstance(position, (list, tuple)) or len(position) != 2:
            raise ConfigurationError(
                "hardware.lidar.position_from_rear_axle_m must be "
                "[forward_m, left_m] from the rear axle center."
            )
        self.lidar_forward_m = float(position[0])
        self.lidar_left_m = float(position[1])
        self.raw_zero_deg = float(
            get_value(config, "hardware.lidar.raw_angle_for_ros_zero_deg")
        )
        self.range_max_mm = float(get_value(config, "hardware.lidar.range_max_mm"))
        self.step_deg = float(get_value(config, "hardware.lidar.angular_step_deg"))
        self.scan_rate_hz = float(get_value(config, "hardware.lidar.scan_rate_hz"))
        self.obstacles = parking_scenario.obstacle_rects(case)
        self.mission = ParkingHardMission(
            ParkingHardConfig(
                enabled=True,
                raw_angle_for_ros_zero_deg=(
                    real_main.LIDAR_RAW_ANGLE_FOR_ROS_ZERO_DEG
                ),
                search_sector_ros_deg=real_main.PARKING_SEARCH_SECTOR_ROS_DEG,
                left_sector_ros_deg=real_main.PARKING_LEFT_SECTOR_ROS_DEG,
                right_sector_ros_deg=real_main.PARKING_RIGHT_SECTOR_ROS_DEG,
                search_detect_mm=real_main.PARKING_SEARCH_DETECT_MM,
                side_detect_mm=(
                    real_main.PARKING_SIDE_DETECT_MM
                    if side_detect_mm_override is None
                    else float(side_detect_mm_override)
                ),
                go_out_detect_mm=real_main.PARKING_GO_OUT_DETECT_MM,
                detect_frames=real_main.PARKING_DETECT_FRAMES,
                clear_frames=real_main.PARKING_CLEAR_FRAMES,
                init_steer=real_main.PARKING_INIT_STEER,
                straight_steer=real_main.PARKING_STRAIGHT_STEER,
                first_stop_steer=real_main.PARKING_FIRST_STOP_STEER,
                reverse_right_steer=(
                    real_main.PARKING_INIT_STEER
                    if reverse_steer_override is None
                    else int(reverse_steer_override)
                ),
                go_out_turn_steer=real_main.PARKING_GO_OUT_TURN_STEER,
                go_out_straight_steer=real_main.PARKING_STRAIGHT_STEER,
                centering_deadband_mm=real_main.PARKING_CENTERING_DEADBAND_MM,
                centering_steer=real_main.PARKING_CENTERING_STEER,
                search_speed=real_main.PARKING_SEARCH_SPEED,
                reverse_right_speed=real_main.PARKING_REVERSE_RIGHT_SPEED,
                reverse_straight_speed=real_main.PARKING_REVERSE_STRAIGHT_SPEED,
                adjust_forward_speed=real_main.PARKING_ADJUST_FORWARD_SPEED,
                go_out_turn_speed=real_main.PARKING_GO_OUT_TURN_SPEED,
                go_out_straight_speed=real_main.PARKING_GO_OUT_STRAIGHT_SPEED,
                first_stop_seconds=real_main.PARKING_FIRST_STOP_SECONDS,
                reverse_pause_seconds=real_main.PARKING_REVERSE_PAUSE_SECONDS,
                adjust_forward_min_seconds=(
                    real_main.PARKING_ADJUST_FORWARD_MIN_SECONDS
                ),
                go_out_turn_min_seconds=(
                    real_main.PARKING_GO_OUT_TURN_MIN_SECONDS
                ),
            )
        )
        self.last_state_name = self.mission.state.name

    def command(self, data: Any, vehicle_state: Any):
        now = float(data.time)
        if (
            self.last_command is not None
            and now - self.last_scan_time < 1.0 / self.scan_rate_hz
        ):
            return self.last_command
        self.last_scan_time = now
        yaw = float(vehicle_state.yaw_rad)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        lidar_x = (
            float(vehicle_state.rear_axle_x_m)
            + cos_yaw * self.lidar_forward_m
            - sin_yaw * self.lidar_left_m
        )
        lidar_y = (
            float(vehicle_state.rear_axle_y_m)
            + sin_yaw * self.lidar_forward_m
            + cos_yaw * self.lidar_left_m
        )
        scan = simulate_scan(
            lidar_x,
            lidar_y,
            yaw,
            self.obstacles,
            raw_angle_for_ros_zero_deg=self.raw_zero_deg,
            angular_step_deg=self.step_deg,
            range_max_mm=self.range_max_mm,
        )
        command = self.mission.step(scan, now_s=now)
        command = self._apply_search_overrun(command, now)
        command = self._apply_reverse_straight_cap(command, now)
        self.last_command = command
        state_name = self.mission.state.name
        if state_name != self.last_state_name:
            print(
                f"PARKING_STATE t={float(data.time):7.2f}s "
                f"{self.last_state_name} -> {state_name} "
                f"x={vehicle_state.rear_axle_x_m:+7.3f} "
                f"y={vehicle_state.rear_axle_y_m:+7.3f} "
                f"yaw={math.degrees(yaw):+7.2f}deg",
                flush=True,
            )
            self.last_state_name = state_name
        return command

    def _apply_search_overrun(self, command, now_s: float):
        """Prototype of a proposed main.py change: keep driving for a fixed
        time after the second parked car clears, so the reverse arc starts
        far enough past the target bay. Holds the mission in FIRST_STOP by
        freezing its timer while substituting the search drive command."""
        from ctrl_zero.control import DriveCommand
        from ctrl_zero.parking_hard import ParkingHardState

        if self.search_overrun_s <= 0.0:
            return command
        if self.mission.state != ParkingHardState.FIRST_STOP:
            return command
        if self.search_cleared_time is None:
            self.search_cleared_time = now_s
        if now_s - self.search_cleared_time < self.search_overrun_s:
            self.mission.state_start_s = now_s
            return DriveCommand(
                steer=self.mission.config.straight_steer,
                speed=self.mission.config.search_speed,
                reason="parking_search_overrun",
            )
        return command

    def _apply_reverse_straight_cap(self, command, now_s: float):
        """Prototype of a proposed main.py change: the reverse arc already
        ends at the correct bay depth, so bound REVERSE_STRAIGHT by time
        instead of waiting for the front-mounted LiDAR to clear both
        neighbors (which reverses the whole car past the bay)."""
        from ctrl_zero.parking_hard import ParkingHardState

        if self.reverse_straight_max_s is None:
            return command
        if self.mission.state != ParkingHardState.REVERSE_STRAIGHT:
            return command
        if now_s - self.mission.state_start_s >= self.reverse_straight_max_s:
            self.mission._enter(ParkingHardState.REVERSE_PAUSE, now_s)
        return command


class TimedParkingControl:
    """Timed two-arc parking driven by main.py's ``TimedParkingMission`` —
    the exact object the real car runs in ``--mode parking_sim`` — fed with
    simulated LiDAR scans at the sensor rate.

    ``parking_scenario.timed_plan`` re-derives the closed-form segment times
    from the calibrated geometry and is printed next to main.py's configured
    constants so any drift between the two is visible.
    """

    def __init__(self, config: dict[str, Any], case: int, marker: int):
        import main as real_main

        for dotted_key in PARKING_LIDAR_REQUIRED:
            if not is_known(get_value(config, dotted_key)):
                raise ConfigurationError(
                    f"{dotted_key} requires a measured value; null is not allowed."
                )
        position = get_value(config, "hardware.lidar.position_from_rear_axle_m")
        self.lidar_forward_m = float(position[0])
        self.lidar_left_m = float(position[1])
        self.raw_zero_deg = float(
            get_value(config, "hardware.lidar.raw_angle_for_ros_zero_deg")
        )
        self.range_max_mm = float(get_value(config, "hardware.lidar.range_max_mm"))
        self.step_deg = float(get_value(config, "hardware.lidar.angular_step_deg"))
        self.scan_rate_hz = float(get_value(config, "hardware.lidar.scan_rate_hz"))
        self.obstacles = parking_scenario.obstacle_rects(case)

        pwm_table = get_value(config, "hardware.drive.pwm_speed_table")
        top_speed = float(pwm_table[-1][1])
        top_pwm = float(pwm_table[-1][0])
        deadband = float(
            get_value(config, "hardware.command.drive_pwm_deadband")
        )

        def pwm_speed(pwm: float) -> float:
            return top_speed * (abs(pwm) - deadband) / (top_pwm - deadband)

        self.mission = real_main.build_timed_parking_mission(marker)
        mission_config = self.mission.config
        plan = parking_scenario.timed_plan(
            marker,
            wheelbase_m=float(get_value(config, "vehicle.wheelbase_m")),
            front_track_m=float(get_value(config, "vehicle.front_track_m")),
            max_steer_deg=float(
                get_value(config, "hardware.steering.right_angle_deg_at_steer_max")
            ),
            v_forward_mps=pwm_speed(mission_config.forward_speed),
            v_reverse_mps=pwm_speed(abs(mission_config.reverse_speed)),
            lidar_forward_m=self.lidar_forward_m,
            trigger_latency_s=(
                mission_config.detect_hold_s + 0.5 / self.scan_rate_hz
            ),
            # Simulation-calibrated centering trim (cone-entry offset and
            # steering slew transients), measured per start column.
            approach_trim_s=0.16 if marker in (1, 3) else 0.26,
        )
        print(
            "TIMED_PLAN(derived) "
            + " ".join(f"{key}={value:.2f}" for key, value in plan.items()),
            flush=True,
        )
        print(
            "TIMED_MAIN(config)  "
            f"approach_s={mission_config.approach_s:.2f} "
            f"swing_s={mission_config.swing_s:.2f} "
            f"arc_s={mission_config.arc_s:.2f}",
            flush=True,
        )
        self.last_state_name = self.mission.state.name
        self.last_scan = None
        self.last_scan_time = -float("inf")
        self.parked_pose: tuple[float, float, float] | None = None

    def command(self, data: Any, vehicle_state: Any):
        now = float(data.time)
        # Regenerate the scan at the sensor rate; between scans the mission
        # sees the latest scan, exactly like the real main.py loop.
        if now - self.last_scan_time >= 1.0 / self.scan_rate_hz:
            self.last_scan_time = now
            yaw = float(vehicle_state.yaw_rad)
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            lidar_x = (
                float(vehicle_state.rear_axle_x_m)
                + cos_yaw * self.lidar_forward_m
                - sin_yaw * self.lidar_left_m
            )
            lidar_y = (
                float(vehicle_state.rear_axle_y_m)
                + sin_yaw * self.lidar_forward_m
                + cos_yaw * self.lidar_left_m
            )
            self.last_scan = simulate_scan(
                lidar_x,
                lidar_y,
                yaw,
                self.obstacles,
                raw_angle_for_ros_zero_deg=self.raw_zero_deg,
                angular_step_deg=self.step_deg,
                range_max_mm=self.range_max_mm,
            )

        command = self.mission.step(self.last_scan, now_s=now)
        state_name = self.mission.state.name
        if state_name != self.last_state_name:
            print(
                f"PARKING_PHASE t={now:7.2f}s "
                f"{self.last_state_name} -> {state_name} "
                f"x={vehicle_state.rear_axle_x_m:+7.3f} "
                f"y={vehicle_state.rear_axle_y_m:+7.3f} "
                f"yaw={math.degrees(vehicle_state.yaw_rad):+7.2f}deg",
                flush=True,
            )
            if state_name == "DONE":
                self.parked_pose = (
                    float(vehicle_state.rear_axle_x_m),
                    float(vehicle_state.rear_axle_y_m),
                    math.degrees(float(vehicle_state.yaw_rad)),
                )
                print(
                    f"PARKED t={now:7.2f}s x={self.parked_pose[0]:+7.3f} "
                    f"y={self.parked_pose[1]:+7.3f} "
                    f"yaw={self.parked_pose[2]:+7.2f}deg",
                    flush=True,
                )
            self.last_state_name = state_name
        return command


def _write_parking_world(world_path: Path, case: int) -> Path:
    """Copy the generated world and add the obstacle-car boxes for one case."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(world_path)
    worldbody = tree.getroot().find("worldbody")
    if worldbody is None:
        raise ConfigurationError(f"Generated world has no worldbody: {world_path}")
    for index, rect in enumerate(parking_scenario.obstacle_rects(case), start=1):
        # Obstacles are parked at exactly yaw 90 deg, so the oriented
        # footprint is expressed by swapping the box half-sizes.
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": f"parking_obstacle_{index}",
                "type": "box",
                "pos": f"{rect.center_x} {rect.center_y} 0.205",
                "size": f"{rect.width_m / 2.0} {rect.length_m / 2.0} 0.205",
                "rgba": "0.93 0.93 0.96 1",
                "contype": "0",
                "conaffinity": "0",
            },
        )
    destination = world_path.with_name(f"world_parking_case{case}.xml")
    tree.write(destination)
    return destination


def _apply_parking_spawn(config: dict[str, Any], marker: int) -> None:
    pose = parking_scenario.start_pose(
        marker,
        float(get_value(config, "vehicle.wheelbase_m")),
        float(get_value(config, "vehicle.front_track_m")),
    )
    set_value(config, "map.spawn.rear_axle_x_m", pose.rear_axle_x_m)
    set_value(config, "map.spawn.rear_axle_y_m", pose.rear_axle_y_m)
    set_value(config, "map.spawn.yaw_deg", math.degrees(pose.yaw_rad))


def _apply_spawn_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    provided = (args.spawn_x, args.spawn_y, args.spawn_yaw_deg)
    if any(value is not None for value in provided) and not all(
        value is not None for value in provided
    ):
        raise ConfigurationError(
            "--spawn-x, --spawn-y, and --spawn-yaw-deg must be supplied together."
        )
    if all(value is not None for value in provided):
        set_value(config, "map.spawn.rear_axle_x_m", args.spawn_x)
        set_value(config, "map.spawn.rear_axle_y_m", args.spawn_y)
        set_value(config, "map.spawn.yaw_deg", args.spawn_yaw_deg)


def _validate_config(config: dict[str, Any], *, auto: bool) -> None:
    missing = missing_parameters(config, include_spawn=True, auto=auto)
    if missing:
        formatted = "\n".join(f"  - {key}" for key in missing)
        raise ConfigurationError(
            "Simulation is blocked until these measured values are provided:\n"
            f"{formatted}\n"
            "No fallback/default/estimated hardware value was used."
        )
    for dotted_key in KINEMATIC_HARDWARE_REQUIRED:
        if not is_known(get_value(config, dotted_key)):
            raise ConfigurationError(
                f"{dotted_key} requires a measured value; null is not allowed."
            )
    driven_axle = str(
        get_value(config, "hardware.drive.driven_axle")
    ).strip().lower()
    if driven_axle not in {"front", "rear", "all"}:
        raise ConfigurationError(
            "hardware.drive.driven_axle must be the measured value: front, rear, or all."
        )
    zero_behavior = str(
        get_value(config, "hardware.drive.zero_pwm_behavior")
    ).strip().lower()
    if zero_behavior not in {"coast", "brake"}:
        raise ConfigurationError(
            "hardware.drive.zero_pwm_behavior must be the observed value: coast or brake."
        )
    if zero_behavior == "coast" and not is_known(
        get_value(config, "hardware.drive.coast_deceleration_mps2")
    ):
        raise ConfigurationError(
            "hardware.drive.coast_deceleration_mps2 requires a measured value "
            "when zero_pwm_behavior is coast; null is not allowed."
        )
    KinematicCalibration.from_config(config)
    _validate_spawn_config(config)
    if auto:
        model_path = readable_path(str(get_value(config, "auto.yolo_model")))
        if not model_path.exists():
            raise ConfigurationError(
                f"Automatic mode requires the trained model, but it is absent: {model_path}"
            )
        _validate_auto_camera_config(config)


def _validate_build_config(config: dict[str, Any]) -> None:
    from_config_keys = (
        *MAP_REQUIRED,
        *VEHICLE_GEOMETRY_REQUIRED,
    )
    missing = [key for key in from_config_keys if not is_known(get_value(config, key))]
    if missing:
        rendered = "\n".join(f"  - {key}" for key in missing)
        raise ConfigurationError(
            "Asset generation is blocked until these measured values are provided:\n"
            f"{rendered}\n"
            "No fallback/default/estimated hardware value was used."
        )


def _print_validation(config: dict[str, Any], *, auto: bool) -> int:
    try:
        _validate_config(config, auto=auto)
    except ConfigurationError as exc:
        print(f"BLOCKED: {exc}")
        return 2
    print("READY: every parameter required by the selected mode is present.")
    return 0


def _prepare_assets(config: dict[str, Any]) -> Path:
    build_map_assets(config, preview_only=False)
    outputs = build_models(config)
    return outputs["mjcf"]


def _run_loop(
    config: dict[str, Any],
    *,
    mode: str,
    view: str,
    headless: bool,
    steps: int | None,
    explicit_steer: int | None,
    explicit_pwm: int | None,
    auto_start: bool,
    parking_case: int = 1,
    parking_reverse_steer: int | None = None,
    parking_search_overrun_s: float = 0.0,
    parking_reverse_straight_max_s: float | None = None,
    parking_side_detect_mm: float | None = None,
    parking_strategy: str = "mission",
    parking_start: int = 1,
) -> int:
    import mujoco

    world_path = _prepare_assets(config)
    if mode == "parking":
        world_path = _write_parking_world(world_path, parking_case)
    model = mujoco.MjModel.from_xml_path(str(world_path))
    data = mujoco.MjData(model)
    calibration = KinematicCalibration.from_config(config)
    vehicle = KinematicVehicle(
        calibration,
        rear_axle_x_m=float(get_value(config, "map.spawn.rear_axle_x_m")),
        rear_axle_y_m=float(get_value(config, "map.spawn.rear_axle_y_m")),
        yaw_rad=math.radians(float(get_value(config, "map.spawn.yaw_deg"))),
    )
    vehicle.sync_mocap(model, data)
    mujoco.mj_forward(model, data)

    controller = build_real_drive_controller()
    manual = ManualControl(controller)
    auto_control = None
    parking_control = None
    if mode == "auto":
        auto_control = AutoLaneControl(config, model, controller)
    elif mode == "parking" and parking_strategy == "timed":
        parking_control = TimedParkingControl(config, parking_case, parking_start)
    elif mode == "parking":
        parking_control = ParkingControl(
            config,
            parking_case,
            reverse_steer_override=parking_reverse_steer,
            search_overrun_s=parking_search_overrun_s,
            side_detect_mm_override=parking_side_detect_mm,
        )
        parking_control.reverse_straight_max_s = parking_reverse_straight_max_s

    if (explicit_steer is None) != (explicit_pwm is None):
        raise ConfigurationError(
            "--command-steer and --command-pwm must be supplied together."
        )
    if explicit_steer is not None:
        manual.arm_with_explicit_command(explicit_steer, explicit_pwm)
    if auto_start:
        manual.arm_auto()

    dt = float(model.opt.timestep)
    completed_steps = 0
    next_status_time = 0.0

    def one_step() -> None:
        nonlocal completed_steps, next_status_time
        if auto_control is not None and manual.drive_started and not manual.manual_takeover:
            command = auto_control.command(data)
        elif (
            parking_control is not None
            and manual.drive_started
            and not manual.manual_takeover
        ):
            command = parking_control.command(data, vehicle.state)
        else:
            command = manual.command()
        vehicle.set_command(command.steer, command.speed)
        state = vehicle.step(dt)
        vehicle.sync_mocap(model, data)
        mujoco.mj_step(model, data)
        completed_steps += 1
        if data.time >= next_status_time:
            print(
                f"t={data.time:7.2f}s x={state.rear_axle_x_m:+7.3f} "
                f"y={state.rear_axle_y_m:+7.3f} yaw={math.degrees(state.yaw_rad):+7.2f}deg "
                f"steer={command.steer:+4d} pwm={command.speed:+4d} "
                f"v={state.speed_mps:+.3f}m/s reason={command.reason}",
                flush=True,
            )
            next_status_time = data.time + 1.0

    try:
        if headless:
            if steps is None or steps <= 0:
                raise ConfigurationError("--headless requires a positive --steps value.")
            while completed_steps < steps:
                one_step()
            return 0

        import mujoco.viewer

        print(
            "Keys: arrows drive/steer | Enter arm | +/- auto max PWM "
            "| Space stop | Q/Escape quit (W is MuJoCo wireframe; not a drive key)"
        )
        with mujoco.viewer.launch_passive(
            model,
            data,
            key_callback=manual.on_key,
            show_left_ui=True,
            show_right_ui=True,
        ) as viewer:
            if view == "onboard":
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = model.camera("onboard").id
            else:
                viewer.cam.lookat[:] = (0.0, 0.0, 0.0)
                viewer.cam.distance = max(
                    float(get_value(config, "map.physical_width_m")),
                    float(get_value(config, "map.physical_height_m")),
                ) * 0.92
                viewer.cam.azimuth = 90.0
                viewer.cam.elevation = -70.0
            while viewer.is_running() and not manual.quit_requested:
                started = time.perf_counter()
                one_step()
                viewer.sync()
                remaining = dt - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)
                if steps is not None and completed_steps >= steps:
                    break
        return 0
    finally:
        vehicle.set_command(0, 0)
        if auto_control is not None:
            auto_control.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real main.py DriveController against a calibrated MuJoCo map "
            "without touching the real vehicle runtime."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--mode", choices=("manual", "auto", "parking"), default="manual"
    )
    parser.add_argument(
        "--parking-case",
        type=int,
        choices=(1, 2),
        default=1,
        help="Obstacle-car layout: 1 = cars in bays 2 and 4, 2 = bays 1 and 3.",
    )
    parser.add_argument(
        "--parking-start",
        type=int,
        choices=(1, 2, 3, 4),
        default=1,
        help="Start marker 1-4 (contest 30 cm grid near the entry line).",
    )
    parser.add_argument(
        "--parking-reverse-steer",
        type=int,
        help=(
            "Simulation-only override of the REVERSE_RIGHT steer command "
            "(main.py uses PARKING_INIT_STEER)."
        ),
    )
    parser.add_argument(
        "--parking-search-overrun-s",
        type=float,
        default=0.0,
        help=(
            "Simulation-only prototype: keep driving this many seconds after "
            "the second parked car clears before the first stop."
        ),
    )
    parser.add_argument(
        "--parking-reverse-straight-max-s",
        type=float,
        help=(
            "Simulation-only prototype: bound REVERSE_STRAIGHT by time instead "
            "of waiting for both side sectors to clear."
        ),
    )
    parser.add_argument(
        "--parking-side-detect-mm",
        type=float,
        help=(
            "Simulation-only override of PARKING_SIDE_DETECT_MM (side-sector "
            "detection threshold during the reverse arc)."
        ),
    )
    parser.add_argument(
        "--parking-strategy",
        choices=("mission", "timed"),
        default="mission",
        help=(
            "mission = main.py's LiDAR-reactive ParkingHardMission; timed = "
            "closed-form two-arc plan (swing left, reverse right, straight "
            "back until both side sectors detect)."
        ),
    )
    parser.add_argument(
        "--view",
        choices=("overhead", "onboard"),
        default="overhead",
    )
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--spawn-x", type=float)
    parser.add_argument("--spawn-y", type=float)
    parser.add_argument("--spawn-yaw-deg", type=float)
    parser.add_argument("--command-steer", type=int)
    parser.add_argument("--command-pwm", type=int)
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Arm automatic control without a keyboard start event (for headless runs).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_config(args.config)
        if args.mode == "parking" and args.spawn_x is None:
            _apply_parking_spawn(config, args.parking_start)
        _apply_spawn_overrides(config, args)
        if args.mode == "parking":
            for dotted_key in PARKING_LIDAR_REQUIRED:
                if not is_known(get_value(config, dotted_key)):
                    raise ConfigurationError(
                        f"{dotted_key} requires a measured value; null is not allowed."
                    )
        if args.validate:
            return _print_validation(config, auto=args.mode == "auto")
        if args.build_only:
            _validate_build_config(config)
            world_path = _prepare_assets(config)
            print(f"Built: {world_path}")
            return 0
        _validate_config(config, auto=args.mode == "auto")
        if (args.command_steer is None) != (args.command_pwm is None):
            raise ConfigurationError(
                "--command-steer and --command-pwm must be supplied together."
            )
        if args.mode == "auto" and args.command_steer is not None:
            raise ConfigurationError(
                "Explicit command injection is only available in manual mode."
            )
        if args.auto_start and args.mode not in ("auto", "parking"):
            raise ConfigurationError(
                "--auto-start is only available in auto or parking mode."
            )
        return _run_loop(
            config,
            mode=args.mode,
            view=args.view,
            headless=args.headless,
            steps=args.steps,
            explicit_steer=args.command_steer,
            explicit_pwm=args.command_pwm,
            auto_start=args.auto_start,
            parking_case=args.parking_case,
            parking_reverse_steer=args.parking_reverse_steer,
            parking_search_overrun_s=args.parking_search_overrun_s,
            parking_reverse_straight_max_s=args.parking_reverse_straight_max_s,
            parking_side_detect_mm=args.parking_side_detect_mm,
            parking_strategy=args.parking_strategy,
            parking_start=args.parking_start,
        )
    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
