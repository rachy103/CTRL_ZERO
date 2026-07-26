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
) -> int:
    import mujoco

    world_path = _prepare_assets(config)
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
    if mode == "auto":
        auto_control = AutoLaneControl(config, model, controller)

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
    parser.add_argument("--mode", choices=("manual", "auto"), default="manual")
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
        _apply_spawn_overrides(config, args)
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
        if args.auto_start and args.mode != "auto":
            raise ConfigurationError("--auto-start is only available in auto mode.")
        return _run_loop(
            config,
            mode=args.mode,
            view=args.view,
            headless=args.headless,
            steps=args.steps,
            explicit_steer=args.command_steer,
            explicit_pwm=args.command_pwm,
            auto_start=args.auto_start,
        )
    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
