from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class LidarConfig:
    port: str | None = None
    scan_type: str = "normal"
    max_buffer_size: int = 3000
    sample_rate: int = 10
    min_distance_mm: float = 0.0
    front_min_angle_deg: float = 87.5
    front_max_angle_deg: float = 92.5
    # Unused: front-LiDAR stop and slowdown were removed, so these no longer
    # affect speed.  Kept only so existing configs/tests keep constructing.
    stop_distance_mm: float = 450.0
    slow_distance_mm: float = 900.0
    min_speed_scale: float = 1.0
    rpm: int | None = None
    reconnect_delay_s: float = 1.0
    stale_scan_seconds: float = 1.0
    motor_spin_up_s: float = 2.0
    serial_timeout_s: float = 3.0


@dataclass
class ObstacleDecision:
    nearest_front_mm: float | None
    speed_scale: float
    should_stop: bool
    front_points: int

    @classmethod
    def clear(cls) -> "ObstacleDecision":
        return cls(nearest_front_mm=None, speed_scale=1.0, should_stop=False, front_points=0)


class LidarReader:
    """Continuously drains the RPLidar on a background thread.

    RPLidar streams measurements without pause, so pulling a scan only once
    every few frames lets the serial buffer accumulate until the byte stream
    desyncs and the driver dies (surfacing as a bare ``StopIteration``).  Here a
    daemon thread consumes ``iter_measures`` as fast as the device produces it,
    keeping the buffer drained, and publishes only the most recent completed
    scan.  ``read_scan`` returns that latest scan without blocking, so it always
    reflects the current frame instead of a backlog.  On any read/connection
    failure the thread tears the device down and reconnects on its own, so a
    transient glitch no longer disables LiDAR for the rest of the run.
    """

    def __init__(self, config: LidarConfig):
        self.config = config
        self.lidar = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_scan: np.ndarray | None = None
        self._latest_scan_time: float = 0.0
        # Per-connection diagnostics: how many raw measurements and completed
        # scans arrived.  measures==0 on failure means no data ever flowed
        # (motor/wiring), measures>0 with scans==0 means data flowed but never
        # formed a rotation (desync/flag parsing).
        self.measures_this_connection = 0
        self.scans_this_connection = 0

    def open(self) -> None:
        if not self.config.port:
            raise RuntimeError("LIDAR_PORT is empty. Set it in main.py before USE_LIDAR=True.")
        try:
            from rplidar import RPLidar  # noqa: F401  (fail fast if the driver is missing)
        except ModuleNotFoundError as exc:
            raise RuntimeError("rplidar-roboticia is required for LiDAR. Install requirements.txt first.") from exc

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="lidar-reader", daemon=True)
        self._thread.start()

    def read_scan(self) -> np.ndarray | None:
        with self._lock:
            scan = self._latest_scan
            stamp = self._latest_scan_time
        if scan is None:
            return None
        stale_after = self.config.stale_scan_seconds
        if stale_after > 0.0 and (time.monotonic() - stamp) > stale_after:
            return None
        return scan

    def close(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            self._thread = None
        self._teardown_device()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.measures_this_connection = 0
            self.scans_this_connection = 0
            try:
                self._connect()
                self._stream()
            except Exception as exc:  # pragma: no cover - hardware dependent
                print(
                    f"LiDAR reader error, reconnecting: {type(exc).__name__}: {exc} "
                    f"(this connection: measures={self.measures_this_connection}, "
                    f"scans={self.scans_this_connection})"
                )
            finally:
                self._teardown_device()
            if self._stop_event.is_set():
                break
            self._stop_event.wait(max(self.config.reconnect_delay_s, 0.0))

    def _connect(self) -> None:
        from rplidar import RPLidar

        # Assign before configuring so a failure mid-handshake is still cleaned up.
        self.lidar = RPLidar(self.config.port)
        self._widen_serial_timeout(self.lidar)
        self._flush_stale_stream(self.lidar)
        if self.config.rpm is not None:
            self.lidar.motor_speed = self.config.rpm
        # Start the motor explicitly and give it time to reach speed: a scan
        # request against a still/slow rotor produces no measurement bytes and
        # the first read times out as "Wrong body size".
        try:
            self.lidar.start_motor()
        except Exception:  # pragma: no cover - hardware dependent
            pass
        if self.config.motor_spin_up_s > 0.0:
            self._stop_event.wait(self.config.motor_spin_up_s)
        print(f"LiDAR info: {self.lidar.get_info()}")
        print(f"LiDAR health: {self.lidar.get_health()}")

    def _widen_serial_timeout(self, lidar) -> None:
        """Raise the driver's 1s serial read timeout to tolerate spin-up gaps."""
        if self.config.serial_timeout_s <= 0.0:
            return
        for attr in ("_serial_port", "_serial"):  # Skoltech / roboticia naming
            serial_port = getattr(lidar, attr, None)
            if serial_port is not None:
                try:
                    serial_port.timeout = self.config.serial_timeout_s
                except Exception:  # pragma: no cover - hardware dependent
                    pass
                return

    @staticmethod
    def _flush_stale_stream(lidar) -> None:
        """Stop any scan left over from a previous session and drop its bytes.

        Sending get_info()/get_health() while stale measurement bytes are still
        sitting in the serial buffer desyncs the response parser and raises
        ``RPLidarException: Wrong body size``, so stop the stream and drain the
        buffer before talking to the device.
        """
        try:
            lidar.stop()
        except Exception:  # pragma: no cover - hardware dependent
            pass
        time.sleep(0.1)  # let in-flight measurement bytes land before draining
        try:
            lidar.clean_input()
        except Exception:  # pragma: no cover - hardware dependent
            pass

    def _stream(self) -> None:
        lidar = self.lidar
        if lidar is None:
            return
        # Streaming method names differ across rplidar forks.  Prefer the
        # per-measure stream; fall back to whole-scan streaming, which every
        # variant provides.
        if hasattr(lidar, "iter_measures"):
            # rplidar-roboticia: iter_measures(scan_type, max_buf_meas)
            self._stream_measures(lidar.iter_measures(self.config.scan_type, self.config.max_buffer_size))
        elif hasattr(lidar, "iter_measurments"):
            # SkoltechRobotics rplidar: iter_measurments(max_buf_meas)  [sic]
            self._stream_measures(lidar.iter_measurments(self.config.max_buffer_size))
        elif hasattr(lidar, "iter_scans"):
            self._stream_scans(lidar)
        else:
            raise RuntimeError(
                "Installed rplidar library exposes none of "
                "iter_measures / iter_measurments / iter_scans."
            )

    def _stream_measures(self, iterator) -> None:
        scan_list: list[tuple[float, float, float]] = []
        for new_scan, quality, angle, distance in iterator:
            if self._stop_event.is_set():
                break
            self.measures_this_connection += 1
            if new_scan:
                if len(scan_list) > self.config.sample_rate:
                    data = np.array(scan_list, dtype=np.float32)
                    self._publish(data[:, 1:])
                scan_list = []
            if distance > self.config.min_distance_mm:
                scan_list.append((quality, angle, distance))

    def _stream_scans(self, lidar) -> None:
        # Call with no positional args: forks disagree on the first parameter
        # (scan_type vs max_buf_meas) but all default it sensibly.  iter_scans
        # yields a full rotation as a list of (quality, angle, distance).
        for scan in lidar.iter_scans():
            if self._stop_event.is_set():
                break
            self.measures_this_connection += len(scan)
            rows = [
                (angle, distance)
                for _quality, angle, distance in scan
                if distance > self.config.min_distance_mm
            ]
            if len(rows) > self.config.sample_rate:
                self._publish(np.array(rows, dtype=np.float32))

    def _publish(self, scan: np.ndarray) -> None:
        self.scans_this_connection += 1
        with self._lock:
            self._latest_scan = scan
            self._latest_scan_time = time.monotonic()

    def _teardown_device(self) -> None:
        lidar = self.lidar
        self.lidar = None
        if lidar is None:
            return
        try:
            lidar.stop()
            lidar.stop_motor()
            lidar.disconnect()
        except Exception:  # pragma: no cover - hardware dependent
            pass


def angle_range(scan: np.ndarray, min_angle_deg: float, max_angle_deg: float) -> np.ndarray:
    if scan.size == 0:
        return scan
    angles = scan[:, 0] % 360.0
    return scan[_angle_mask(angles, min_angle_deg, max_angle_deg)]


def ros_angle_range(
    scan: np.ndarray,
    min_ros_angle_deg: float,
    max_ros_angle_deg: float,
    raw_angle_for_ros_zero_deg: float = 180.0,
) -> np.ndarray:
    """Select a ROS-style angular sector from raw RPLidar measurements.

    ``rplidar-roboticia`` returns the device's raw clockwise heading.  With the
    original contest launch settings (RPLidar ROS ``inverted=False`` and zero
    TF yaw), the official driver publishes ``ros_angle = 180 - raw_angle``.
    ``raw_angle_for_ros_zero_deg`` keeps the same physical sector configurable
    when the sensor is mounted with a different yaw on the Windows vehicle.
    """
    if scan.size == 0:
        return scan
    ros_angles = (raw_angle_for_ros_zero_deg - scan[:, 0]) % 360.0
    return scan[_angle_mask(ros_angles, min_ros_angle_deg, max_ros_angle_deg)]


def average_distance_mm_in_ros_sector(
    scan: np.ndarray | None,
    min_ros_angle_deg: float,
    max_ros_angle_deg: float,
    raw_angle_for_ros_zero_deg: float = 180.0,
) -> float | None:
    """Return the source-compatible mean range for one ROS angular sector."""
    if scan is None or len(scan) == 0:
        return None
    sector = ros_angle_range(
        scan,
        min_ros_angle_deg,
        max_ros_angle_deg,
        raw_angle_for_ros_zero_deg,
    )
    if len(sector) == 0:
        return None
    distances = sector[:, 1]
    valid = distances[np.isfinite(distances) & (distances > 0.0)]
    if len(valid) == 0:
        return None
    return float(np.mean(valid))


def _angle_mask(angles: np.ndarray, min_angle_deg: float, max_angle_deg: float) -> np.ndarray:
    min_angle = min_angle_deg % 360.0
    max_angle = max_angle_deg % 360.0
    if min_angle <= max_angle:
        return (angles >= min_angle) & (angles <= max_angle)
    return (angles >= min_angle) | (angles <= max_angle)


# --- Vehicle mounting convention -------------------------------------------
# Confirmed on the physical car with lidar_probe.py: the RPLidar raw heading
# points forward at raw 180 deg, left at raw 90 deg, right at raw 270 deg, and
# rear at raw 0/360 deg.  The ROS convention used elsewhere is ros = 180 - raw,
# so forward is ROS 0, left ROS +90, right ROS -90 (matches
# ``raw_angle_for_ros_zero_deg = 180.0``).
FORWARD_RAW_ANGLE_DEG = 180.0
LEFT_RAW_ANGLE_DEG = 90.0
RIGHT_RAW_ANGLE_DEG = 270.0
REAR_RAW_ANGLE_DEG = 0.0
DEFAULT_RAW_ANGLE_FOR_ROS_ZERO_DEG = 180.0


@dataclass(frozen=True)
class NearestObstacle:
    """Closest valid LiDAR return, in both raw and ROS angle conventions."""

    raw_angle_deg: float
    ros_angle_deg: float
    distance_mm: float


def raw_to_ros_angle_deg(
    raw_angle_deg: float,
    raw_angle_for_ros_zero_deg: float = DEFAULT_RAW_ANGLE_FOR_ROS_ZERO_DEG,
) -> float:
    """Convert a raw heading to the signed ROS angle (forward=0, left=+, right=-)."""
    ros = (raw_angle_for_ros_zero_deg - raw_angle_deg) % 360.0
    return ros - 360.0 if ros > 180.0 else ros


def nearest_obstacle(
    scan: np.ndarray | None,
    max_range_mm: float | None = None,
    raw_angle_for_ros_zero_deg: float = DEFAULT_RAW_ANGLE_FOR_ROS_ZERO_DEG,
) -> NearestObstacle | None:
    """Return the closest valid point in a scan, or ``None`` if there is none.

    This is the reusable form of lidar_probe.py's ``nearest`` report: it filters
    out zero/non-finite ranges (and anything beyond ``max_range_mm``) and reports
    where the nearest remaining point is.
    """
    if scan is None or len(scan) == 0:
        return None
    distances = scan[:, 1]
    mask = np.isfinite(distances) & (distances > 0.0)
    if max_range_mm is not None:
        mask = mask & (distances <= max_range_mm)
    valid = scan[mask]
    if len(valid) == 0:
        return None
    row = valid[int(np.argmin(valid[:, 1]))]
    raw_angle = float(row[0]) % 360.0
    return NearestObstacle(
        raw_angle_deg=raw_angle,
        ros_angle_deg=raw_to_ros_angle_deg(raw_angle, raw_angle_for_ros_zero_deg),
        distance_mm=float(row[1]),
    )


def direction_min_distance_mm(
    scan: np.ndarray | None,
    center_raw_angle_deg: float,
    half_width_deg: float,
) -> float | None:
    """Nearest valid range within ``+/- half_width`` of a raw direction.

    Pass ``FORWARD_RAW_ANGLE_DEG`` / ``LEFT_RAW_ANGLE_DEG`` / ``RIGHT_RAW_ANGLE_DEG``
    for the car-relative direction.  ``angle_range`` handles the 0/360 wrap, so a
    rear sector centred on 0 deg works too.
    """
    if scan is None or len(scan) == 0:
        return None
    sector = angle_range(scan, center_raw_angle_deg - half_width_deg, center_raw_angle_deg + half_width_deg)
    if len(sector) == 0:
        return None
    distances = sector[:, 1]
    valid = distances[np.isfinite(distances) & (distances > 0.0)]
    if len(valid) == 0:
        return None
    return float(np.min(valid))


def min_distance_mm_in_ros_sector(
    scan: np.ndarray | None,
    min_ros_angle_deg: float,
    max_ros_angle_deg: float,
    raw_angle_for_ros_zero_deg: float = DEFAULT_RAW_ANGLE_FOR_ROS_ZERO_DEG,
) -> float | None:
    """Nearest (min) range in a ROS sector.

    The ``average_`` variant smooths over the whole sector, so a single near
    obstacle flanked by far background reads as far.  For obstacle detection the
    minimum is what should trip the threshold.
    """
    if scan is None or len(scan) == 0:
        return None
    sector = ros_angle_range(scan, min_ros_angle_deg, max_ros_angle_deg, raw_angle_for_ros_zero_deg)
    if len(sector) == 0:
        return None
    distances = sector[:, 1]
    valid = distances[np.isfinite(distances) & (distances > 0.0)]
    if len(valid) == 0:
        return None
    return float(np.min(valid))


def analyze_obstacles(scan: np.ndarray | None, config: LidarConfig) -> ObstacleDecision:
    if scan is None or len(scan) == 0:
        return ObstacleDecision.clear()

    front = angle_range(scan, config.front_min_angle_deg, config.front_max_angle_deg)
    if len(front) == 0:
        return ObstacleDecision.clear()

    nearest = float(np.min(front[:, 1]))
    # LiDAR stopping AND slowdown are both intentionally removed: obstacles are
    # handled by the lane-change mission, so the front LiDAR never alters speed.
    # The measured distance is still reported for logging/UI, but the car always
    # runs at full speed (speed_scale=1.0, should_stop=False).
    return ObstacleDecision(
        nearest_front_mm=nearest,
        speed_scale=1.0,
        should_stop=False,
        front_points=len(front),
    )
