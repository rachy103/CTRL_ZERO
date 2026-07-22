"""LiDAR forward-angle probe.

차 정면 약 50cm 지점에 물체(상자 등)를 놓고 실행하세요.
가장 가까운 점의 raw 각도가 곧 "전방 raw 각도"입니다.

  python lidar_probe.py            # COM4
  python lidar_probe.py --port COM5

출력 해석:
  - nearest: 전체 스캔에서 가장 가까운 점 (raw 각도 / ROS 각도 / 거리)
  - sector table: 주요 방향(raw 0/90/180/270 ±15도)별 최소 거리
  - 물체를 정면에 두었을 때 nearest의 raw 각도가
      ~180 이면 main.py 의 안전정지 섹터(LIDAR_FRONT 177.5~182.5)가 맞고,
      ~90  이면 미션 lane2 섹터(ROS +87.5~+92.5)가 맞습니다.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from ctrl_zero.lidar import (
    FORWARD_RAW_ANGLE_DEG,
    LEFT_RAW_ANGLE_DEG,
    REAR_RAW_ANGLE_DEG,
    RIGHT_RAW_ANGLE_DEG,
    LidarConfig,
    LidarReader,
    direction_min_distance_mm,
    nearest_obstacle,
)

# Named car-relative directions, straight from the ported lidar.py convention.
SECTORS = (
    ("front (raw 180)", FORWARD_RAW_ANGLE_DEG),
    ("left  (raw  90)", LEFT_RAW_ANGLE_DEG),
    ("right (raw 270)", RIGHT_RAW_ANGLE_DEG),
    ("rear  (raw   0)", REAR_RAW_ANGLE_DEG),
)
SECTOR_HALF_WIDTH_DEG = 15.0


def print_report(scan: np.ndarray, max_range_mm: float) -> None:
    nearest = nearest_obstacle(scan, max_range_mm=max_range_mm)
    if nearest is None:
        print(f"points={len(scan)} (no valid point within {max_range_mm:.0f}mm)")
        return

    print(
        f"nearest: raw={nearest.raw_angle_deg:6.1f}deg  ros={nearest.ros_angle_deg:+7.1f}deg  "
        f"dist={nearest.distance_mm:7.0f}mm  (points={len(scan)})"
    )
    for name, center in SECTORS:
        minimum = direction_min_distance_mm(scan, center, SECTOR_HALF_WIDTH_DEG)
        text = f"{minimum:7.0f}mm" if minimum is not None else "   --   "
        print(f"  {name}: {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print where the nearest LiDAR obstacle is.")
    parser.add_argument("--port", default="COM4")
    parser.add_argument("--interval", type=float, default=0.5, help="seconds between reports")
    parser.add_argument("--max-range-mm", type=float, default=3000.0, help="ignore points farther than this")
    args = parser.parse_args()

    import rplidar

    iter_methods = [name for name in dir(rplidar.RPLidar) if "iter" in name.lower()]
    print(f"rplidar driver: {getattr(rplidar, '__file__', '?')}")
    print(f"stream methods: {iter_methods}")

    reader = LidarReader(LidarConfig(port=args.port))
    reader.open()
    print(f"Probing LiDAR on {args.port}. Put an object ~50cm in FRONT of the car. Ctrl+C to quit.")
    try:
        while True:
            scan = reader.read_scan()
            if scan is None:
                print(
                    "waiting for scan... "
                    f"(this connection: measures={reader.measures_this_connection}, "
                    f"scans={reader.scans_this_connection})"
                )
            else:
                print_report(scan, args.max_range_mm)
            time.sleep(max(args.interval, 0.1))
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
