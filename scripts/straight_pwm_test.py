"""직진 최고속 테스트: 조향 0, 속도 고정으로 아두이노에 명령을 계속 보낸다.

카메라/YOLO 없이 모터만 돌려서 auto 주행 속도와 비교하는 용도.
아두이노는 0.5초 동안 명령이 없으면 모터를 끄므로 50ms마다 다시 보낸다.

  python scripts/straight_pwm_test.py                 # COM5, 255, 3초
  python scripts/straight_pwm_test.py --seconds 2 --speed 200
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as runtime  # noqa: E402
from ctrl_zero.arduino import ArduinoConfig, ArduinoMotorController  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Straight-line PWM test (steer 0, fixed speed).")
    parser.add_argument("--port", default=runtime.ARDUINO_PORT)
    parser.add_argument("--speed", type=int, default=255)
    parser.add_argument("--steer", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--countdown", type=float, default=3.0, help="출발 전 대기 시간")
    args = parser.parse_args()

    motor = ArduinoMotorController(
        ArduinoConfig(port=args.port, baudrate=runtime.ARDUINO_BAUDRATE, drive_max_pwm=runtime.DRIVE_MAX_PWM)
    )
    motor.open()
    try:
        print(f"steer={args.steer} speed={args.speed} for {args.seconds:.1f}s, start in {args.countdown:.0f}s (Ctrl+C = stop)")
        time.sleep(max(args.countdown, 0.0))
        end = time.time() + args.seconds
        while time.time() < end:
            motor.send(args.steer, args.speed)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        motor.stop()
        motor.close()
        print("stopped")


if __name__ == "__main__":
    main()
