from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ctrl_zero.arduino import find_arduino_port, format_serial_ports


POT_LEFT = 569
POT_RIGHT = 447


def expected_pot(steer: int) -> int:
    steer = max(-100, min(100, int(steer)))
    t = (steer + 100.0) / 200.0
    raw = POT_LEFT + t * (POT_RIGHT - POT_LEFT)
    return int(max(min(raw, max(POT_LEFT, POT_RIGHT)), min(POT_LEFT, POT_RIGHT)))


def read_status(port, timeout_s: float) -> str | None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        line = port.readline().decode("ascii", errors="replace").strip()
        if line.startswith("STATUS "):
            return line
    return None


def status_pot(line: str | None) -> int | None:
    if not line:
        return None
    match = re.search(r"\bpot=(-?\d+)\b", line)
    return int(match.group(1)) if match else None


def send_command(port, steer: int, speed: int) -> None:
    port.write(f"{int(steer)},{int(speed)}\n".encode("ascii"))
    port.flush()


def query_status(port, timeout_s: float) -> str | None:
    port.reset_input_buffer()
    port.write(b"?\n")
    port.flush()
    return read_status(port, timeout_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check CTRL_ZERO steering edge commands against Arduino potentiometer feedback.")
    parser.add_argument("--port", default="auto", help="Arduino serial port, such as COM7. Default: auto")
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--hold", type=float, default=1.2, help="Seconds to hold each steering command.")
    parser.add_argument("--status-timeout", type=float, default=0.6)
    parser.add_argument("--speed", type=int, default=0, help="Keep this 0 for a steering-only test.")
    parser.add_argument("--skip-wait", action="store_true", help="Do not wait for Arduino reset after opening serial.")
    parser.add_argument("--dry-run", action="store_true", help="Print expected command to pot mapping without opening serial.")
    args = parser.parse_args()

    sequence = [(-100, "left max"), (0, "center"), (100, "right max"), (0, "center")]
    print("Expected mapping:")
    for steer, label in sequence:
        print(f"  {label:9s}: steer={steer:+4d} -> target_pot={expected_pot(steer)}")

    if args.dry_run:
        return

    port_name = args.port
    if port_name in ("", "auto", None):
        port_name = find_arduino_port()
        if port_name is None:
            raise SystemExit("Could not auto-detect Arduino port.\nAvailable ports:\n" + format_serial_ports())

    try:
        import serial
    except ModuleNotFoundError as exc:
        raise SystemExit("pyserial is required. Run inside the ctrl_zero conda environment.") from exc

    print(f"Opening {port_name} at {args.baud} baud")
    with serial.Serial(port_name, args.baud, timeout=0.1, write_timeout=1) as port:
        if not args.skip_wait:
            time.sleep(2.0)

        initial = query_status(port, args.status_timeout)
        if initial is None:
            print("No STATUS response. Upload the updated Arduino firmware first if you want pot feedback.")
        else:
            print(f"initial: {initial}")

        previous_pot = status_pot(initial)
        for steer, label in sequence:
            target = expected_pot(steer)
            print(f"\nCommand {label}: steer={steer:+d}, speed={args.speed}, expected_target={target}")
            send_command(port, steer, args.speed)
            time.sleep(max(args.hold, 0.0))
            status = query_status(port, args.status_timeout)
            if status is None:
                print("  status: no response")
                continue

            pot = status_pot(status)
            movement = "" if pot is None or previous_pot is None else f", delta_pot={pot - previous_pot:+d}"
            print(f"  status: {status}{movement}")
            previous_pot = pot

        send_command(port, 0, 0)
        time.sleep(0.1)
        print("\nSent final center/stop command: 0,0")


if __name__ == "__main__":
    main()
