from __future__ import annotations

import time
from dataclasses import dataclass

from ctrl_zero.common import clamp


ARDUINO_KEYWORDS = ("arduino", "ch340", "wch", "usb serial", "cp210", "mega", "uno", "nano")


@dataclass
class ArduinoConfig:
    port: str | None = None
    baudrate: int = 9600
    connect_wait_s: float = 2.0
    drive_max_pwm: int = 255


def serial_port_rows():
    try:
        from serial.tools import list_ports
    except ModuleNotFoundError:
        return []
    return list(list_ports.comports())


def format_serial_ports() -> str:
    ports = serial_port_rows()
    if not ports:
        return "  no serial ports found"
    return "\n".join(f"  {port.device}: {port.description} [{port.hwid}]" for port in ports)


def find_arduino_port() -> str | None:
    for port in serial_port_rows():
        text = f"{port.device} {port.description} {port.hwid}".lower()
        if any(keyword in text for keyword in ARDUINO_KEYWORDS):
            return port.device
    return None


class ArduinoMotorController:
    """Serial controller for the Arduino firmware in arduino/CTRL_ZERO_Controller."""

    def __init__(self, config: ArduinoConfig, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.serial = None

    def open(self) -> None:
        if self.dry_run:
            print("Arduino motor output is disabled (dry-run).")
            return

        port = self.config.port
        if port in (None, "", "auto"):
            port = find_arduino_port()
            if port is None:
                raise RuntimeError("Could not auto-detect Arduino port.\nAvailable ports:\n" + format_serial_ports())
            self.config.port = port
            print(f"Auto-detected Arduino port: {port}")

        try:
            import serial

            self.serial = serial.Serial(port, self.config.baudrate, timeout=1, write_timeout=1)
            time.sleep(self.config.connect_wait_s)
            print(f"Arduino serial opened: port={port}, baudrate={self.config.baudrate}")
        except Exception as exc:  # pragma: no cover - hardware dependent
            raise RuntimeError(
                f"Could not open Arduino serial port {port}.\n"
                f"Original error: {exc}\n"
                "Close Arduino IDE Serial Monitor/Plotter and check the USB COM port.\n"
                "Available ports:\n" + format_serial_ports()
            ) from exc

    def send(self, steer: int, speed: int) -> None:
        steer = int(clamp(steer, -100, 100))
        speed = int(clamp(speed, -self.config.drive_max_pwm, self.config.drive_max_pwm))
        if self.serial is None:
            return
        self.serial.write(f"{steer},{speed}\n".encode("ascii"))

    def stop(self) -> None:
        self.send(0, 0)

    def close(self) -> None:
        if self.serial is not None:
            self.stop()
            self.serial.close()
            self.serial = None
