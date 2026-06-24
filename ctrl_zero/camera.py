from __future__ import annotations

from dataclasses import dataclass

import cv2


@dataclass
class CameraConfig:
    index: int = 0
    backend: str = "dshow"
    width: int = 0
    height: int = 0
    fps: int = 0


class CameraReader:
    def __init__(self, config: CameraConfig):
        self.config = config
        self.capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        backend = self._backend_value(self.config.backend)
        if backend is None:
            capture = cv2.VideoCapture(self.config.index)
        else:
            capture = cv2.VideoCapture(self.config.index, backend)

        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open camera index={self.config.index}, backend={self.config.backend}. "
                "Try --camera-index 0, --camera-backend any, or run scripts\\probe_cameras.py."
            )

        if self.config.width > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        if self.config.height > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if self.config.fps > 0:
            capture.set(cv2.CAP_PROP_FPS, self.config.fps)

        self.capture = capture
        backend_name = capture.getBackendName() if hasattr(capture, "getBackendName") else "unknown"
        print(f"Camera opened: index={self.config.index}, backend={backend_name}, opencv={cv2.__version__}")

    def read(self):
        if self.capture is None:
            raise RuntimeError("Camera is not open.")
        return self.capture.read()

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    @staticmethod
    def _backend_value(name: str):
        normalized = name.lower()
        if normalized == "any":
            return None
        if normalized == "dshow":
            return cv2.CAP_DSHOW
        if normalized == "msmf":
            return cv2.CAP_MSMF
        raise ValueError(f"Unsupported camera backend: {name}")
