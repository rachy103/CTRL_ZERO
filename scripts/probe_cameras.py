from __future__ import annotations

import argparse

import cv2


BACKENDS = {
    "any": None,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
}


def probe(index: int, backend_name: str, width: int, height: int) -> tuple[bool, str]:
    backend = BACKENDS[backend_name]
    cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
    try:
        if not cap.isOpened():
            return False, "not opened"
        if width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = cap.read()
        if not ok or frame is None:
            return False, "opened but frame read failed"
        backend_actual = cap.getBackendName() if hasattr(cap, "getBackendName") else backend_name
        return True, f"{frame.shape[1]}x{frame.shape[0]}, backend={backend_actual}"
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe OpenCV camera indexes/backends.")
    parser.add_argument("--max-index", type=int, default=5)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    args = parser.parse_args()

    print(f"OpenCV {cv2.__version__}")
    found = []
    for index in range(args.max_index + 1):
        for backend_name in BACKENDS:
            ok, detail = probe(index, backend_name, args.width, args.height)
            status = "OK" if ok else "--"
            print(f"{status} index={index} backend={backend_name}: {detail}")
            if ok:
                found.append((index, backend_name, detail))

    if found:
        print("\nUse one of these:")
        for index, backend_name, detail in found:
            print(f"  python main.py --mode vision --backend opencv --camera-index {index} --camera-backend {backend_name}")
    else:
        print("\nNo camera opened. Close Zoom/Teams/Camera app, reconnect USB, then retry.")


if __name__ == "__main__":
    main()
