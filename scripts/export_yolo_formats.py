from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "models" / "yolo"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download/export an Ultralytics YOLO model to ONNX and OpenVINO.")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="Ultralytics model name or local .pt path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--skip-openvino", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    model_path = Path(args.model)
    model_source = str(model_path.resolve()) if model_path.exists() else args.model

    previous_cwd = Path.cwd()
    try:
        os.chdir(args.output_dir)
        model = YOLO(model_source)
        if not args.skip_onnx:
            model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=True, device="cpu")
        if not args.skip_openvino:
            model.export(format="openvino", imgsz=args.imgsz, half=False, int8=False, device="cpu")
    finally:
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
