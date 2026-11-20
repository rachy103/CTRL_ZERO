"""Pick the inference device and the matching model file (CPU vs NVIDIA GPU).

CPU  -> OpenVINO export  (<stem>_openvino_model), fallback <stem>.pt
GPU  -> TensorRT engine  (<stem>.engine, built ON that GPU machine), fallback <stem>.pt
OpenVINO models do not run on an NVIDIA GPU, so the file must follow the device.
"""
from __future__ import annotations

from pathlib import Path

_OPENVINO_SUFFIX = "_openvino_model"


def resolve_device(preference: str | int | None = "auto") -> str:
    """'auto' -> '0' when CUDA is available, else 'cpu'.  Anything else is kept."""
    pref = "auto" if preference is None else str(preference).strip().lower()
    if pref != "auto":
        return pref
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def is_gpu(device: str) -> bool:
    return str(device).strip().lower() not in ("cpu", "")


def model_stem(path: Path) -> Path:
    path = Path(path)
    name = path.name
    if name.endswith(_OPENVINO_SUFFIX):
        name = name[: -len(_OPENVINO_SUFFIX)]
    elif path.suffix in (".pt", ".engine", ".onnx"):
        name = path.stem
    return path.with_name(name)


def resolve_model_path(path: Path, device: str) -> Path:
    """Return the model file that fits the device (falls back to `path`)."""
    stem = model_stem(path)
    if is_gpu(device):
        candidates = [stem.with_name(stem.name + ".engine"), stem.with_name(stem.name + ".pt")]
    else:
        candidates = [stem.with_name(stem.name + _OPENVINO_SUFFIX), stem.with_name(stem.name + ".pt")]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(path)


def describe_device(device: str) -> str:
    if not is_gpu(device):
        return "cpu"
    try:
        import torch

        index = int(device) if str(device).isdigit() else 0
        return f"cuda:{index} ({torch.cuda.get_device_name(index)})"
    except Exception:
        return f"cuda:{device}"
