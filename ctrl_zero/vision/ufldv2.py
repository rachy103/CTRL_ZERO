from __future__ import annotations

import importlib.util
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import cv2
import numpy as np

from ctrl_zero.common import clamp
from ctrl_zero.vision.base import LaneDetection, Point


@dataclass
class UFLDv2DetectorConfig:
    repo_dir: Path
    config_path: Path
    model_path: Path
    device: str = "cpu"
    torch_num_threads: int = 4
    control_near_y_ratio: float = 0.95
    control_far_y_ratio: float = 0.68
    min_points_per_lane: int = 5
    min_valid_y_span_ratio: float = 0.18
    default_lane_width_ratio: float = 0.48
    min_lane_width_ratio: float = 0.22
    max_lane_width_ratio: float = 0.88
    lane_width_smoothing: float = 0.18
    center_smoothing: float = 0.35
    max_missed_frames: int = 8
    show_raw_points: bool = True


class UFLDv2LaneDetector:
    """CPU-capable UFLDv2 inference wrapper.

    The default main.py config uses CULane ResNet34, replacing the previous
    Tusimple ResNet18 setup with a deeper official checkpoint.
    """

    def __init__(self, config: UFLDv2DetectorConfig):
        self.config = config
        self.torch = None
        self.cfg = None
        self.net = None
        self.device = None
        self.lane_width_px = None
        self.center_near_x = None
        self.center_far_x = None
        self.missed_frames = 0
        self._load()

    def detect(self, frame: np.ndarray) -> LaneDetection:
        torch = self.torch
        h, w = frame.shape[:2]
        near_y = int(h * self.config.control_near_y_ratio)
        far_y = int(h * self.config.control_far_y_ratio)
        frame_center_x = w / 2.0

        with torch.inference_mode():
            input_tensor = self._preprocess_frame(frame)
            pred = self.net(input_tensor)

        lanes = self._pred_to_coords(pred, w, h)
        fits = [self._fit_lane_x_of_y(lane, h) for lane in lanes]
        fits = [fit for fit in fits if fit is not None]
        left_fit, right_fit = self._select_left_right_fits(fits, frame_center_x, near_y)
        left_near = self._x_at_y(left_fit, near_y)
        right_near = self._x_at_y(right_fit, near_y)
        left_far = self._x_at_y(left_fit, far_y)
        right_far = self._x_at_y(right_fit, far_y)
        width_px = self._update_lane_width(left_near, right_near, w)

        center_near = None
        center_far = None
        if left_near is not None and right_near is not None:
            center_near = (left_near + right_near) / 2.0
            center_far = (left_far + right_far) / 2.0 if left_far is not None and right_far is not None else None
        elif left_near is not None and width_px is not None:
            center_near = left_near + width_px / 2.0
            center_far = left_far + width_px / 2.0 if left_far is not None else None
        elif right_near is not None and width_px is not None:
            center_near = right_near - width_px / 2.0
            center_far = right_far - width_px / 2.0 if right_far is not None else None

        confidence = self._confidence(left_fit, right_fit, len(lanes))
        if center_near is None:
            self.missed_frames += 1
            if self.missed_frames <= self.config.max_missed_frames and self.center_near_x is not None:
                center_near = self.center_near_x
                center_far = self.center_far_x
                confidence = min(confidence, 0.25)
        else:
            self.missed_frames = 0
            center_near = self._smooth("near", center_near)
            if center_far is not None:
                center_far = self._smooth("far", center_far)

        offset_px = center_near - frame_center_x if center_near is not None else None
        offset_norm = offset_px / max(frame_center_x, 1.0) if offset_px is not None else None
        heading_deg = None
        if center_near is not None and center_far is not None:
            heading_deg = math.degrees(math.atan2(center_far - center_near, max(near_y - far_y, 1)))

        annotated = self._annotate(frame, lanes, left_fit, right_fit, center_near, center_far, near_y, far_y, confidence)
        return LaneDetection(
            lanes=lanes,
            left_fit=left_fit,
            right_fit=right_fit,
            lane_center_near_x=center_near,
            lane_center_far_x=center_far,
            frame_center_x=frame_center_x,
            offset_px=offset_px,
            offset_norm=offset_norm,
            heading_deg=heading_deg,
            lane_width_px=width_px,
            confidence=confidence,
            mask=None,
            annotated=annotated,
        )

    def _load(self) -> None:
        if not self.config.config_path.exists():
            raise FileNotFoundError(f"UFLDv2 config not found: {self.config.config_path}")
        if not self.config.model_path.exists():
            raise FileNotFoundError(
                f"UFLDv2 weight file not found: {self.config.model_path}\n"
                "Run: python scripts/download_ufldv2_weights.py --model culane_res34\n"
                "Or set LANE_BACKEND = 'opencv' in main.py for camera/Arduino calibration."
            )

        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("torch and torchvision are required for UFLDv2. Install requirements.txt first.") from exc

        self.torch = torch
        torch.set_num_threads(self.config.torch_num_threads)
        device_name = self.config.device
        if device_name == "cuda" and not torch.cuda.is_available():
            device_name = "cpu"
        self.device = torch.device(device_name)

        self.cfg = self._load_config_py(self.config.config_path)
        self._set_anchors()
        self._install_ufldv2_common_stub()
        if str(self.config.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.config.repo_dir))
        from model.model_culane import parsingNet

        self.net = parsingNet(
            pretrained=False,
            backbone=self.cfg.backbone,
            num_grid_row=self.cfg.num_cell_row,
            num_cls_row=self.cfg.num_row,
            num_grid_col=self.cfg.num_cell_col,
            num_cls_col=self.cfg.num_col,
            num_lane_on_row=self.cfg.num_lanes,
            num_lane_on_col=self.cfg.num_lanes,
            use_aux=self.cfg.use_aux,
            input_height=self.cfg.train_height,
            input_width=self.cfg.train_width,
            fc_norm=self.cfg.fc_norm,
        ).to(self.device)

        try:
            checkpoint = torch.load(str(self.config.model_path), map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(str(self.config.model_path), map_location=self.device)

        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        compatible_state_dict = {
            key[7:] if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }
        missing, unexpected = self.net.load_state_dict(compatible_state_dict, strict=False)
        if missing:
            print(f"UFLDv2 warning: missing model keys={len(missing)}")
        if unexpected:
            print(f"UFLDv2 warning: unexpected model keys={len(unexpected)}")
        self.net.eval()
        print(f"UFLDv2 loaded: config={self.config.config_path.name}, weights={self.config.model_path}, device={self.device}")

    @staticmethod
    def _load_config_py(path: Path) -> SimpleNamespace:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import config: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        values = {name: value for name, value in vars(module).items() if not name.startswith("__")}
        return SimpleNamespace(**values)

    def _set_anchors(self) -> None:
        if self.cfg.dataset == "CULane":
            self.cfg.row_anchor = np.linspace(0.42, 1.0, self.cfg.num_row)
            self.cfg.col_anchor = np.linspace(0.0, 1.0, self.cfg.num_col)
        elif self.cfg.dataset == "Tusimple":
            self.cfg.row_anchor = np.linspace(160, 710, self.cfg.num_row) / 720.0
            self.cfg.col_anchor = np.linspace(0.0, 1.0, self.cfg.num_col)
        elif self.cfg.dataset == "CurveLanes":
            self.cfg.row_anchor = np.linspace(0.4, 1.0, self.cfg.num_row)
            self.cfg.col_anchor = np.linspace(0.0, 1.0, self.cfg.num_col)
        else:
            raise ValueError(f"Unsupported UFLDv2 dataset: {self.cfg.dataset}")

    @staticmethod
    def _install_ufldv2_common_stub() -> None:
        common_stub = types.ModuleType("utils.common")

        def initialize_weights(*models):
            for model in models:
                for module in model.modules():
                    if hasattr(module, "reset_parameters"):
                        module.reset_parameters()

        common_stub.initialize_weights = initialize_weights
        sys.modules["utils.common"] = common_stub

    def _preprocess_frame(self, frame: np.ndarray):
        resize_h = int(self.cfg.train_height / self.cfg.crop_ratio)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.cfg.train_width, resize_h), interpolation=cv2.INTER_LINEAR)
        cropped = resized[-self.cfg.train_height :, :, :]
        tensor = cropped.astype(np.float32) / 255.0
        tensor = (tensor - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        tensor = np.transpose(tensor, (2, 0, 1))
        return self.torch.from_numpy(tensor).unsqueeze(0).to(self.device)

    def _pred_to_coords(self, pred, original_width: int, original_height: int, local_width: int = 1) -> list[list[Point]]:
        torch = self.torch
        _, num_grid_row, num_cls_row, _ = pred["loc_row"].shape
        _, num_grid_col, num_cls_col, _ = pred["loc_col"].shape

        loc_row = pred["loc_row"].detach().cpu()
        loc_col = pred["loc_col"].detach().cpu()
        max_indices_row = loc_row.argmax(1)
        valid_row = pred["exist_row"].detach().cpu().argmax(1)
        max_indices_col = loc_col.argmax(1)
        valid_col = pred["exist_col"].detach().cpu().argmax(1)

        coords = []
        for lane_idx in (1, 2):
            lane = []
            if valid_row[0, :, lane_idx].sum() > num_cls_row / 2:
                for anchor_idx in range(valid_row.shape[1]):
                    if valid_row[0, anchor_idx, lane_idx]:
                        center_grid = int(max_indices_row[0, anchor_idx, lane_idx])
                        grid_ids = torch.tensor(
                            list(range(max(0, center_grid - local_width), min(num_grid_row - 1, center_grid + local_width) + 1))
                        )
                        x = (loc_row[0, grid_ids, anchor_idx, lane_idx].softmax(0) * grid_ids.float()).sum() + 0.5
                        x = x / (num_grid_row - 1) * original_width
                        y = self.cfg.row_anchor[anchor_idx] * original_height
                        lane.append((int(x), int(y)))
                coords.append(lane)

        for lane_idx in (0, 3):
            lane = []
            if valid_col[0, :, lane_idx].sum() > num_cls_col / 4:
                for anchor_idx in range(valid_col.shape[1]):
                    if valid_col[0, anchor_idx, lane_idx]:
                        center_grid = int(max_indices_col[0, anchor_idx, lane_idx])
                        grid_ids = torch.tensor(
                            list(range(max(0, center_grid - local_width), min(num_grid_col - 1, center_grid + local_width) + 1))
                        )
                        y = (loc_col[0, grid_ids, anchor_idx, lane_idx].softmax(0) * grid_ids.float()).sum() + 0.5
                        y = y / (num_grid_col - 1) * original_height
                        x = self.cfg.col_anchor[anchor_idx] * original_width
                        lane.append((int(x), int(y)))
                coords.append(lane)

        return [lane for lane in coords if len(lane) >= self.config.min_points_per_lane]

    def _fit_lane_x_of_y(self, points: Sequence[Point], image_height: int):
        if len(points) < self.config.min_points_per_lane:
            return None
        xs = np.array([p[0] for p in points], dtype=np.float64)
        ys = np.array([p[1] for p in points], dtype=np.float64)
        if ys.max() - ys.min() < image_height * self.config.min_valid_y_span_ratio:
            return None
        try:
            return np.polyfit(ys, xs, deg=1)
        except np.linalg.LinAlgError:
            return None

    @staticmethod
    def _x_at_y(fit, y: float) -> float | None:
        if fit is None:
            return None
        return float(fit[0] * y + fit[1])

    def _select_left_right_fits(self, fits, frame_center_x: float, near_y: int):
        left_candidates = []
        right_candidates = []
        for fit in fits:
            x_near = self._x_at_y(fit, near_y)
            if x_near is None:
                continue
            if x_near < frame_center_x:
                left_candidates.append((abs(frame_center_x - x_near), fit))
            else:
                right_candidates.append((abs(x_near - frame_center_x), fit))
        left_fit = min(left_candidates, default=(None, None), key=lambda item: item[0])[1]
        right_fit = min(right_candidates, default=(None, None), key=lambda item: item[0])[1]
        return left_fit, right_fit

    def _update_lane_width(self, left_x, right_x, image_width: int):
        if self.lane_width_px is None:
            self.lane_width_px = image_width * self.config.default_lane_width_ratio
        if left_x is None or right_x is None:
            return self.lane_width_px
        new_width = right_x - left_x
        if image_width * self.config.min_lane_width_ratio <= new_width <= image_width * self.config.max_lane_width_ratio:
            alpha = self.config.lane_width_smoothing
            self.lane_width_px = alpha * new_width + (1.0 - alpha) * self.lane_width_px
        return self.lane_width_px

    def _smooth(self, which: str, value: float) -> float:
        if which == "near":
            previous = self.center_near_x
            smoothed = value if previous is None else self.config.center_smoothing * value + (1.0 - self.config.center_smoothing) * previous
            self.center_near_x = smoothed
            return smoothed
        previous = self.center_far_x
        smoothed = value if previous is None else self.config.center_smoothing * value + (1.0 - self.config.center_smoothing) * previous
        self.center_far_x = smoothed
        return smoothed

    @staticmethod
    def _confidence(left_fit, right_fit, lane_count: int) -> float:
        confidence = 0.0
        if left_fit is not None:
            confidence += 0.42
        if right_fit is not None:
            confidence += 0.42
        confidence += min(lane_count, 4) * 0.04
        return float(clamp(confidence, 0.0, 1.0))

    def _annotate(self, frame, lanes, left_fit, right_fit, center_near, center_far, near_y, far_y, confidence):
        vis = frame.copy()
        if self.config.show_raw_points:
            colors = [(0, 255, 255), (0, 220, 0), (255, 180, 0), (255, 0, 255)]
            for idx, lane in enumerate(lanes):
                for point in lane:
                    cv2.circle(vis, point, 3, colors[idx % len(colors)], -1)
        self._draw_fit(vis, left_fit, near_y, far_y, (0, 255, 0))
        self._draw_fit(vis, right_fit, near_y, far_y, (0, 255, 0))
        h, w = vis.shape[:2]
        cv2.line(vis, (w // 2, h - 1), (w // 2, int(h * 0.55)), (0, 0, 255), 1)
        if center_near is not None and center_far is not None:
            cv2.line(vis, (int(center_near), near_y), (int(center_far), far_y), (255, 255, 0), 2)
            cv2.circle(vis, (int(center_near), near_y), 6, (255, 255, 0), -1)
        cv2.putText(vis, f"ufldv2 conf={confidence:.2f}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(vis, f"ufldv2 conf={confidence:.2f}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        return vis

    def _draw_fit(self, image, fit, near_y, far_y, color) -> None:
        if fit is None:
            return
        h, w = image.shape[:2]
        x_near = int(clamp(self._x_at_y(fit, near_y), 0, w - 1))
        x_far = int(clamp(self._x_at_y(fit, far_y), 0, w - 1))
        cv2.line(image, (x_near, near_y), (x_far, far_y), color, 3)
