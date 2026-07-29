"""Offline check of the rear reverse control: bbox -> infinite lane -> lane model.

Method B, keypoints excluded.  The slot model supplies only the bbox (where to
look); the bay's side lines are then repainted as two lines spanning the FULL
frame height, and the existing lane model is run on that synthetic image.

The problem this addresses: deep in the bay the horizontal bay-end line covers
the drivable area, the lane model runs out of vertical evidence and forces a fit
from the leftovers, so the heading swings hard right at the end.  With a lane
that never ends, the fit stays well conditioned; the bay-end line is measured but
never drawn, and is used only for the far_y >= 0.9 stop.

Renders a 3-panel 30 fps video:
  LEFT   input + slot bbox + what was found (orange dashes / red horizontals /
         green infinite lanes) + the far_y stop line
  MIDDLE the synthetic frame actually fed to the lane model
  RIGHT  the lane model's own annotation + heading/offset/steer + STOP state

    python scripts/parking_infinite_lane_experiment.py captures/manual_20260729_104045
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ctrl_zero.vision.parking_line import (  # noqa: E402
    ParkingLineConfig,
    prepare_infinite_lane_frame,
)
from ctrl_zero.vision.parking_slot import ParkingSlotDetector, best_slot  # noqa: E402

DEFAULT_SLOT_MODEL = REPO_ROOT / "models" / "yolo" / "best_openvino_model"
DEFAULT_LANE_MODEL = REPO_ROOT / "models" / "yolo" / "final_openvino_model"

REVERSE_STEER_SIGN = -1
HEADING_GAIN = 1.5
OFFSET_GAIN = 40.0
STEER_LIMIT = 80
STOP_FAR_Y = 0.9


def steer_from(heading: float, offset: float) -> int:
    raw = REVERSE_STEER_SIGN * (HEADING_GAIN * heading + OFFSET_GAIN * offset)
    return int(max(-STEER_LIMIT, min(STEER_LIMIT, raw)))


def build_lane_detector(model_path: Path):
    from ctrl_zero.vision.yolo_lane import YOLOLaneConfig, YOLOLaneDetector

    return YOLOLaneDetector(YOLOLaneConfig(
        model_path=model_path, device="cpu", image_size=640,
        confidence=0.25, iou=0.45,
        class_names=("car", "obstacle", "lane1", "lane2", "traffic_light"),
    ))


def _title(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)


def _lines(img, rows):
    y = img.shape[0] - 10 - 22 * (len(rows) - 1)
    for text, color in rows:
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += 22


def process_dir(cap_dir: Path, slot_det, lane_det, fps: int, conf_min: float):
    frames = sorted(cap_dir.glob("frame_*.jpg"))
    if not frames:
        print(f"  no frames in {cap_dir}")
        return
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    out_path = cap_dir.parent / f"{cap_dir.name}_infinite_lane.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 3, h))
    cfg = ParkingLineConfig()

    n_slot = n_fit = n_stop = 0
    headings = []
    stopped = False
    for i, fp in enumerate(frames):
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        dets = slot_det.detect(frame)
        det = best_slot(dets, w)
        bbox = det.bbox if det is not None else None
        if det is not None:
            n_slot += 1

        model_frame, vis, n_lanes, h_far_y = prepare_infinite_lane_frame(frame, bbox, cfg)
        if det is not None:
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(vis, f"slot {det.confidence:.2f}", (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)

        lane = lane_det.detect(model_frame)
        heading = lane.heading_deg if lane.heading_deg is not None else None
        offset = lane.offset_norm if lane.offset_norm is not None else None
        ok = lane.confidence > 0.0 and heading is not None and offset is not None
        steer = steer_from(heading, offset) if ok else None
        if ok:
            n_fit += 1
            headings.append(heading)

        bay_end = h_far_y is not None and h_far_y >= STOP_FAR_Y
        if bay_end and not stopped:
            stopped = True
            n_stop = i
        state = "STOPPED (far_y)" if stopped else ("STOP now" if bay_end else "reversing")

        _title(vis, f"IN  [{i+1}/{len(frames)}] {fp.name}")
        _lines(vis, [
            (f"lanes={n_lanes}  hFarY={'--' if h_far_y is None else f'{h_far_y:.2f}'}/{STOP_FAR_Y}",
             (0, 255, 0) if n_lanes >= 2 else (0, 200, 255)),
            (state, (0, 0, 255) if stopped else (0, 255, 0)),
        ])
        mid = model_frame.copy()
        _title(mid, "MODEL INPUT (infinite lane)")
        res = lane.annotated.copy()
        _title(res, "LANE MODEL OUTPUT")
        _lines(res, [
            (f"conf={lane.confidence:.2f} head={heading:+.1f} off={offset:+.3f}"
             if ok else f"no fit (conf={lane.confidence:.2f})",
             (0, 255, 0) if ok else (0, 200, 255)),
            (f"steer={steer:+d}" if ok else "steer=--", (0, 255, 0) if ok else (0, 200, 255)),
        ])
        writer.write(np.hstack([vis, mid, res]))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(frames)}", flush=True)

    writer.release()
    n = max(len(frames), 1)
    hj = np.abs(np.diff(headings)) if len(headings) > 1 else np.array([0.0])
    print(f"  -> {out_path}")
    print(f"     frames={n}  slot={n_slot} ({100*n_slot/n:.0f}%)  lane fit={n_fit} ({100*n_fit/n:.0f}%)")
    print(f"     heading jitter med={np.median(hj):.2f} p90={np.percentile(hj,90):.2f} max={hj.max():.1f} deg")
    if n_stop:
        print(f"     far_y stop first triggered at frame {n_stop}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--slot-model", type=Path, default=DEFAULT_SLOT_MODEL)
    ap.add_argument("--lane-model", type=Path, default=DEFAULT_LANE_MODEL)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    slot_det = ParkingSlotDetector(args.slot_model, confidence=args.conf)
    lane_det = build_lane_detector(args.lane_model)
    for d in args.dirs:
        if not d.is_dir():
            print(f"skip: {d}")
            continue
        print(f"[{d.name}]")
        process_dir(d, slot_det, lane_det, args.fps, args.conf)


if __name__ == "__main__":
    main()
