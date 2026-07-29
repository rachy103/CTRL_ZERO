from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import cv2

from ctrl_zero.arduino import ArduinoConfig, ArduinoMotorController, format_serial_ports
from ctrl_zero.camera import CameraConfig, CameraReader
from ctrl_zero.common import clamp
from ctrl_zero.control import DriveCommand, DriveConfig, DriveController
from ctrl_zero.lidar import LidarConfig, LidarReader, ObstacleDecision, analyze_obstacles
from ctrl_zero.logger import DriveLogger, LogConfig
from ctrl_zero.obstacles import ContestObstacleMission, ContestObstacleMissionConfig
from ctrl_zero.parking_logic import ParkingLogicConfig, ParkingLogicMission
from ctrl_zero.parking_timed import TimedParkingConfig, TimedParkingMission
from ctrl_zero.vision.parking_line import (
    ParkingLineConfig,
    prepare_infinite_lane_frame,
    prepare_parking_line_frame,
    vanishing_point,
)
from ctrl_zero.vision.parking_slot import ParkingSlotDetector, best_slot
from ctrl_zero.safety import SafetyDecision, build_safety_decision
from ctrl_zero.traffic_light import traffic_light_object
from ctrl_zero.ui import draw_status
from ctrl_zero.vision.preprocess import BirdEyeConfig, LanePreprocessor, ROICropConfig


# =============================================================================
# USER TUNING PARAMETERS
# 이 영역만 수정해도 카메라, 라이다, Arduino, 모델, 조향/속도 튜닝을 바꿀 수 있게 둡니다.
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

# Runtime mode: "vision"은 모터 미출력, "manual"은 키보드 수동, "auto"는 차선+라이다 자동 주행입니다.
RUN_MODE = "auto"
LANE_BACKEND = "yolo"

# Camera
CAMERA_INDEX = 1
CAMERA_BACKEND = "dshow"  # Windows: "dshow" 권장. 필요 시 "msmf" 또는 "any".
CAMERA_WIDTH = 0
CAMERA_HEIGHT = 0
CAMERA_FPS = 0

# Arduino / motor
USE_ARDUINO = True
ARDUINO_PORT = "COM3"
ARDUINO_BAUDRATE = 115200
DRIVE_MAX_PWM = 255

# LiDAR
USE_LIDAR = True
LIDAR_PORT = "COM4"
LIDAR_POLL_EVERY_N_FRAMES = 1
LIDAR_STOP_DISTANCE_MM = 450.0
LIDAR_SLOW_DISTANCE_MM = 900.0
LIDAR_MIN_SPEED_SCALE = 1.0

# Timed obstacle mode: shift to the opposite lane for a fixed time, then hand
# control straight back to normal lane following (no counter-steer pass phase).
OBSTACLE_MISSION_ENABLED = True
LIDAR_RAW_ANGLE_FOR_ROS_ZERO_DEG = 180.0

OBSTACLE_DISTANCE_MM = 2000.0
OBSTACLE_TRIGGER_FRAMES = 2
# Choose the avoidance proximity source with USE_LIDAR:
#   True  -> LiDAR front sector (OBSTACLE_LIDAR_* / OBSTACLE_DISTANCE_MM)
#   False -> camera only, using the car bbox bottom edge (far_y).  Trigger once
#            far_y / frame_height >= this ratio.  A closer car sits lower in the
#            frame, so RAISE it to react only when nearer, LOWER it to react
#            farther, 0.0 = any central car triggers.  Watch the [OBS] log's
#            far_y= value to tune it to your camera geometry.
OBSTACLE_CAMERA_TRIGGER_FAR_Y_RATIO = 0.0

OBSTACLE_LANE2_SHIFT_STEER = -100
OBSTACLE_LANE1_SHIFT_STEER = 100
OBSTACLE_SHIFT_OUT_SECONDS = 1.5
# Extra shift-out time per unit of steering at detection (seconds per steer unit).
# duration = OBSTACLE_SHIFT_OUT_SECONDS + |steer_at_detection| * this. 0 disables.
OBSTACLE_SHIFT_OUT_STEER_WEIGHT = 0.003
OBSTACLE_RETRIGGER_COOLDOWN_SECONDS = 0.0
# Single source of truth for the front LiDAR sector, in ROS angles (forward=0,
# left=+, right=-).  Drives BOTH the obstacle-avoidance trigger AND the UI/log
# nearest-range display, so the number on screen always matches the wedge the
# trigger looks at.  ROS = LIDAR_RAW_ANGLE_FOR_ROS_ZERO_DEG - raw_angle.
OBSTACLE_LIDAR_MIN_ROS_DEG = -20.0
OBSTACLE_LIDAR_MAX_ROS_DEG = 20.0
# Print the full per-frame obstacle diagnostic (phase, lane, every gate, trigger
# reason, far_y value) to the console.  Set False to only log detect/clear.
OBSTACLE_LOG_EVERY_FRAME = True

# --mode parking_sim: LiDAR + 시간 기반 직각 주차/출차 설정 -----------------
# LiDAR 각도는 ROS 기준이다: 정면=180도, 왼쪽=+90도, 오른쪽=-90도(=270도).

# 첫 번째 주차 차량을 확인하는 오른쪽 측면 각도 범위.
PARKING_SEARCH_SECTOR_ROS_DEG = (270.0, 275.0)
# 직선 후진 중 왼쪽 주차 차량과의 거리를 확인하는 각도 범위 (min<max 폭이 있어야 함!).
PARKING_LEFT_SECTOR_ROS_DEG = (70.0, 90.0)
# 직선 후진 중 오른쪽 주차 차량과의 거리를 확인하는 각도 범위 (min<max 폭이 있어야 함!).
PARKING_RIGHT_SECTOR_ROS_DEG = (270.0, 290.0)

# 탐색 범위의 평균 거리가 이 값(mm) 이하면 첫 번째 차량을 감지한 것으로 판단한다.
PARKING_SEARCH_DETECT_MM = 2500.0
# 직선 후진 중 좌우 평균 거리가 모두 이 값(mm) 이하면 정상 주차 완료로 판단한다.
PARKING_SIDE_DETECT_MM = 1500.0

# 직진에 사용하는 조향 명령값
PARKING_STRAIGHT_STEER = 0
# 첫 차량 탐색, 접근, 출차에 사용하는 전진 PWM 속도.
PARKING_SEARCH_SPEED = 150
# 우측 곡선 후진(ARC_BACK)에 사용하는 후진 PWM 속도.
PARKING_REVERSE_STRAIGHT_SPEED = -150
# 직선 후진(STRAIGHT_BACK)에 사용하는 후진 PWM 속도 (곡선 후진과 별도, 음수=후진).
# 주의: 펌웨어 DRIVE_DEADBAND_PWM=8 이라 |값|이 8 이하면 안 움직인다.
PARKING_SIM_STRAIGHT_BACK_SPEED = -30

# 첫 번째 차량의 뒤쪽 끝이 탐색 범위에서 사라진 뒤 직진하는 시간(초).
# 늘리면 전진 스윙을 더 앞쪽에서 시작하고, 줄이면 더 뒤쪽에서 시작한다.
PARKING_SIM_APPROACH_S = 3.8
# 왼쪽 최대 조향으로 전진하여 차 앞부분을 주차 공간 반대쪽으로 벌리는 시간(초).
PARKING_SIM_SWING_S = 1.70
# 오른쪽 최대 조향으로 후진하여 차체를 주차 공간 안으로 회전시키는 시간(초).
PARKING_SIM_ARC_S = 5.0
# 전진 스윙 후 정지한 채 바퀴를 오른쪽으로 미리 돌려 놓는 시간(초).
PARKING_SIM_GEAR_PAUSE_SECONDS = 1.0
# 좌우 차량 감지에 실패했을 때 직선 후진을 강제로 끝내는 최대 시간(초).
# 이 타임아웃이 발생하면 자동 출차하지 않고 정지 상태를 유지한다.
PARKING_SIM_STRAIGHT_BACK_MAX_SECONDS = 10.0

## 출차
# 출차 시작 전 정지 시간(초).
PARKING_SIM_PARKED_PAUSE_S = 4.0
# 조향 0으로 직진 출차하는 시간(초).
PARKING_SIM_EXIT_APPROACH_S = 2.0
# 우측 조향을 유지하며 곡선 전진 출차하는 시간(초).
PARKING_SIM_EXIT_ARC_S = 7.5

# 전진 스윙 단계의 조향값: -100은 왼쪽 최대 조향.
PARKING_SIM_SWING_STEER = -100
# 곡선 후진 단계의 조향값: +100은 오른쪽 최대 조향.
PARKING_SIM_ARC_STEER = 100
# 곡선 전진 출차 단계의 조향값: +100은 오른쪽 최대 조향.
PARKING_SIM_EXIT_ARC_STEER = 100

# 주차 및 출차 상태 머신에 전달할 전진 속도.
PARKING_SIM_FORWARD_SPEED = PARKING_SEARCH_SPEED
# 주차 상태 머신에 전달할 후진 속도.
PARKING_SIM_REVERSE_SPEED = PARKING_REVERSE_STRAIGHT_SPEED

# 첫 차량이 연속으로 감지되어야 하는 최소 시간(초). 0이면 즉시 인정한다.
PARKING_SIM_DETECT_HOLD_SECONDS = 0.0
# 첫 차량의 미감지 상태가 유지되어야 하는 최소 시간(초).
# 0이면 한 번의 미감지로 즉시 차량의 뒤쪽 끝을 통과했다고 판단한다.
PARKING_SIM_CLEAR_HOLD_SECONDS = 0.0
# 직선 후진 중 좌우 차량이 모두 가까운 상태로 유지되어야 하는 최소 시간(초).
# 0이면 한 번의 측정만으로 정상 주차 완료를 판단한다.
PARKING_SIM_SIDE_HOLD_SECONDS = 0.0

## 첫 직진(SEARCH/APPROACH) 카메라 헤딩 가이드 — 모델 기반.
# 왼쪽 ROI만 남기고(나머지 검게), horizontal 흰 선을 모델 추론 전에 지운 뒤,
# 기존 YOLO 차선 모델을 돌려 나온 driving area의 heading/offset으로 조향한다.
# (전처리는 parking_logic과 동일한 ctrl_zero/vision/parking_line.py를 공유)
PARKING_SIM_USE_CAMERA_GUIDE = True
PARKING_GUIDE_ROI_LEFT_RATIO = 0.5        # 왼쪽에서 이 비율만큼만 모델에 보임
PARKING_GUIDE_ROI_TOP_RATIO = 0.20        # 위쪽 이 비율은 잘라냄(원경/천장)
# 전방캠은 원근 때문에 차선이 이미지상 많이 기울어 보임 -> 가이드 전용 각도 임계.
# (parking_logic 후방캠의 PARKING_LINE_*와 별개)
PARKING_GUIDE_VERTICAL_MAX_DEV_DEG = 75.0   # 수직에서 이 이내면 차선 대시로 유지+연장
PARKING_GUIDE_HORIZONTAL_MIN_DEV_DEG = 85.0 # 이 이상만 horizontal로 보고 지움
# 조향 = 소실점(VP) 홀드. 두 선이 실제 평행이면, 차가 그 선들과 평행한 순간
# 두 선의 이미지 교점(소실점)이 화면 중앙 x에 온다 — 측방 위치와 무관, 요(yaw)만
# 반영되는 불변량이라 기준값 캘리브레이션이 필요 없다.
#   steer = VP_GAIN * (vp_x - cx) / cx
PARKING_GUIDE_VP_GAIN = 60.0              # 반대로 꺾이면 부호 반전
PARKING_GUIDE_STEER_LIMIT = 60            # 가이드 조향 클램프


# --mode parking_logic: 후방카메라 주차선 후진 진입 ---------------------------
# 첫 차 감지 -> x1초 직선 후진 -> 좌 풀락 6.7초 주행 -> 후방캠 하단 ROI에서
# 끊긴 주차선을 연장해 그린 프레임으로 YOLO(final openvino)를 돌려 조향하며 후진.
PARKING_LOGIC_REAR_CAMERA_INDEX = 0       # 후방 카메라 인덱스 (probe_cameras.py로 확인)
PARKING_LOGIC_BACK_S = 0.0                # x1: 첫 차 감지 후 우회전 후진 시간(초)
PARKING_LOGIC_BACK_STEER = 100            # 후진 중 조향: +100 = 우 최대
PARKING_LOGIC_TURN_S = 5.7                # 좌 풀락 주행 시간(초)
PARKING_LOGIC_TURN_STEER = -100           # 좌 최대 조향
PARKING_LOGIC_SEARCH_SPEED = 150
PARKING_LOGIC_BACK_SPEED = -150
PARKING_LOGIC_TURN_SPEED = 150
PARKING_LOGIC_FOLLOW_SPEED = -80          # 주차선 추종 후진 속도
PARKING_LOGIC_FOLLOW_MAX_S = 20.0         # 추종 후진 안전 타임아웃(초)
# 후진(LINE_FOLLOW) 시작 후 이 시간 동안은 모든 센서를 무시하고 직선 후진만 한다.
# 이 구간에는 슬롯 모델·차선 모델·Hough를 아예 돌리지 않는다.
PARKING_LOGIC_FOLLOW_OPEN_LOOP_S = 5.7
# far_y 정지 후 카메라 조향 없이 직선으로 더 후진하는 시간(초).
PARKING_LOGIC_FINAL_BACK_S = 3.0
## 출차 (parking_sim과 동일한 3단계: 정차 -> 직진 전진 -> 우 풀락 곡선 전진)
PARKING_LOGIC_PARKED_PAUSE_S = PARKING_SIM_PARKED_PAUSE_S
PARKING_LOGIC_EXIT_APPROACH_S = PARKING_SIM_EXIT_APPROACH_S
PARKING_LOGIC_EXIT_ARC_S = PARKING_SIM_EXIT_ARC_S
PARKING_LOGIC_EXIT_ARC_STEER = PARKING_SIM_EXIT_ARC_STEER
# 주차선 추종 조향 = sign * (heading_gain*heading_deg + offset_gain*offset_norm)
# 후진이라 조향 부호가 반대일 수 있음: 반대로 꺾이면 SIGN을 +1로.
PARKING_LOGIC_REVERSE_STEER_SIGN = -1
PARKING_LOGIC_HEADING_GAIN = 1.7
PARKING_LOGIC_OFFSET_GAIN = 40.0
PARKING_LOGIC_STEER_LIMIT = 80
# SEARCH 직진 구간의 소실점(VP) 가이드 — 전방캠 사용(parking_sim과 동일 기하).
PARKING_LOGIC_GUIDE_VP_GAIN = 60.0
PARKING_LOGIC_GUIDE_STEER_LIMIT = 60
# 후진 정지 보강: 후방캠에서 흰색 horizontal line(주차 베이 끝선)의 far_y 비율이
# 이 값 이상이면 정지한다. 라이다 측면 감지만으로는 차선이 시야에서 사라진 뒤
# 헤딩 제어가 망가지므로, 이쪽이 더 직접적인 신호다.
PARKING_LOGIC_STOP_FAR_Y_RATIO = 0.8

# --best-model: 슬롯 모델(bbox)로 탐색 범위를 잡고, 세로선을 프레임 전체로 연장한
# 합성 프레임을 기존 차선 모델에 입력해 후진을 제어한다. keypoint는 쓰지 않는다.
PARKING_LOGIC_BEST_MODEL_PATH = BASE_DIR / "models" / "yolo" / "parking_v2_openvino_model"
PARKING_LOGIC_BEST_CONF = 0.4
# 슬롯 bbox 미검출 프레임: 직전 bbox를 중심 고정한 채 이 배율로 키워서 사용한다.
# 주차선 전처리 (ctrl_zero/vision/parking_line.py):
# ① horizontal 흰 선은 모델 추론 전에 지우고(검게), ② 끊긴 세로 대시는 연장해
# 이어 그린 뒤, 그 프레임으로 기존 YOLO 모델을 돌린다.
PARKING_LINE_ROI_TOP_RATIO = 0.1         # 프레임 위쪽 이 비율은 잘라냄(검게)
PARKING_LINE_WHITE_THRESH = 180           # 흰 선 밝기 임계
PARKING_LINE_HOUGH_THRESHOLD = 30
PARKING_LINE_MIN_LINE_FRAC = 0.12         # 대시 최소 길이 = ROI 높이 비율
PARKING_LINE_MAX_GAP_PX = 40
PARKING_LINE_VERTICAL_MAX_DEV_DEG = 45.0  # 수직에서 이 이내 = 주차선 대시(유지+연장)
# 같은 물리적 주차선의 대시들을 하나로 묶는 기준: ROI 바닥 교차점 x와 각도가 이내면 동일선.
PARKING_LINE_MERGE_X_TOL_PX = 40
PARKING_LINE_MERGE_ANGLE_TOL_DEG = 8.0
# 증거(대시 총 길이)가 강한 상위 N개 클러스터만 연장해 그림. 베이 좌우 2줄 = 2.
PARKING_LINE_MAX_LINES = 2
PARKING_LINE_HORIZONTAL_MIN_DEV_DEG = 60.0  # 수직에서 이 이상 = horizontal(지움)
# 이 높이 비율보다 아래(화면 맨 끝쪽)의 horizontal은 far_y로 인정하지 않는다.
# 후방 시야 맨 아래는 베이 끝선이 아니라 바닥 마감선/매트 경계라 far_y=1.00으로
# 즉시 정지시켜 버렸다. 따라서 far_y는 이 값을 넘을 수 없다.
PARKING_LINE_HORIZONTAL_MAX_Y_RATIO = 0.9
# 베이 끝선으로 인정할 가로선의 최소 길이(프레임 폭 대비 비율).
# 바닥의 옅은 얼룩이 짧은 "가로선"으로 잡혀 far_y를 띄우는 것을 막는다.
PARKING_LINE_HORIZONTAL_MIN_LEN_FRAC = 0.5
PARKING_LINE_EXTEND_THICKNESS_PX = 7
PARKING_LINE_ERASE_THICKNESS_PX = 15      # horizontal 지울 때 붓 두께


def build_parking_logic_mission() -> ParkingLogicMission:
    return ParkingLogicMission(
        ParkingLogicConfig(
            raw_angle_for_ros_zero_deg=LIDAR_RAW_ANGLE_FOR_ROS_ZERO_DEG,
            search_sector_ros_deg=PARKING_SEARCH_SECTOR_ROS_DEG,
            left_sector_ros_deg=PARKING_LEFT_SECTOR_ROS_DEG,
            right_sector_ros_deg=PARKING_RIGHT_SECTOR_ROS_DEG,
            search_detect_mm=PARKING_SEARCH_DETECT_MM,
            side_detect_mm=PARKING_SIDE_DETECT_MM,
            detect_hold_s=PARKING_SIM_DETECT_HOLD_SECONDS,
            straight_steer=PARKING_STRAIGHT_STEER,
            back_steer=PARKING_LOGIC_BACK_STEER,
            turn_steer=PARKING_LOGIC_TURN_STEER,
            search_speed=PARKING_LOGIC_SEARCH_SPEED,
            back_speed=PARKING_LOGIC_BACK_SPEED,
            turn_speed=PARKING_LOGIC_TURN_SPEED,
            follow_speed=PARKING_LOGIC_FOLLOW_SPEED,
            back_s=PARKING_LOGIC_BACK_S,
            turn_s=PARKING_LOGIC_TURN_S,
            follow_max_s=PARKING_LOGIC_FOLLOW_MAX_S,
            follow_open_loop_s=PARKING_LOGIC_FOLLOW_OPEN_LOOP_S,
            final_back_s=PARKING_LOGIC_FINAL_BACK_S,
            parked_pause_s=PARKING_LOGIC_PARKED_PAUSE_S,
            exit_approach_s=PARKING_LOGIC_EXIT_APPROACH_S,
            exit_arc_s=PARKING_LOGIC_EXIT_ARC_S,
            exit_arc_steer=PARKING_LOGIC_EXIT_ARC_STEER,
        )
    )


def build_parking_line_config() -> ParkingLineConfig:
    return ParkingLineConfig(
        roi_top_ratio=PARKING_LINE_ROI_TOP_RATIO,
        white_thresh=PARKING_LINE_WHITE_THRESH,
        hough_threshold=PARKING_LINE_HOUGH_THRESHOLD,
        min_line_frac=PARKING_LINE_MIN_LINE_FRAC,
        max_gap_px=PARKING_LINE_MAX_GAP_PX,
        vertical_max_dev_deg=PARKING_LINE_VERTICAL_MAX_DEV_DEG,
        horizontal_min_dev_deg=PARKING_LINE_HORIZONTAL_MIN_DEV_DEG,
        horizontal_max_y_ratio=PARKING_LINE_HORIZONTAL_MAX_Y_RATIO,
        horizontal_min_len_frac=PARKING_LINE_HORIZONTAL_MIN_LEN_FRAC,
        merge_x_tol_px=PARKING_LINE_MERGE_X_TOL_PX,
        merge_angle_tol_deg=PARKING_LINE_MERGE_ANGLE_TOL_DEG,
        max_lines=PARKING_LINE_MAX_LINES,
        extend_thickness_px=PARKING_LINE_EXTEND_THICKNESS_PX,
        erase_thickness_px=PARKING_LINE_ERASE_THICKNESS_PX,
    )


def build_parking_guide_line_config() -> ParkingLineConfig:
    # parking_sim 첫 직진 가이드용 전처리: 왼쪽 ROI + horizontal 지움 + 대시 연장.
    # 검출 파라미터(흰색 임계 등)는 parking_logic의 PARKING_LINE_*를 공유한다.
    return ParkingLineConfig(
        roi_top_ratio=PARKING_GUIDE_ROI_TOP_RATIO,
        keep_left_ratio=PARKING_GUIDE_ROI_LEFT_RATIO,
        white_thresh=PARKING_LINE_WHITE_THRESH,
        hough_threshold=PARKING_LINE_HOUGH_THRESHOLD,
        min_line_frac=PARKING_LINE_MIN_LINE_FRAC,
        max_gap_px=PARKING_LINE_MAX_GAP_PX,
        vertical_max_dev_deg=PARKING_GUIDE_VERTICAL_MAX_DEV_DEG,
        horizontal_min_dev_deg=PARKING_GUIDE_HORIZONTAL_MIN_DEV_DEG,
        merge_x_tol_px=PARKING_LINE_MERGE_X_TOL_PX,
        merge_angle_tol_deg=PARKING_LINE_MERGE_ANGLE_TOL_DEG,
        max_lines=PARKING_LINE_MAX_LINES,
        extend_thickness_px=PARKING_LINE_EXTEND_THICKNESS_PX,
        erase_thickness_px=PARKING_LINE_ERASE_THICKNESS_PX,
    )


def build_timed_parking_mission() -> TimedParkingMission:
    return TimedParkingMission(
        TimedParkingConfig(
            enabled=True,
            raw_angle_for_ros_zero_deg=LIDAR_RAW_ANGLE_FOR_ROS_ZERO_DEG,
            search_sector_ros_deg=PARKING_SEARCH_SECTOR_ROS_DEG,
            left_sector_ros_deg=PARKING_LEFT_SECTOR_ROS_DEG,
            right_sector_ros_deg=PARKING_RIGHT_SECTOR_ROS_DEG,
            search_detect_mm=PARKING_SEARCH_DETECT_MM,
            side_detect_mm=PARKING_SIDE_DETECT_MM,
            detect_hold_s=PARKING_SIM_DETECT_HOLD_SECONDS,
            clear_hold_s=PARKING_SIM_CLEAR_HOLD_SECONDS,
            side_hold_s=PARKING_SIM_SIDE_HOLD_SECONDS,
            swing_steer=PARKING_SIM_SWING_STEER,
            arc_steer=PARKING_SIM_ARC_STEER,
            straight_steer=PARKING_STRAIGHT_STEER,
            forward_speed=PARKING_SIM_FORWARD_SPEED,
            reverse_speed=PARKING_SIM_REVERSE_SPEED,
            straight_back_speed=PARKING_SIM_STRAIGHT_BACK_SPEED,
            approach_s=PARKING_SIM_APPROACH_S,
            swing_s=PARKING_SIM_SWING_S,
            gear_pause_s=PARKING_SIM_GEAR_PAUSE_SECONDS,
            arc_s=PARKING_SIM_ARC_S,
            straight_back_max_s=PARKING_SIM_STRAIGHT_BACK_MAX_SECONDS,
            parked_pause_s=PARKING_SIM_PARKED_PAUSE_S,
            exit_approach_s=PARKING_SIM_EXIT_APPROACH_S,
            exit_arc_s=PARKING_SIM_EXIT_ARC_S,
            exit_arc_steer=PARKING_SIM_EXIT_ARC_STEER,
        )
    )


# Traffic light stop gating. Stop only when red/yellow bbox area reaches this frame-area ratio.
TRAFFIC_LIGHT_STOP_AREA_RATIO = 0.05

# YOLO lane model. Use a lane-trained Ultralytics .pt file, not a generic COCO model.
YOLO_MODEL_PATH = BASE_DIR / "models" / "yolo" / "final_openvino_model"
# YOLO_MODEL_PATH = BASE_DIR / "models" / "yolo" / "final.pt"
YOLO_DEVICE = "cpu"
YOLO_IMAGE_SIZE = 640
YOLO_CONFIDENCE = 0.25
YOLO_IOU = 0.45
YOLO_CLASS_NAMES = ("car", "obstacle", "lane1", "lane2", "traffic_light")
YOLO_FRAME_SKIP = 1
YOLO_MIN_POINTS_PER_LANE = 3
YOLO_MIN_VALID_Y_SPAN_RATIO = 0.08
YOLO_MASK_SAMPLE_STEP_PX = 6
YOLO_SKELETON_ENABLED = True
YOLO_SKELETON_BRIDGE_GAP_PX = 17
YOLO_DASHED_MERGE_MAX_X_GAP_RATIO = 0.12
YOLO_CURVE_FIT_DEGREE = 2
YOLO_SOLIDIFY_STEP_PX = 6
YOLO_FIT_OUTLIER_REJECTION_PX = 35.0
YOLO_SEGMENTATION_MODE = "lane_area"  # "lane_area" follows lane1/lane2 area centers. "drivable_area" follows one road mask.
YOLO_DRIVABLE_MIN_ROW_WIDTH_RATIO = 0.05
YOLO_DRIVABLE_EDGE_PERCENTILE = 2.0
YOLO_TARGET_LANE_PAIR = "right"  # "right"=right lane, "left"=left lane, "closest"=nearest pair.
YOLO_TARGET_PATH_MODE = "closest_line"  # "closest_line" follows one line. "lane_center" follows the lane pair center.
YOLO_LANE_PAIR_SELECT_Y_RATIO = 0.78
YOLO_LANE_PAIR_TARGET_OFFSET_RATIO = -0.03
YOLO_DISPLAY_BIRD_EYE_VIEW = True

# Camera/IPM preprocessing. Bird-eye source points are image ratios in this order:
# bottom-left, bottom-right, top-right, top-left.
ROI_ENABLED = False
ROI_TOP_RATIO = 0.0
ROI_BOTTOM_RATIO = 1.0
ROI_LEFT_RATIO = 0.0
ROI_RIGHT_RATIO = 1.0
BIRD_EYE_ENABLED = False
BIRD_EYE_SRC_BOTTOM_LEFT = (0.10, 0.98)
BIRD_EYE_SRC_BOTTOM_RIGHT = (0.90, 0.98)
BIRD_EYE_SRC_TOP_RIGHT = (0.62, 0.35)
BIRD_EYE_SRC_TOP_LEFT = (0.38, 0.35)
BIRD_EYE_DST_MARGIN_RATIO = 0.18
BIRD_EYE_MASK_SOURCE_POLYGON = True

# Shared lane geometry
DEFAULT_LANE_WIDTH_RATIO = 0.50
MIN_LANE_WIDTH_RATIO = 0.35
MAX_LANE_WIDTH_RATIO = 0.65

# Driving controller. Positive steer means right.
CONTROL_MODE = "contest"  # "contest" uses angle/position weights.
MIN_SPEED = 230
MAX_SPEED = 255
MIN_LANE_CONFIDENCE_TO_DRIVE = 0.45
CONTEST_ANGLE_WEIGHT = 0.45
CONTEST_POSITION_WEIGHT = 0.10
CONTEST_STEERING_ANGLE_NORM_DEG = 50.0
CONTEST_STEER_LIMIT = 10.0

# Common steering limits & safety
MAX_STEER = 80
REVERSE_STEER = False
MAX_HOLD_FRAMES = 6
HOLD_DECEL_STEP = 6

# Manual mode (--mode manual): hold-to-drive, KartRider style (pynput).
MANUAL_DRIVE_POWER = 150    # W/S = forward/back, fixed 150 PWM WHILE HELD
MANUAL_STEER_POWER = 100    # A/D = steer, fixed +/-100 WHILE HELD
# Legacy tap values (only the space->manual fallback inside the auto loop).
MANUAL_SPEED_STEP = 10
MANUAL_STEER_HOLD_MS = 180

# Frame capture (--capture-frame; in manual mode press 'c' to start/stop).
CAPTURE_DIR = BASE_DIR / "captures"
CAPTURE_FPS = 30

# Display/logging
DISPLAY_ENABLED = True
SHOW_MASK_WINDOW = False
LOG_ENABLED = False
LOG_DIR = BASE_DIR / "Log"
SAVE_EVERY_N_FRAMES = 5
PRINT_EVERY_N_FRAMES = 15

START_KEYS = (ord("t"), ord("T"))


def parse_ratio_point(value: str) -> tuple[float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected x,y ratio pair, for example 0.38,0.35")
    try:
        x_ratio = float(parts[0])
        y_ratio = float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ratio point values must be numbers") from exc
    if not 0.0 <= x_ratio <= 1.0 or not 0.0 <= y_ratio <= 1.0:
        raise argparse.ArgumentTypeError("ratio point values must be between 0.0 and 1.0")
    return x_ratio, y_ratio


def is_start_key(key: int) -> bool:
    return key in START_KEYS


def _free_manual_command(free_input) -> DriveCommand:
    """Hold-to-drive command from held W/S/A/D keys, for the --free-manual
    waiting phase (before the mode is started with 't')."""
    speed = (
        MANUAL_DRIVE_POWER if free_input.held("w")
        else -MANUAL_DRIVE_POWER if free_input.held("s")
        else 0
    )
    steer = (
        MANUAL_STEER_POWER if free_input.held("d")
        else -MANUAL_STEER_POWER if free_input.held("a")
        else 0
    )
    return DriveCommand(steer=steer, speed=speed, reason="free_manual")


class ManualInput:
    """System-wide keyboard state for hold-to-drive manual mode (pynput).

    Listens globally so keys register even without the OpenCV window focused;
    'q' quits and 'c' toggles capture from anywhere. Only used in --mode manual.
    """

    def __init__(self) -> None:
        self._held = set()
        self._quit = False
        self._toggle_capture = False
        self._lock = threading.Lock()
        self._listener = None

    def start(self) -> None:
        from pynput import keyboard

        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._listener.daemon = True
        self._listener.start()

    @staticmethod
    def _char(key):
        try:
            return key.char.lower()
        except AttributeError:
            return None

    def _on_press(self, key) -> None:
        c = self._char(key)
        if c is None:
            return
        with self._lock:
            if c in ("w", "s", "a", "d"):
                self._held.add(c)
            elif c == "q":
                self._quit = True
            elif c == "c":
                self._toggle_capture = True

    def _on_release(self, key) -> None:
        c = self._char(key)
        if c is None:
            return
        with self._lock:
            self._held.discard(c)

    def held(self, c: str) -> bool:
        with self._lock:
            return c in self._held

    def quit_requested(self) -> bool:
        with self._lock:
            return self._quit

    def take_capture_toggle(self) -> bool:
        with self._lock:
            v = self._toggle_capture
            self._toggle_capture = False
            return v

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()


def read_console_key() -> int:
    try:
        import msvcrt
    except ImportError:
        return -1

    if not msvcrt.kbhit():
        return -1
    key = msvcrt.getwch()
    if key in ("\x00", "\xe0"):
        msvcrt.getwch()
        return -1
    return ord(key)


def _format_obstacle_debug(status: dict) -> str:
    """One-line dump of every gate the obstacle trigger depends on."""

    def flag(value) -> str:
        return "T" if value else "F"

    def num(value, fmt: str) -> str:
        return "NA" if value is None else format(value, fmt)

    sector = status.get("lidar_sector_ros", (None, None))
    if status.get("use_lidar"):
        proximity = (
            f"lidar={num(status.get('lidar_mm'), '.0f')}mm near={flag(status.get('lidar_near'))} "
            f"sect_ros=({sector[0]},{sector[1]}) thr={num(status.get('lidar_dist_thresh_mm'), '.0f')}"
        )
    else:
        proximity = (
            f"far_y={num(status.get('car_far_y_ratio'), '.3f')}"
            f">={num(status.get('camera_trigger_far_y_ratio'), '.3f')}"
        )
    return (
        f"phase={status.get('phase')} active={flag(status.get('active'))} "
        f"mode={'lidar' if status.get('use_lidar') else 'cam'} "
        f"lane={status.get('current_lane')} "
        f"cam={flag(status.get('camera_car'))}(conf={num(status.get('car_conf'), '.2f')}) "
        f"{proximity} "
        f"cnt={status.get('trigger_counter')}/{status.get('trigger_frames')} "
        f"present={flag(status.get('present'))} alert={flag(status.get('alert'))} "
        f"WHY={status.get('reason')} "
        f"src={status.get('source_lane')} shift_steer={status.get('shift_steer')} "
        f"head={num(status.get('heading_deg'), '.1f')} off={num(status.get('offset_norm'), '.2f')} "
        f"raw={status.get('raw_lane_label', '')!r}"
    )


def apply_obstacle_mission_override(
    base_command: DriveCommand,
    mission_command: DriveCommand | None,
    safety: SafetyDecision,
) -> DriveCommand:
    if mission_command is None:
        return base_command
    if safety.reason == "traffic_light_stop":
        return base_command
    return mission_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CTRL_ZERO camera/LiDAR/Arduino autonomous driving runtime.")
    parser.add_argument("--mode", choices=("vision", "manual", "auto", "parking_sim", "parking_logic"), default=RUN_MODE)
    parser.add_argument("--backend", choices=("yolo",), default=LANE_BACKEND)
    parser.add_argument("--yolo-model", type=Path, default=YOLO_MODEL_PATH)
    parser.add_argument("--yolo-frame-skip", type=int, default=YOLO_FRAME_SKIP)
    parser.add_argument("--yolo-imgsz", type=int, default=YOLO_IMAGE_SIZE)
    parser.add_argument("--yolo-conf", type=float, default=YOLO_CONFIDENCE)
    parser.add_argument("--yolo-iou", type=float, default=YOLO_IOU)
    parser.add_argument("--yolo-classes", default=",".join(YOLO_CLASS_NAMES))
    parser.add_argument("--yolo-min-points", type=int, default=YOLO_MIN_POINTS_PER_LANE)
    parser.add_argument("--yolo-min-y-span", type=float, default=YOLO_MIN_VALID_Y_SPAN_RATIO)
    parser.add_argument("--yolo-mask-step", type=int, default=YOLO_MASK_SAMPLE_STEP_PX)
    parser.add_argument("--yolo-skeleton", action=argparse.BooleanOptionalAction, default=YOLO_SKELETON_ENABLED)
    parser.add_argument("--yolo-bridge-gap", type=int, default=YOLO_SKELETON_BRIDGE_GAP_PX)
    parser.add_argument("--yolo-merge-gap-ratio", type=float, default=YOLO_DASHED_MERGE_MAX_X_GAP_RATIO)
    parser.add_argument("--yolo-curve-degree", type=int, default=YOLO_CURVE_FIT_DEGREE)
    parser.add_argument("--yolo-solid-step", type=int, default=YOLO_SOLIDIFY_STEP_PX)
    parser.add_argument("--yolo-fit-outlier-px", type=float, default=YOLO_FIT_OUTLIER_REJECTION_PX)
    parser.add_argument("--yolo-segmentation-mode", choices=("lane_area", "drivable_area", "lane_lines"), default=YOLO_SEGMENTATION_MODE)
    parser.add_argument("--yolo-drivable-min-row-width", type=float, default=YOLO_DRIVABLE_MIN_ROW_WIDTH_RATIO)
    parser.add_argument("--yolo-drivable-edge-percentile", type=float, default=YOLO_DRIVABLE_EDGE_PERCENTILE)
    parser.add_argument("--yolo-target-lane-pair", choices=("right", "left", "closest", "center", "split"), default=YOLO_TARGET_LANE_PAIR)
    parser.add_argument("--yolo-target-path", choices=("closest_line", "left_line", "right_line", "lane_center"), default=YOLO_TARGET_PATH_MODE)
    parser.add_argument("--yolo-pair-select-y", type=float, default=YOLO_LANE_PAIR_SELECT_Y_RATIO)
    parser.add_argument("--yolo-pair-target-offset", type=float, default=YOLO_LANE_PAIR_TARGET_OFFSET_RATIO)
    parser.add_argument("--yolo-display-bird-eye", action=argparse.BooleanOptionalAction, default=YOLO_DISPLAY_BIRD_EYE_VIEW)
    parser.add_argument("--roi", action=argparse.BooleanOptionalAction, default=ROI_ENABLED)
    parser.add_argument("--roi-top", type=float, default=ROI_TOP_RATIO)
    parser.add_argument("--roi-bottom", type=float, default=ROI_BOTTOM_RATIO)
    parser.add_argument("--roi-left", type=float, default=ROI_LEFT_RATIO)
    parser.add_argument("--roi-right", type=float, default=ROI_RIGHT_RATIO)
    parser.add_argument("--bird-eye", action=argparse.BooleanOptionalAction, default=BIRD_EYE_ENABLED)
    parser.add_argument("--bird-eye-src-bottom-left", type=parse_ratio_point, default=BIRD_EYE_SRC_BOTTOM_LEFT)
    parser.add_argument("--bird-eye-src-bottom-right", type=parse_ratio_point, default=BIRD_EYE_SRC_BOTTOM_RIGHT)
    parser.add_argument("--bird-eye-src-top-right", type=parse_ratio_point, default=BIRD_EYE_SRC_TOP_RIGHT)
    parser.add_argument("--bird-eye-src-top-left", type=parse_ratio_point, default=BIRD_EYE_SRC_TOP_LEFT)
    parser.add_argument("--bird-eye-dst-margin", type=float, default=BIRD_EYE_DST_MARGIN_RATIO)
    parser.add_argument("--bird-eye-mask-source-polygon", action=argparse.BooleanOptionalAction, default=BIRD_EYE_MASK_SOURCE_POLYGON)
    parser.add_argument("--camera-index", type=int, default=CAMERA_INDEX)
    parser.add_argument("--camera-backend", choices=("dshow", "msmf", "any"), default=CAMERA_BACKEND)
    parser.add_argument("--camera-width", type=int, default=CAMERA_WIDTH)
    parser.add_argument("--camera-height", type=int, default=CAMERA_HEIGHT)
    parser.add_argument("--camera-fps", type=int, default=CAMERA_FPS)
    parser.add_argument("--arduino-port", default=ARDUINO_PORT)
    parser.add_argument("--no-arduino", action="store_true")
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--capture-frame", action="store_true",
                        help="manual mode: press 'c' to start/stop 30 fps frame capture to captures/")
    parser.add_argument("--free-manual", action="store_true",
                        help="any mode: hold-to-drive (W/S/A/D) until you press 't' to start the mode")
    parser.add_argument("--rear-camera-index", type=int, default=PARKING_LOGIC_REAR_CAMERA_INDEX,
                        help="parking_logic: rear camera index (default from PARKING_LOGIC_REAR_CAMERA_INDEX)")
    parser.add_argument("--best-model", action="store_true",
                        help="parking_logic: use the slot model's bbox to build an infinite-lane "
                             "frame for the lane model (keypoints are not used)")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--no-log", action="store_true")
    return parser


def build_preprocessor(args) -> LanePreprocessor:
    return LanePreprocessor(
        roi=ROICropConfig(
            enabled=args.roi,
            top_ratio=args.roi_top,
            bottom_ratio=args.roi_bottom,
            left_ratio=args.roi_left,
            right_ratio=args.roi_right,
        ),
        bird_eye=BirdEyeConfig(
            enabled=args.bird_eye,
            src_bottom_left=args.bird_eye_src_bottom_left,
            src_bottom_right=args.bird_eye_src_bottom_right,
            src_top_right=args.bird_eye_src_top_right,
            src_top_left=args.bird_eye_src_top_left,
            dst_margin_ratio=args.bird_eye_dst_margin,
            mask_source_polygon=args.bird_eye_mask_source_polygon,
        ),
    )


def build_lane_detector(args):
    from ctrl_zero.vision.yolo_lane import YOLOLaneConfig, YOLOLaneDetector
    from ctrl_zero.vision.cache import FrameSkippingLaneDetector

    yolo_classes = tuple(item.strip() for item in args.yolo_classes.split(",") if item.strip())
    detector = YOLOLaneDetector(
        YOLOLaneConfig(
            model_path=args.yolo_model,
            device=YOLO_DEVICE,
            image_size=args.yolo_imgsz,
            confidence=args.yolo_conf,
            iou=args.yolo_iou,
            class_names=yolo_classes,
            min_points_per_lane=args.yolo_min_points,
            min_valid_y_span_ratio=args.yolo_min_y_span,
            mask_sample_step_px=args.yolo_mask_step,
            skeleton_enabled=args.yolo_skeleton,
            skeleton_bridge_gap_px=args.yolo_bridge_gap,
            dashed_merge_max_x_gap_ratio=args.yolo_merge_gap_ratio,
            curve_fit_degree=args.yolo_curve_degree,
            solidify_step_px=args.yolo_solid_step,
            fit_outlier_rejection_px=args.yolo_fit_outlier_px,
            segmentation_mode=args.yolo_segmentation_mode,
            drivable_min_row_width_ratio=args.yolo_drivable_min_row_width,
            drivable_edge_percentile=args.yolo_drivable_edge_percentile,
            target_lane_pair=args.yolo_target_lane_pair,
            target_path_mode=args.yolo_target_path,
            lane_pair_select_y_ratio=args.yolo_pair_select_y,
            lane_pair_target_offset_ratio=args.yolo_pair_target_offset,
            default_lane_width_ratio=DEFAULT_LANE_WIDTH_RATIO,
            min_lane_width_ratio=MIN_LANE_WIDTH_RATIO,
            max_lane_width_ratio=MAX_LANE_WIDTH_RATIO,
            display_bird_eye_view=args.yolo_display_bird_eye,
            preprocessor=build_preprocessor(args),
        )
    )
    return FrameSkippingLaneDetector(detector, skip=args.yolo_frame_skip) if args.yolo_frame_skip > 1 else detector


def main() -> None:
    args = build_parser().parse_args()
    if args.list_ports:
        print(format_serial_ports())
        return

    display_enabled = DISPLAY_ENABLED and not args.no_display
    motor_enabled = USE_ARDUINO and not args.no_arduino and args.mode != "vision"
    arduino_port = args.arduino_port
    if motor_enabled and arduino_port in (None, "", "auto"):
        arduino_port = "auto"

    # parking_sim is LiDAR-driven and skips YOLO.  It still opens the camera when
    # the heading guide is on (to steer straight off the left-half ROI); with the
    # guide off it is fully camera-less.
    parking_sim = args.mode == "parking_sim"
    parking_logic = args.mode == "parking_logic"
    parking_sim_guide = parking_sim and PARKING_SIM_USE_CAMERA_GUIDE
    # ``camera`` is the FRONT camera everywhere (parking_logic uses it for the
    # SEARCH vanishing-point guide); parking_logic additionally opens the REAR
    # camera for the line-follow reverse.
    camera = None if (parking_sim and not parking_sim_guide) else CameraReader(
        CameraConfig(
            index=args.camera_index,
            backend=args.camera_backend,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
        )
    )
    rear_camera = CameraReader(
        CameraConfig(
            index=args.rear_camera_index,
            backend=args.camera_backend,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
        )
    ) if parking_logic else None
    motor = ArduinoMotorController(
        ArduinoConfig(port=arduino_port, baudrate=ARDUINO_BAUDRATE, drive_max_pwm=DRIVE_MAX_PWM),
        dry_run=not motor_enabled,
    )
    controller = DriveController(
        DriveConfig(
            control_mode=CONTROL_MODE,
            min_speed=MIN_SPEED,
            max_speed=MAX_SPEED,
            contest_angle_weight=CONTEST_ANGLE_WEIGHT,
            contest_position_weight=CONTEST_POSITION_WEIGHT,
            contest_angle_norm_deg=CONTEST_STEERING_ANGLE_NORM_DEG,
            contest_steer_limit=CONTEST_STEER_LIMIT,
            min_confidence=MIN_LANE_CONFIDENCE_TO_DRIVE,
            max_steer=MAX_STEER,
            reverse_steer=REVERSE_STEER,
            max_hold_frames=MAX_HOLD_FRAMES,
            hold_decel_step=HOLD_DECEL_STEP,
        )
    )

    # parking_sim's guide steers on the vanishing point of the two fitted lines —
    # no model inference — so parking_sim never needs YOLO loaded.
    lane_detector = (
        None if (parking_sim or args.mode == "manual") else build_lane_detector(args)
    )
    lidar_config = LidarConfig(
        port=LIDAR_PORT,
        front_min_ros_deg=OBSTACLE_LIDAR_MIN_ROS_DEG,
        front_max_ros_deg=OBSTACLE_LIDAR_MAX_ROS_DEG,
        raw_angle_for_ros_zero_deg=LIDAR_RAW_ANGLE_FOR_ROS_ZERO_DEG,
        stop_distance_mm=LIDAR_STOP_DISTANCE_MM,
        slow_distance_mm=LIDAR_SLOW_DISTANCE_MM,
        min_speed_scale=LIDAR_MIN_SPEED_SCALE,
    )
    lidar = LidarReader(lidar_config) if USE_LIDAR else None
    last_obstacle = ObstacleDecision.clear() if USE_LIDAR else None
    last_lidar_scan = None
    obstacle_mission = ContestObstacleMission(
        ContestObstacleMissionConfig(
            enabled=OBSTACLE_MISSION_ENABLED,
            use_lidar=USE_LIDAR,
            camera_trigger_far_y_ratio=OBSTACLE_CAMERA_TRIGGER_FAR_Y_RATIO,
            raw_angle_for_ros_zero_deg=LIDAR_RAW_ANGLE_FOR_ROS_ZERO_DEG,
            lidar_obstacle_distance_mm=OBSTACLE_DISTANCE_MM,
            trigger_frames=OBSTACLE_TRIGGER_FRAMES,
            cruise_speed=MAX_SPEED,
            lane2_shift_steer=OBSTACLE_LANE2_SHIFT_STEER,
            lane1_shift_steer=OBSTACLE_LANE1_SHIFT_STEER,
            shift_out_duration_s=OBSTACLE_SHIFT_OUT_SECONDS,
            shift_out_steer_weight=OBSTACLE_SHIFT_OUT_STEER_WEIGHT,
            retrigger_cooldown_s=OBSTACLE_RETRIGGER_COOLDOWN_SECONDS,
            lidar_min_ros_deg=OBSTACLE_LIDAR_MIN_ROS_DEG,
            lidar_max_ros_deg=OBSTACLE_LIDAR_MAX_ROS_DEG,
        )
    )
    parking_sim_mission = build_timed_parking_mission()
    parking_logic_mission = build_parking_logic_mission()
    logger = DriveLogger(
        LogConfig(
            enabled=(LOG_ENABLED or args.log) and not args.no_log,
            directory=LOG_DIR,
            save_every_n_frames=SAVE_EVERY_N_FRAMES,
        )
    )

    manual_steer = 0
    manual_steer_until = 0.0
    manual_speed = 0
    frame_count = 0
    fps = 0.0
    last_time = time.perf_counter()
    drive_started = not motor_enabled
    prev_obstacle_detected = False

    # --free-manual: hold-to-drive (W/S/A/D) while waiting, until 't' starts the
    # mode. (manual mode is already hold-to-drive, so it needs no extra listener.)
    free_input = None
    if args.free_manual and args.mode != "manual":
        free_input = ManualInput()
        free_input.start()

    try:
        motor.open()
        if camera is not None:
            camera.open()
        if rear_camera is not None:
            rear_camera.open()
        if lidar is not None:
            lidar.open()
        logger.open()

        print("Keys: q quit, t start/resume, space stop + manual, +/- max speed, l toggle log.")
        print("Manual driving (space or --mode manual): w/s speed, a/d steer pulse, c center steer.")
        if free_input is not None:
            print("--free-manual: hold W/S/A/D to drive now; press 't' to start the mode.")
        print(f"Runtime: mode={args.mode}, backend={args.backend}, motor={'on' if motor_enabled else 'dry'}")
        if args.mode == "parking_sim":
            print(
                "Parking_sim mode: timed two-arc perpendicular parking "
                f"(all start markers: approach {PARKING_SIM_APPROACH_S}s, "
                f"swing {PARKING_SIM_SWING_S}s, arc {PARKING_SIM_ARC_S}s, "
                f"parked pause {PARKING_SIM_PARKED_PAUSE_S}s, "
                f"exit approach {PARKING_SIM_EXIT_APPROACH_S}s, "
                f"exit arc {PARKING_SIM_EXIT_ARC_S}s)."
            )
            if lidar is None:
                print("WARNING: USE_LIDAR is False; parking_sim has no sensor input and cannot work.")
        if motor_enabled:
            print("Drive is waiting. Press t/T to start motor output.")

        if parking_sim:
            # LiDAR-driven parking.  During the first straight (SEARCH/APPROACH)
            # an optional camera guide steers: the left ROI is kept (rest blacked),
            # horizontal white paint is ERASED, broken dashes are extended, and the
            # UNCHANGED YOLO lane model runs on that frame; its driving-area
            # heading/offset produce the steer.  Other phases use mission steering.
            guide_line_cfg = build_parking_guide_line_config()
            while True:
                frame_count += 1
                frame = None
                if camera is not None:
                    ret, frame = camera.read()
                    if not ret:
                        frame = None
                if lidar is not None and frame_count % max(LIDAR_POLL_EVERY_N_FRAMES, 1) == 0:
                    try:
                        last_lidar_scan = lidar.read_scan()
                    except Exception as exc:  # pragma: no cover - hardware dependent
                        print(f"LiDAR read failed: {exc}")
                        last_lidar_scan = None
                if drive_started:
                    parking_command = parking_sim_mission.step(last_lidar_scan)
                    command = parking_command or DriveCommand(steer=0, speed=0, reason="parking_sim_idle")
                elif free_input is not None:
                    command = _free_manual_command(free_input)
                else:
                    command = DriveCommand(steer=0, speed=0, reason="waiting_for_start")

                # Guide pipeline runs every frame for inspection (prep + model);
                # its steer is APPLIED only while driving the initial straight.
                guide_vis = frame
                guide_active = (
                    drive_started and parking_sim_mission.in_initial_straight
                )
                guide_steer = None
                vp_err = None
                if frame is not None:
                    model_frame, guide_vis, n_dashes, guide_lines, _hy = prepare_parking_line_frame(
                        frame, guide_line_cfg
                    )
                    # Vanishing-point hold: parallel to the two lines <=> their
                    # image intersection sits at the frame centre x.  Position-
                    # independent, so no reference calibration is needed.
                    vp = vanishing_point(guide_lines)
                    cx = frame.shape[1] / 2.0
                    if vp is not None:
                        vp_err = (vp[0] - cx) / cx
                        guide_steer = int(
                            clamp(PARKING_GUIDE_VP_GAIN * vp_err,
                                  -PARKING_GUIDE_STEER_LIMIT, PARKING_GUIDE_STEER_LIMIT)
                        )
                        cv2.line(guide_vis, (int(cx), 0), (int(cx), 40), (200, 200, 200), 1)
                        vx = int(clamp(vp[0], 0, frame.shape[1] - 1))
                        cv2.circle(guide_vis, (vx, max(0, min(int(vp[1]), 40))), 6, (0, 255, 255), -1)
                    if guide_active and guide_steer is not None and command.speed != 0:
                        command = DriveCommand(
                            steer=guide_steer, speed=command.speed,
                            reason=command.reason + "+guide",
                        )
                    if display_enabled:
                        cv2.imshow("parking_guide model input", model_frame)

                motor.send(command.steer, command.speed if motor_enabled else 0)
                if frame_count % PRINT_EVERY_N_FRAMES == 0:
                    print(
                        f"frame={frame_count} steer={command.steer:+d} "
                        f"speed={command.speed:+d} reason={command.reason}",
                        flush=True,
                    )

                key = read_console_key()
                if display_enabled and guide_vis is not None:
                    applied = "+guide" in command.reason
                    hud = (
                        f"phase={parking_sim_mission.state.name}  "
                        f"vp_err={'--' if vp_err is None else f'{vp_err:+.3f}'}  "
                        f"guide={'--' if guide_steer is None else f'{guide_steer:+d}'}  "
                        f"applied={command.steer:+d} spd={command.speed:+d}  "
                        f"{'GUIDE->STEER' if applied else ('guide idle' if guide_active else 'guide off')}"
                    )
                    y = guide_vis.shape[0] - 44
                    cv2.putText(guide_vis, hud, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(guide_vis, hud, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0) if applied else (255, 255, 0), 1, cv2.LINE_AA)
                    cv2.imshow("parking_guide", guide_vis)
                    k2 = cv2.waitKey(1) & 0xFF
                    if key == -1:
                        key = k2
                if key == ord("q"):
                    break
                if key == ord(" "):
                    drive_started = False
                    motor.stop()
                    print("Stopped. Press t/T to restart.")
                elif motor_enabled and not drive_started and is_start_key(key):
                    drive_started = True
                    print("Drive started.")
                if camera is None:
                    time.sleep(0.02)
            return

        if parking_logic:
            # Rear-camera parking: FSM drives the timed phases; during LINE_FOLLOW
            # the broken parking-lines are extended and fed to the (unchanged)
            # YOLO lane model, whose heading/offset then steers the reverse.
            line_cfg = build_parking_line_config()
            guide_line_cfg = build_parking_guide_line_config()
            slot_detector = None
            if args.best_model:
                print("--best-model: slot bbox -> infinite-lane frame -> lane model "
                      f"({PARKING_LOGIC_BEST_MODEL_PATH})")
                slot_detector = ParkingSlotDetector(
                    PARKING_LOGIC_BEST_MODEL_PATH,
                    device=YOLO_DEVICE,
                    image_size=args.yolo_imgsz,
                    confidence=PARKING_LOGIC_BEST_CONF,
                    iou=args.yolo_iou,
                )
            print(
                "Parking_logic: first car -> back "
                f"{PARKING_LOGIC_BACK_S}s -> full-left {PARKING_LOGIC_TURN_S}s -> "
                "rear-cam line-follow reverse."
            )
            while True:
                frame_count += 1
                front_frame = None
                rear_frame = None
                if camera is not None:
                    ret, front_frame = camera.read()
                    if not ret:
                        front_frame = None
                if rear_camera is not None:
                    ret_r, rear_frame = rear_camera.read()
                    if not ret_r:
                        rear_frame = None
                if lidar is not None and frame_count % max(LIDAR_POLL_EVERY_N_FRAMES, 1) == 0:
                    try:
                        last_lidar_scan = lidar.read_scan()
                    except Exception as exc:  # pragma: no cover - hardware dependent
                        print(f"LiDAR read failed: {exc}")
                        last_lidar_scan = None

                if drive_started:
                    parking_command = parking_logic_mission.step(last_lidar_scan)
                    command = parking_command or DriveCommand(steer=0, speed=0, reason="parking_logic_idle")
                elif free_input is not None:
                    command = _free_manual_command(free_input)
                else:
                    command = DriveCommand(steer=0, speed=0, reason="waiting_for_start")

                # SEARCH: FRONT camera vanishing-point guide holds the heading
                # straight (same invariant as parking_sim; no model inference).
                if front_frame is not None and (
                    parking_logic_mission.in_initial_straight or not drive_started
                ):
                    _gm, guide_vis, _gd, guide_lines, _hy = prepare_parking_line_frame(
                        front_frame, guide_line_cfg
                    )
                    vp = vanishing_point(guide_lines)
                    cx = front_frame.shape[1] / 2.0
                    guide_steer = None
                    vp_err = None
                    if vp is not None:
                        vp_err = (vp[0] - cx) / cx
                        guide_steer = int(
                            clamp(PARKING_LOGIC_GUIDE_VP_GAIN * vp_err,
                                  -PARKING_LOGIC_GUIDE_STEER_LIMIT,
                                  PARKING_LOGIC_GUIDE_STEER_LIMIT)
                        )
                        cv2.line(guide_vis, (int(cx), 0), (int(cx), 40), (200, 200, 200), 1)
                        vx = int(clamp(vp[0], 0, front_frame.shape[1] - 1))
                        cv2.circle(guide_vis, (vx, max(0, min(int(vp[1]), 40))), 6, (0, 255, 255), -1)
                    if (
                        drive_started
                        and parking_logic_mission.in_initial_straight
                        and guide_steer is not None
                        and command.speed != 0
                    ):
                        command = DriveCommand(
                            steer=guide_steer, speed=command.speed,
                            reason=command.reason + "+guide",
                        )
                    if display_enabled:
                        applied = "+guide" in command.reason
                        hud = (
                            f"phase={parking_logic_mission.state.name}  "
                            f"vp_err={'--' if vp_err is None else f'{vp_err:+.3f}'}  "
                            f"guide={'--' if guide_steer is None else f'{guide_steer:+d}'}  "
                            f"applied={command.steer:+d} spd={command.speed:+d}  "
                            f"{'GUIDE->STEER' if applied else 'guide idle'}"
                        )
                        y = guide_vis.shape[0] - 12
                        cv2.putText(guide_vis, hud, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(guide_vis, hud, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (0, 255, 0) if applied else (255, 255, 0), 1, cv2.LINE_AA)
                        cv2.imshow("parking_logic guide", guide_vis)

                # LINE_FOLLOW: REAR camera bay lines -> model -> reverse steering.
                # Skipped during LINE_FOLLOW's open-loop opening (plain straight
                # reverse) — no slot model, no lane model, no Hough.
                line_vis = rear_frame
                if (
                    rear_frame is not None
                    and drive_started
                    and parking_logic_mission.line_follow_camera_active
                ):
                    # The slot model supplies the bbox (where to look); the bay's
                    # side lines are repainted spanning the whole frame — a lane
                    # that never ends — so the fit stays well conditioned even
                    # when the bay-end line covers the drivable area.  Without
                    # --best-model the same preparation runs over the full frame.
                    bbox = None
                    if slot_detector is not None:
                        det = best_slot(slot_detector.detect(rear_frame), rear_frame.shape[1])
                        bbox = det.bbox if det is not None else None
                    model_frame, line_vis, n_dashes, h_far_y = prepare_infinite_lane_frame(
                        rear_frame, bbox, line_cfg
                    )
                    if bbox is not None:
                        bx1, by1, bx2, by2 = (int(v) for v in bbox)
                        cv2.rectangle(line_vis, (bx1, by1), (bx2, by2), (255, 200, 0), 2)
                    lane = lane_detector.detect(model_frame)
                    heading = lane.heading_deg if lane.heading_deg is not None else 0.0
                    offset = lane.offset_norm if lane.offset_norm is not None else 0.0
                    line_conf = lane.confidence
                    line_ok = n_dashes > 0 and lane.confidence > 0.0
                    raw_steer = PARKING_LOGIC_REVERSE_STEER_SIGN * (
                        PARKING_LOGIC_HEADING_GAIN * heading
                        + PARKING_LOGIC_OFFSET_GAIN * offset
                    )
                    line_steer = int(clamp(raw_steer, -PARKING_LOGIC_STEER_LIMIT, PARKING_LOGIC_STEER_LIMIT))

                    # Bay-end trigger: the white horizontal line has risen past the
                    # far_y threshold -> end camera steering before the lines leave
                    # the view and the model heading turns to noise.  The mission
                    # then backs straight for final_back_s, pauses, and exits.
                    bay_end = (
                        h_far_y is not None and h_far_y >= PARKING_LOGIC_STOP_FAR_Y_RATIO
                    )
                    if bay_end and parking_logic_mission.stop_line_follow():
                        print(f"[parking_logic] bay-end line far_y={h_far_y:.2f} "
                              f"-> final back {PARKING_LOGIC_FINAL_BACK_S}s", flush=True)
                    elif (
                        drive_started
                        and parking_logic_mission.in_line_follow
                        and line_ok
                        and command.speed != 0
                    ):
                        command = DriveCommand(
                            steer=line_steer, speed=command.speed,
                            reason=command.reason + "+line",
                        )
                    if display_enabled and line_vis is not None:
                        applied = "+line" in command.reason
                        hud = (
                            f"phase={parking_logic_mission.state.name}  "
                            f"lanes={n_dashes} "
                            f"conf={line_conf:.2f} head={heading:+.1f} off={offset:+.2f} "
                            f"hFarY={'--' if h_far_y is None else f'{h_far_y:.2f}'}/{PARKING_LOGIC_STOP_FAR_Y_RATIO:.2f} "
                            f"line_steer={line_steer:+d} | applied={command.steer:+d} spd={command.speed:+d} "
                            f"{'LINE->STEER' if applied else 'line idle'}"
                        )
                        y = line_vis.shape[0] - 12
                        cv2.putText(line_vis, hud, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(line_vis, hud, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (0, 255, 0) if applied else (255, 255, 0), 1, cv2.LINE_AA)
                        cv2.imshow("parking_logic", line_vis)
                        cv2.imshow("parking_logic model input", model_frame)
                        # What the LANE model made of the frame.
                        cv2.imshow("parking_logic lane result", lane.annotated)

                motor.send(command.steer, command.speed if motor_enabled else 0)
                if frame_count % PRINT_EVERY_N_FRAMES == 0:
                    print(
                        f"frame={frame_count} steer={command.steer:+d} "
                        f"speed={command.speed:+d} reason={command.reason}",
                        flush=True,
                    )

                key = read_console_key()
                if display_enabled:
                    k2 = cv2.waitKey(1) & 0xFF
                    if key == -1:
                        key = k2
                if key == ord("q"):
                    break
                if key == ord(" "):
                    drive_started = False
                    motor.stop()
                    print("Stopped. Press t/T to restart.")
                elif motor_enabled and not drive_started and is_start_key(key):
                    drive_started = True
                    print("Drive started.")
                if camera is None:
                    time.sleep(0.02)
            return

        if args.mode == "manual":
            # Hold-to-drive (KartRider style): W/S=fwd/back, A/D=steer, only WHILE
            # held (release = stop). No YOLO/LiDAR -> light enough for 30 fps capture.
            manual_input = ManualInput()
            manual_input.start()
            cap_active = False
            cap_count = 0
            cap_last_s = 0.0
            cap_dir = None
            print("MANUAL hold-to-drive: hold W/S = fwd/back @150, A/D = steer @100. q = quit.")
            if args.capture_frame:
                print("  --capture-frame ON: press 'c' to start/stop 30 fps capture -> captures/.")
            try:
                while True:
                    ret, frame = camera.read()
                    if not ret or frame is None:
                        print("Camera frame read failed.")
                        break
                    frame_count += 1
                    if manual_input.quit_requested():
                        break

                    speed = (
                        MANUAL_DRIVE_POWER if manual_input.held("w")
                        else -MANUAL_DRIVE_POWER if manual_input.held("s")
                        else 0
                    )
                    steer = (
                        MANUAL_STEER_POWER if manual_input.held("d")
                        else -MANUAL_STEER_POWER if manual_input.held("a")
                        else 0
                    )
                    motor.send(steer, speed if motor_enabled else 0)

                    if args.capture_frame and manual_input.take_capture_toggle():
                        cap_active = not cap_active
                        if cap_active and cap_dir is None:
                            cap_dir = CAPTURE_DIR / time.strftime("manual_%Y%m%d_%H%M%S")
                            cap_dir.mkdir(parents=True, exist_ok=True)
                        print(
                            f"Capture {'STARTED -> ' + str(cap_dir) if cap_active else 'stopped'}",
                            flush=True,
                        )
                    if cap_active and cap_dir is not None:
                        now = time.time()
                        if now - cap_last_s >= 1.0 / CAPTURE_FPS:
                            cap_last_s = now
                            cv2.imwrite(str(cap_dir / f"frame_{cap_count:06d}.jpg"), frame)
                            cap_count += 1

                    if display_enabled:
                        disp = frame.copy()
                        cv2.putText(
                            disp,
                            f"MANUAL steer={steer:+d} speed={speed:+d} cap={'ON' if cap_active else 'off'}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
                        )
                        cv2.imshow("CTRL_ZERO", disp)
                        if (cv2.waitKey(1) & 0xFF) == ord("q"):
                            break
            finally:
                manual_input.stop()
            return

        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                print("Camera frame read failed.")
                break

            frame_count += 1
            lane = lane_detector.detect(frame)

            if lidar is not None and frame_count % max(LIDAR_POLL_EVERY_N_FRAMES, 1) == 0:
                try:
                    last_lidar_scan = lidar.read_scan()
                    last_obstacle = analyze_obstacles(last_lidar_scan, lidar_config)
                except Exception as exc:  # pragma: no cover - hardware dependent
                    print(f"LiDAR read failed: {exc}")
                    last_lidar_scan = None
                    last_obstacle = ObstacleDecision.clear()

            if args.mode == "manual" and time.time() > manual_steer_until:
                manual_steer = 0

            lane_for_control = lane
            traffic_obj = traffic_light_object(lane_for_control.objects)
            frame_area = float(max(lane_for_control.annotated.shape[0] * lane_for_control.annotated.shape[1], 1))
            safety = build_safety_decision(
                lidar=last_obstacle,
                traffic_light_state=lane_for_control.traffic_light_state,
                traffic_light_object=traffic_obj,
                traffic_light_frame_area=frame_area,
                traffic_light_min_stop_area_ratio=TRAFFIC_LIGHT_STOP_AREA_RATIO,
            )
            if drive_started:
                if args.mode == "parking_sim":
                    # Parking runs standalone off LiDAR; the lane controller and
                    # obstacle mission are bypassed entirely.
                    parking_command = parking_sim_mission.step(last_lidar_scan)
                    command = parking_command or DriveCommand(steer=0, speed=0, reason="parking_sim_idle")
                else:
                    base_command = controller.compute(
                        lane_for_control,
                        safety,
                        args.mode,
                        manual_steer=manual_steer,
                        manual_speed=manual_speed,
                    )
                    mission_command = None
                    if args.mode == "auto":
                        mission_command = obstacle_mission.step(
                            objects=lane.objects,
                            current_lane=lane.lane_label or lane.lane_pair_label,
                            frame_width=lane.annotated.shape[1],
                            frame_height=lane.annotated.shape[0],
                            lidar_scan=last_lidar_scan,
                            heading_deg=lane.heading_deg,
                            offset_norm=lane.offset_norm,
                            lane_follow_steer=base_command.steer,
                            cruise_speed=controller.config.max_speed,
                        )
                    command = apply_obstacle_mission_override(base_command, mission_command, safety)
            elif free_input is not None:
                command = _free_manual_command(free_input)
            else:
                command = DriveCommand(steer=0, speed=0, reason="waiting_for_start")
            motor.send(command.steer, command.speed if motor_enabled else 0)

            # Obstacle-detection telemetry for the UI alert, the console, and the
            # log.  Only meaningful in auto mode where the mission runs.
            obstacle_detected = args.mode == "auto" and obstacle_mission.obstacle_alert
            obstacle_status = None
            if args.mode == "auto":
                obstacle_status = obstacle_mission.debug_status()
                obstacle_status["detected"] = obstacle_detected
                obstacle_status["raw_lane_label"] = lane.lane_label or lane.lane_pair_label
                debug_line = _format_obstacle_debug(obstacle_status)
                if obstacle_detected and not prev_obstacle_detected:
                    print(f"[OBSTACLE] DETECTED frame={frame_count} {debug_line}", flush=True)
                elif not obstacle_detected and prev_obstacle_detected:
                    print(f"[OBSTACLE] CLEARED  frame={frame_count} {debug_line}", flush=True)
                if OBSTACLE_LOG_EVERY_FRAME:
                    print(f"[OBS] f={frame_count} {debug_line}", flush=True)
            prev_obstacle_detected = obstacle_detected

            now = time.perf_counter()
            elapsed = now - last_time
            last_time = now
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if elapsed > 0 else fps

            if frame_count % PRINT_EVERY_N_FRAMES == 0:
                print(
                    f"frame={frame_count} fps={fps:.1f} conf={lane_for_control.confidence:.2f} "
                    f"steer={command.steer:+d} speed={command.speed:+d} reason={command.reason} "
                    f"obstacle={'!' if obstacle_detected else '-'}",
                    flush=True,
                )

            logger.log(frame, args.mode, args.backend, lane_for_control, safety, command, obstacle_status)

            key = -1
            if display_enabled:
                display = lane_for_control.annotated
                draw_status(display, lane_for_control, safety, command, args.mode, args.backend, fps, motor_enabled, obstacle_detected)
                cv2.imshow("CTRL_ZERO", display)
                if SHOW_MASK_WINDOW and lane_for_control.mask is not None:
                    cv2.imshow("CTRL_ZERO lane mask", lane_for_control.mask)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = read_console_key()

            if key == ord("q"):
                break
            started_this_frame = False
            if key == ord(" "):
                # Space: stop immediately and hand control to manual WASD driving.
                # Drive stays armed so w/s/a/d work at once (speed starts at 0).
                args.mode = "manual"
                manual_speed = 0
                manual_steer = 0
                manual_steer_until = 0.0
                motor.stop()
                print("Stopped. MANUAL mode: w/s speed, a/d steer, c center.")
            elif motor_enabled and not drive_started and is_start_key(key):
                drive_started = True
                started_this_frame = True
                print("Drive started.")
            elif key in (ord("+"), ord("=")):
                controller.config.max_speed = int(clamp(controller.config.max_speed + 5, 0, MAX_SPEED))
            elif key in (ord("-"), ord("_")):
                controller.config.max_speed = int(clamp(controller.config.max_speed - 5, MIN_SPEED, MAX_SPEED))
            elif key == ord("l"):
                logger.set_enabled(not logger.config.enabled)
                print(f"Logging {'enabled' if logger.config.enabled else 'disabled'}")

            if args.mode == "manual" and not started_this_frame:
                if key == ord("w"):
                    manual_speed = int(clamp(manual_speed + MANUAL_SPEED_STEP, -MAX_SPEED, MAX_SPEED))
                elif key == ord("s"):
                    manual_speed = int(clamp(manual_speed - MANUAL_SPEED_STEP, -MAX_SPEED, MAX_SPEED))
                elif key == ord("a"):
                    manual_steer = -int(clamp(MANUAL_STEER_POWER, 0, MAX_STEER))
                    manual_steer_until = time.time() + MANUAL_STEER_HOLD_MS / 1000.0
                elif key == ord("d"):
                    manual_steer = int(clamp(MANUAL_STEER_POWER, 0, MAX_STEER))
                    manual_steer_until = time.time() + MANUAL_STEER_HOLD_MS / 1000.0
                elif key == ord("c"):
                    manual_steer = 0
                    manual_steer_until = 0.0
    finally:
        if free_input is not None:
            free_input.stop()
        motor.close()
        if camera is not None:
            camera.release()
        if rear_camera is not None:
            rear_camera.release()
        if lidar is not None:
            lidar.close()
        logger.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
