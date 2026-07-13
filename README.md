# CTRL_ZERO

카메라, LiDAR, Arduino, DC 모터/조향 모터를 이용한 소형 자율주행 코드입니다. `main.py`가 카메라 입력, YOLO 차선/객체 인식, 선택적 LiDAR 장애물 판단, 안전 판단, 조향/속도 제어, Arduino 모터 출력을 조립합니다.

## 프로젝트 구조

```text
main.py                              # 실행 모드와 튜닝 파라미터
ctrl_zero/
  camera.py                          # OpenCV 카메라 입력
  arduino.py                         # Arduino serial 통신과 steer,speed 전송
  control.py                         # 차선/안전 판단 기반 조향, 속도 제어
  lidar.py                           # RPLidar scan 분석
  obstacles.py                       # 비전 장애물 감지와 차선 변경 경로 생성
  safety.py                          # LiDAR, 신호등, 비전 장애물 안전 판단 통합
  traffic_light.py                   # 신호등 객체/색상 판정
  ui.py                              # 화면 오버레이
  vision/
    preprocess.py                    # ROI crop, bird-eye view
    yolo_lane.py                     # YOLO 차선/객체/신호등 인식
arduino/CTRL_ZERO_Controller/        # Arduino 업로드용 펌웨어
scripts/                             # 카메라 확인, smoke test, YOLO export 도구
tests/                               # 제어/안전/차선 로직 테스트
docs/                                # 하드웨어, 모델, 튜닝 문서
```

## 설치

Python 3.10 이상을 권장합니다. 현재 기본 설정은 CPU 실행입니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## YOLO 모델

기본 backend는 `yolo`입니다. 일반 COCO YOLO 모델은 차선 클래스를 학습하지 않았으므로, 차선/객체/신호등 클래스를 포함한 학습 모델을 사용해야 합니다.

기본 모델 경로:

```text
models/yolo/final.pt
```

다른 모델을 사용할 때:

```powershell
python main.py --mode vision --backend yolo --yolo-model models\yolo\lane.pt
```

ONNX/OpenVINO export가 필요하면:

```powershell
python -m pip install -r requirements-export.txt
python scripts\export_yolo_formats.py --model models\yolo\final.pt
```

## 실행 방법

카메라와 인식 화면만 확인합니다. 이 모드는 Arduino 모터 출력을 보내지 않습니다.

```powershell
python main.py --mode vision --backend yolo --yolo-model models\yolo\final.pt
```

Arduino 포트를 확인합니다.

```powershell
python main.py --list-ports
```

수동 주행 모드입니다.

```powershell
python main.py --mode manual --backend yolo --yolo-model models\yolo\final.pt --arduino-port COM3
```

자동 주행 모드입니다.

```powershell
python main.py --mode auto --backend yolo --yolo-model models\yolo\final.pt --arduino-port COM3
```

## 키 조작

`--mode manual` 또는 `--mode auto`처럼 모터 출력이 켜진 실행에서는 프로그램 시작 직후 바로 출발하지 않습니다. 화면 또는 콘솔에서 `d` 또는 `D`를 눌러야 모터 출력이 시작됩니다.

- `d` / `D`: 출발 또는 재출발
- `space`: 정지하고 다시 출발 대기 상태로 전환
- `q`: 종료
- `+` / `=`: 최대 속도 증가
- `-` / `_`: 최대 속도 감소
- `l`: 로그 저장 on/off
- 수동 모드에서 출발 후 `w` / `s`: 속도 증가/감소
- 수동 모드에서 출발 후 `a` / `d`: 좌/우 조향 pulse
- 수동 모드에서 출발 후 `c`: 조향 중앙 복귀

주의: 수동 모드에서도 시작 전의 `d`는 출발 키로 먼저 처리됩니다. 출발한 뒤에는 `d`가 오른쪽 조향 pulse로 동작합니다.

## 작동 방식

1. `CameraReader`가 프레임을 읽습니다.
2. ROI crop과 bird-eye view가 켜져 있으면 전처리합니다.
3. YOLO가 차선, 장애물 객체, 신호등 객체를 인식합니다.
4. LiDAR가 켜져 있으면 전방 거리 기반 감속/정지를 계산합니다.
5. 신호등은 red/yellow 상태이고 bbox 면적이 기준 이상일 때만 정지합니다.
6. 비전 장애물이 켜져 있으면 현재 차선 전방 객체를 보고 회피 차선 변경 경로를 만듭니다.
7. `safety.py`가 LiDAR, 신호등, 비전 장애물 판단을 통합합니다.
8. `control.py`가 최종 조향/속도 명령을 계산합니다.
9. 출발 대기 상태가 아니면 Arduino로 `steer,speed`를 전송합니다.
10. 화면에는 차선, 신호등 면적, 장애물, 회피 경로, 안전 사유가 표시됩니다.

## 최근 변경사항

- 실행 직후 자동 출발을 막고, `d/D` 입력 후에만 모터 출력이 시작되도록 변경했습니다.
- `space`를 누르면 모터를 정지하고 다시 출발 대기 상태로 돌아갑니다.
- 디스플레이가 꺼져 있어도 콘솔 키 입력으로 `d/D`, `space`, `q`를 받을 수 있게 했습니다.
- 비전 장애물 회피가 단순 조향 보정에서 차선 변경 경로 기반으로 확장되었습니다.
- 차선 변경 중 현재 차선에서 목표 차선까지 부드럽게 이어지는 path를 생성하고 오버레이에 표시합니다.
- 신호등 정지는 red/yellow 판정만으로 멈추지 않고, bbox 면적 비율이 기준 이상일 때만 적용됩니다.
- UI 오버레이에 신호등 면적, 비전 장애물, 회피 목표 차선 정보가 표시됩니다.

## 튜닝해야 하는 main.py 라인

아래 라인 번호는 현재 `main.py` 기준입니다. 값은 모두 `main.py` 상단 `USER TUNING PARAMETERS` 영역에 있습니다.

| 라인 | 파라미터 | 조정 목적 |
| --- | --- | --- |
| 30-31 | `RUN_MODE`, `LANE_BACKEND` | 기본 실행 모드와 차선 backend |
| 34-38 | `CAMERA_INDEX`, `CAMERA_BACKEND`, `CAMERA_WIDTH`, `CAMERA_HEIGHT`, `CAMERA_FPS` | 카메라 장치, backend, 해상도/FPS |
| 41-44 | `USE_ARDUINO`, `ARDUINO_PORT`, `ARDUINO_BAUDRATE`, `DRIVE_MAX_PWM` | Arduino 사용 여부, 포트, 모터 PWM 상한 |
| 47-54 | `USE_LIDAR`, `LIDAR_*` | LiDAR 사용 여부, 전방 각도 범위, 정지/감속 거리 |
| 57-65 | `VISION_OBSTACLE_*` | 비전 장애물 사용 여부, 장애물 confidence, 차선 변경 시작 크기, 회피 path 속도 |
| 68 | `TRAFFIC_LIGHT_STOP_AREA_RATIO` | 신호등 정지 bbox 면적 기준 |
| 71-94 | `YOLO_*` | 모델 경로, confidence, class 이름, 차선 mask/점선/곡선 처리, 목표 차선 선택 |
| 98-109 | `ROI_*`, `BIRD_EYE_*` | 입력 영역 crop과 bird-eye view 원근 변환 |
| 112-114 | `DEFAULT_LANE_WIDTH_RATIO`, `MIN_LANE_WIDTH_RATIO`, `MAX_LANE_WIDTH_RATIO` | 차선 폭 추정 범위 |
| 117-130 | `CONTROL_MODE`, `MIN_SPEED`, `MAX_SPEED`, `CONTEST_*`, `MAX_STEER`, `REVERSE_STEER`, `MAX_HOLD_FRAMES`, `HOLD_DECEL_STEP` | 조향/속도 제어, 최대 조향, 차선 소실 시 감속 |
| 133-135 | `MANUAL_*` | 수동 모드 속도 step, 조향 power, 조향 pulse 유지 시간 |
| 138-145 | `DISPLAY_*`, `LOG_*`, `SAVE_EVERY_N_FRAMES`, `PRINT_EVERY_N_FRAMES`, `START_KEYS` | 화면/로그/출력 주기와 출발 키 |

### 자주 조정하는 값

- 차선이 흔들리면 `YOLO_CONFIDENCE`, `YOLO_MIN_POINTS_PER_LANE`, `YOLO_MIN_VALID_Y_SPAN_RATIO`, `YOLO_DASHED_MERGE_MAX_X_GAP_RATIO`를 먼저 조정합니다.
- 차가 중앙에서 치우쳐 달리면 `YOLO_TARGET_LANE_PAIR`, `YOLO_TARGET_PATH_MODE`, `YOLO_LANE_PAIR_TARGET_OFFSET_RATIO`, `CONTEST_POSITION_WEIGHT`를 조정합니다.
- 조향이 너무 급하면 `CONTEST_STEER_LIMIT`, `MAX_STEER`, `CONTEST_ANGLE_WEIGHT`를 낮춥니다.
- 속도가 너무 빠르거나 느리면 `MIN_SPEED`, `MAX_SPEED`, `DRIVE_MAX_PWM`를 조정합니다.
- 빨간/노란 신호등에서 너무 늦게 멈추면 `TRAFFIC_LIGHT_STOP_AREA_RATIO`를 낮춥니다. 너무 자주 멈추면 높입니다.
- 비전 장애물 회피를 쓰려면 `VISION_OBSTACLE_ENABLED = True`로 켭니다.
- 장애물을 너무 쉽게 회피하면 `VISION_OBSTACLE_MIN_CONFIDENCE`나 `VISION_OBSTACLE_LANE_CHANGE_AREA_RATIO`를 높입니다.
- 차선 변경 path가 너무 빠르면 `VISION_OBSTACLE_LANE_CHANGE_PATH_PROGRESS_STEP`을 낮춥니다.
- 차선 변경 완료 판정이 빨리/늦게 끝나면 `VISION_OBSTACLE_LANE_CHANGE_COMPLETE_OFFSET_NORM`, `VISION_OBSTACLE_LANE_CHANGE_COMPLETE_FRAMES`를 조정합니다.
- 카메라 설치 각도가 바뀌면 `ROI_*`와 `BIRD_EYE_SRC_*`를 다시 맞춥니다.

## 검증

하드웨어 없이 가능한 기본 검증:

```powershell
python scripts\smoke_test.py
pytest
```

실차 주행 전 최소 확인:

1. `python main.py --mode vision ...`으로 카메라, 차선, 신호등, 장애물 표시를 확인합니다.
2. `python main.py --list-ports`로 Arduino 포트를 확인합니다.
3. 바퀴를 띄운 상태에서 `manual` 모드로 `d/D`, `space`, `q` 동작을 확인합니다.
4. `auto` 모드에서는 시작 직후 정지 상태인지 확인한 뒤 `d/D`로 출발시킵니다.

## 문서

- [하드웨어 설정](docs/하드웨어_설정.md)
- [모델 가이드](docs/모델_가이드.md)
- [튜닝 가이드](docs/튜닝_가이드.md)
