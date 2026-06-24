# CTRL_ZERO

카메라, LiDAR, Arduino, DC 모터/조향 모터를 이용한 소형 자율주행 실무 코드입니다. 기존 교육용 예제에서 실제 주행에 필요한 기능만 분리했고, `main.py`가 카메라, 차선 인식, LiDAR 장애물 판단, Arduino 모터 출력을 조립합니다.

## 핵심 구조

```text
main.py                              # 실행 모드와 튜닝 파라미터를 한곳에서 조정
ctrl_zero/
  camera.py                          # OpenCV 카메라 입력
  arduino.py                         # Arduino 직렬 통신과 steer,speed 전송
  lidar.py                           # RPLidar 스캔과 전방 장애물 판단
  control.py                         # 차선/장애물 기반 조향, 속도 제어
  logger.py                          # CSV와 프레임 저장
  ui.py                              # 화면 오버레이
  vision/
    preprocess.py                    # ROI crop, Bird-eye view 전처리
    classical_lane.py                # OpenCV 차선 검출 fallback
    yolo_lane.py                     # YOLO 차선 검출/세그멘테이션 어댑터
arduino/CTRL_ZERO_Controller/        # Arduino 업로드용 펌웨어
scripts/export_yolo_formats.py       # YOLO .pt를 ONNX/OpenVINO로 export
scripts/smoke_test.py                # 카메라 없이 실행 가능한 최소 검증
docs/                                # 하드웨어, 모델, 튜닝 문서
```

## 설치

Python 3.10 이상을 권장합니다. CUDA는 필요하지 않고, 현재 기본 설정도 CPU입니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## YOLO 차선 모델

`main` 브랜치는 YOLO backend를 기본값으로 사용합니다. 단, 일반 COCO YOLO 모델은 차선 클래스를 학습하지 않았으므로 차선을 잡지 못합니다. 차선으로 학습된 Ultralytics YOLO `.pt` 파일을 준비해 아래 경로에 둡니다.

```text
models/yolo/lane.pt
```

다른 위치에 있다면 실행 시 `--yolo-model`로 지정합니다.

ONNX나 OpenVINO 포맷도 지정할 수 있습니다. 현재 로컬에는 `yolov8n-seg.pt`, `yolov8n-seg.onnx`, `yolov8n-seg_openvino_model/`을 만들어두었습니다.

```powershell
python -m pip install -r requirements-export.txt
python scripts\export_yolo_formats.py --model yolov8n-seg.pt
```

## 실행

먼저 카메라와 화면만 확인합니다. 이 모드는 Arduino로 모터 명령을 보내지 않습니다.

```powershell
python main.py --mode vision --backend opencv
```

YOLO 차선 모델을 확인합니다.

```powershell
python main.py --mode vision --backend yolo --yolo-model models\yolo\lane.pt
```

ROI crop은 기본으로 켜져 있습니다. Bird-eye view는 카메라 각도에 맞춰 기준점을 잡아야 하므로 기본값은 꺼져 있고, 필요할 때 켭니다.

```powershell
python main.py --mode vision --backend yolo --bird-eye --yolo-model models\yolo\lane.pt
```

Arduino 포트를 확인합니다.

```powershell
python main.py --list-ports
```

수동 주행 테스트입니다.

```powershell
python main.py --mode manual --backend opencv --arduino-port COM3
```

자동 주행입니다.

```powershell
python main.py --mode auto --backend yolo --yolo-model models\yolo\lane.pt --arduino-port COM3
```

## 튜닝 방식

튜닝 파라미터는 `main.py` 상단 `USER TUNING PARAMETERS` 영역에 모아두었습니다.

- `CAMERA_INDEX`, `CAMERA_BACKEND`: 카메라 장치 선택
- `LANE_BACKEND`: `yolo` 또는 `opencv`
- `YOLO_MODEL_PATH`, `YOLO_IMAGE_SIZE`, `YOLO_CONFIDENCE`: YOLO 차선 모델 선택
- `ROI_TOP_RATIO`, `ROI_LEFT_RATIO`, `ROI_RIGHT_RATIO`: YOLO 입력에서 실제 도로 영역만 crop
- `BIRD_EYE_ENABLED`, `BIRD_EYE_SRC_*`: Bird-eye view 원근 변환 기준점
- `BASE_SPEED`, `MAX_SPEED`: 기본 속도와 최대 속도
- `KP_OFFSET`, `KP_HEADING`, `KD_OFFSET`: 조향 제어 게인
- `USE_LIDAR`, `LIDAR_STOP_DISTANCE_MM`, `LIDAR_SLOW_DISTANCE_MM`: 장애물 감속/정지
- `USE_ARDUINO`, `ARDUINO_PORT`: 모터 출력 여부와 포트

자세한 내용은 [튜닝 가이드](docs/튜닝_가이드.md)를 보세요.

## 검증

하드웨어 없이 최소 로직을 확인합니다.

```powershell
python scripts\smoke_test.py
pytest
```

## 문서

- [하드웨어 설정](docs/하드웨어_설정.md)
- [모델 가이드](docs/모델_가이드.md)
- [튜닝 가이드](docs/튜닝_가이드.md)
