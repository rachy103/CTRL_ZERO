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
    classical_lane.py                # OpenCV 차선 검출 fallback
    ufldv2.py                        # UFLDv2 ResNet34 CPU 추론 어댑터
arduino/CTRL_ZERO_Controller/        # Arduino 업로드용 펌웨어
scripts/download_ufldv2_weights.py   # 공식 pretrained weight 다운로드
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

## 모델 다운로드

기존 작업의 Tusimple ResNet18 대신, 기본값은 UFLDv2 공식 CULane ResNet34입니다. ResNet34는 ResNet18보다 깊고, CULane 기준 공식 F1도 더 높습니다.

```powershell
python scripts\download_ufldv2_weights.py --model culane_res34
```

다운로드 파일은 `models/ufldv2/culane_res34.pth`에 저장됩니다. 모델 파일은 GitHub에 커밋하지 않습니다.

## 실행

먼저 카메라와 화면만 확인합니다. 이 모드는 Arduino로 모터 명령을 보내지 않습니다.

```powershell
python main.py --mode vision --backend opencv
```

ResNet34 모델을 받은 뒤 딥러닝 차선 인식을 확인합니다.

```powershell
python main.py --mode vision --backend ufldv2
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
python main.py --mode auto --backend ufldv2 --arduino-port COM3
```

## 튜닝 방식

튜닝 파라미터는 `main.py` 상단 `USER TUNING PARAMETERS` 영역에 모아두었습니다.

- `CAMERA_INDEX`, `CAMERA_BACKEND`: 카메라 장치 선택
- `LANE_BACKEND`: `ufldv2` 또는 `opencv`
- `UFLDV2_CONFIG_PATH`, `UFLDV2_MODEL_PATH`: 딥러닝 차선 모델 선택
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
