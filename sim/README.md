# CTRL_ZERO 차량 주행 시뮬레이션

이 폴더는 실제 유아용 차량과 트랙을 MuJoCo에서 재현하기 위한 독립 실행 영역입니다. 새 코드, 생성 자산, 설정, 캐시, 임시 파일은 모두 `sim` 아래에만 기록합니다. 루트의 `main.py`, `arduino`, `models`는 읽기만 하며 수정하지 않습니다.

측정하지 않은 물리·하드웨어 값은 `null`로 남겨 둡니다. 실행기가 임의의 기본값이나 예상값으로 바꾸지 않으며, 필요한 값이 하나라도 없으면 어떤 항목이 비었는지 출력하고 중단합니다.

## 현재 구현 상태

- 제공된 RGB 이미지에서 트랙 영역을 자르고, 회전한 도면 프레임의 종횡비에
  맞춰 Lanczos 보간으로 확대합니다.
- 도면의 cyan 외곽 프레임 네 모서리를 RGB 맵 외곽 네 모서리에 정규화한 뒤
  선 도면을 겹쳐 정합 미리보기를 만듭니다.
- 알려진 차체 외형 치수 `0.99 × 0.55 × 0.41 m`는 사용자 제공 이미지에서 가져왔습니다.
- 사진에서 실제로 보이는 차체 양 끝점과 왼쪽 앞·뒤 바퀴 중심을 표시해
  앞·뒤 차축의 세로 방향 위치와 축간거리를 구하는 GUI가 있습니다.
- MuJoCo용 `world.xml`과 시각 확인용 `vehicle.urdf` 생성기가 있습니다.
- `sim_run.py`는 루트 `main.py`의 `DriveController`와 조작 규칙을 불러와 시뮬레이션 명령에 사용합니다. 매 tick마다 차량 상태를 갱신하고 `mj_step()`을 호출합니다.
- 자동 모드는 `main.py`와 같은 차선 제어 및 장애물 미션 경로를 사용합니다. 현재 시뮬레이터에는 LiDAR 입력이 없으므로 `main.py`의 LiDAR 조건이 켜져 있으면 `lidar_scan=None`으로 전달되어 장애물 회피가 시작되지 않습니다.

현재 생성된 `track_texture_unscaled_preview.png`는 crop·보간·도면 정합만 확인하는 미리보기입니다. 실제 바닥 크기를 입력하기 전에는 최종 물리 맵이 아닙니다.

## 1. 사진에서 차량 형상 측정

```powershell
python -B .\sim\calibrate_vehicle.py
```

GUI에서 사진에 실제로 보이는 다음 여섯 점만 순서대로 누릅니다.

1. 차체의 맨 뒤 끝점
2. 차체의 맨 앞 끝점
3. 보이는 왼쪽 뒤 바퀴의 중심
4. 보이는 왼쪽 뒤 바퀴의 바깥 둘레 한 점
5. 보이는 왼쪽 앞 바퀴의 중심
6. 보이는 왼쪽 앞 바퀴의 바깥 둘레 한 점

제공된 옆면 사진 기준으로 차량 앞은 화면 왼쪽, 뒤는 화면 오른쪽입니다.
따라서 이 사진에서는 오른쪽의 차체 뒤 끝점을 먼저 누르고, 왼쪽의 차체 앞
끝점을 누릅니다. 바퀴도 같은 순서로 화면 오른쪽의 뒤 바퀴 중심과 둘레,
화면 왼쪽의 앞 바퀴 중심과 둘레를 누릅니다. 둘레 점은 중심에서 가장
분명하게 보이는 바깥 타이어 경계에 찍습니다.

GUI는 바퀴 중심을 뒤 끝점에서 앞 끝점으로 향하는 차체 길이 축에 투영하고,
알려진 전체 길이 `0.99 m`로 길이 방향 거리와 바퀴 반지름을 환산합니다.
앞·뒤 바퀴에서 따로 구한 반지름의 평균을 시뮬레이션 반지름으로 저장하며,
각 측정값도 결과 JSON에 남깁니다. `U`는 마지막 점 취소, `R`은 초기화,
`S`는 검증 후 저장, `Q`는 종료입니다. 여섯 점이 모두 유효할 때만
`config/vehicle_geometry.json`에 아래 값을 저장합니다.

- `rear_axle_from_rear_m`
- `front_axle_from_rear_m`
- `wheelbase_m`
- `wheel_radius_m`

사진 한 장의 차체 양 끝점 두 개만으로는 원근을 완전히 제거할 수 없습니다.
따라서 저장되는 값은 정측면에 가까운 제품 사진에서 구한 투영값이며, 이 한계도
결과 JSON에 함께 기록합니다. 보이지 않는 오른쪽 바퀴를 대칭이라고 가정하지
않으므로 다음 윤거 값은 `null`로 남기고 실물을 직접 재서
`config/sim_config.json`에 입력해야 합니다.

- `vehicle.front_track_m`: 앞바퀴 좌우 중심 사이 거리
- `vehicle.rear_track_m`: 뒷바퀴 좌우 중심 사이 거리
- `vehicle.wheel_width_m`: 바퀴 폭

`vehicle.wheel_material`에는 사용자가 알려준 `polyurethane`을 기록했습니다.
재질명만으로 접지 마찰계수, 구름저항, 탄성계수를 정할 수는 없으므로 이런
물성치는 임의로 넣지 않습니다.

정면 사진과 알려진 전체 폭 `0.55 m`를 비교해 앞·뒤 윤거는 각각 `0.47 m`,
바퀴 폭은 `0.08 m`로 추정했습니다. 여기서 윤거는 타이어 안쪽 면 사이의
빈 공간이 아니라 좌우 바퀴 중심선 사이의 거리입니다. 사용자가 앞·뒤 치수가
같다고 확인하고 근삿값 사용을 허용했으며, 출처 사진은
`assets/source/vehicle_front_profile.png`에 보관합니다.

## 2. 맵 실제 크기 확정

도면 외곽 치수가 `9,000 × 13,000 mm`인지 먼저 확인해야 합니다. 회전된 시뮬레이션 맵의 실제 가로·세로가 `13 × 9 m`로 확인된 경우에만 다음처럼 최종 텍스처와 설정을 만듭니다.

```powershell
python -B .\sim\build_map.py --width-m 13 --height-m 9 --write-config
```

치수가 다르면 확인된 meter 값을 넣습니다. 이 명령은 `assets/generated/track_texture.png`와 정합 미리보기를 만들고 `map.physical_width_m`, `map.physical_height_m`를 기록합니다.

실제 치수를 쓰기 전 crop과 정합 상태만 다시 확인하려면 다음 명령을 사용합니다.

```powershell
python -B .\sim\build_map.py --preview-only
```

맵 좌표 원점은 중앙입니다. 가로는 world `x`, 세로는 world `y`이며, 차량 위치는 뒷차축 중심으로 기록합니다. `yaw=0°`에서는 차량 앞쪽이 `+x`를 향합니다. 시작 위치는 다음 세 항목을 실제 배치에 맞게 입력합니다.

- `map.spawn.rear_axle_x_m`
- `map.spawn.rear_axle_y_m`
- `map.spawn.yaw_deg`

## 3. 반드시 측정해야 하는 하드웨어 값

`config/sim_config.json`의 `null` 항목을 실측값으로 채웁니다.

조향:

- `hardware.steering.straight_command`: 바퀴가 실제로 직진을 향하는 명령
- `hardware.steering.left_angle_deg_at_steer_min`: `steer=-100`일 때 실차 왼쪽 조향각
- `hardware.steering.right_angle_deg_at_steer_max`: `steer=+100`일 때 실차 오른쪽 조향각

구동:

- `hardware.drive.driven_axle`: 실제 구동축인 `front`, `rear`, `all` 중 하나
- `hardware.drive.pwm_speed_table`: 측정한 `[PWM, 정상상태 속도(m/s)]` 행 목록
- `hardware.drive.acceleration_limit_mps2`: PWM을 올렸을 때 측정한 최대 가속도
- `hardware.drive.braking_deceleration_mps2`: 제동할 때 측정한 감속도
- `hardware.drive.zero_pwm_behavior`: PWM 0에서 실제 동작이 관성 주행이면 `coast`, 능동 제동이면 `brake`
- `hardware.drive.coast_deceleration_mps2`: `zero_pwm_behavior`가 `coast`일 때 측정한 자연 감속도

현재 `zero_pwm_behavior`는 사용자 확인에 따라 `brake`로 기록했습니다.

`pwm_speed_table`은 PWM 오름차순으로 적고, 각 속도는 같은 조건에서 측정한 값이어야 합니다. 현재 펌웨어에서 확인된 PWM 범위와 deadband를 기준으로 다음 조건을 모두 만족해야 합니다.

- 첫 행은 PWM `-255`의 실측 후진 속도, 마지막 행은 PWM `255`의 실측 전진 속도입니다.
- PWM `-8`, `0`, `8` 행을 포함하며 각 속도는 `0 m/s`입니다.
- PWM이 커질수록 속도는 감소하지 않아야 합니다.
- 구간별 실제 응답을 반영하려면 운전 가능한 범위의 측정 행을 추가합니다. 시뮬레이터는 인접한 실측 행 사이만 선형 보간합니다.

속도와 가속도는 배터리 상태, 탑승 하중, 노면을 기록한 동일한 시험 조건에서 측정해야 합니다. 측정하지 않은 값을 표에 넣으면 안 됩니다.

자동 주행 카메라도 실차와 맞춰야 합니다.

- `hardware.camera.width_px`, `height_px`, `fps`
- `hardware.camera.horizontal_fov_deg`
- `hardware.camera.position_from_rear_axle_m`: 뒷차축 중심 기준 `[앞, 왼쪽, 위]` 위치
- `hardware.camera.pitch_deg`, `yaw_deg`
- `auto.yolo_model`: 읽을 수 있는 학습 모델 경로. 새 모델 자산을 추가한다면
  `sim/assets/models` 아래에 둔 뒤 그 경로를 입력합니다.

## 펌웨어에서 확인한 값

다음 값은 `arduino/CTRL_ZERO_Controller/CTRL_ZERO_Controller.ino`에서 직접 읽었으며 설정 파일에 이미 기록했습니다.

- 조향 명령 범위 `-100..100`, 양수는 우회전
- 구동 PWM 범위 `-255..255`, 양수는 전진
- 무출력 구간 `abs(PWM) <= 8`
- 조향 포텐셔미터 왼쪽/오른쪽 `573/436`, 안전 범위 `436..573`
- 조향 PID `Kp=2.2`, `Ki=0`, `Kd=0.35`, deadband `6 count`
- 조향 모터 feed-forward/max PWM `55/255`
- 조향 목표 slew `900 command-unit/s`
- 좌우 구동 채널에 같은 PWM 전달, 두 채널 모두 inverted
- 명령 timeout `500 ms`

`CONTEST_STEERING_ANGLE_NORM_DEG=50`은 영상 차선 오차를 조향 명령으로 정규화할 때 쓰는 값입니다. 실차의 최대 조향각이 아니므로 물리 모델에 사용하지 않습니다.

## 4. 검증, 모델 생성, 실행

제공된 AMG `.blend` 원본에서 차체와 바퀴 메쉬를 다시 만들려면 Blender
백그라운드 모드로 다음 스크립트를 실행합니다.

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" `
  --background --python .\sim\prepare_vehicle_mesh.py
```

이 변환은 앞끝·앞차축·뒷차축·뒤끝을 현재 측정 위치에 각각 고정하는 구간별
길이 변형을 사용합니다. 차체 최외곽 폭은 `0.55 m`, 높이는 `0.41 m`에
맞추고, 원본의 각 타이어·림·브레이크·휠 로고를 바퀴 중심 기준으로 분리해
측정 반지름과 폭에 맞춥니다. 생성 메쉬와 치수 근거는
`assets/generated/vehicle_mesh`와 `config/vehicle_visual_mesh.json`에
기록합니다.

주행 파라미터가 아직 비어 있어도 현재 맵과 출발 자세만 확인할 수 있습니다.

```powershell
python -B .\sim\preview_scene.py
```

이 명령은 `assets/generated/scene_spawn_overview.png`와
`assets/generated/scene_spawn_closeup.png`를 만듭니다. MuJoCo 창에서 직접
돌려 보려면 다음처럼 실행합니다.

```powershell
python -B .\sim\preview_scene.py --interactive
```

필수 측정값이 모두 채워졌는지 확인합니다.

```powershell
.\sim\run.ps1 --validate
```

자동 모드까지 검증하려면 다음처럼 실행합니다.

```powershell
.\sim\run.ps1 --mode auto --validate
```

맵과 차량 모델만 생성합니다.

```powershell
.\sim\run.ps1 --build-only
```

수동 주행:

```powershell
.\sim\run.ps1 --mode manual
```

키 조작은 루트 `main.py`의 수동 조작과 같습니다.

- 첫 `D`: 주행 시작
- `W/S`: PWM을 10씩 증감
- `A/D`: `-80/+80` 조향 pulse
- `C`: 조향 명령 0
- `Space`: 즉시 정지하고 수동 모드 유지
- `Q` 또는 `Esc`: 종료

자동 차선 주행:

```powershell
.\sim\run.ps1 --mode auto
```

키 입력이 없는 headless 자동 회귀 실행에서는 시작 게이트를 명시적으로 엽니다.

```powershell
.\sim\run.ps1 --mode auto --auto-start --headless --steps <step수>
```

주차 미션 (수직주차):

```powershell
.\sim\run.ps1 --mode parking --parking-case <1|2> --parking-start <1..4>
```

`--parking-case`는 장애물 차량 배치(1=베이 2·4, 2=베이 1·3), `--parking-start`는
도면에 그려진 3cm 시작 마커 번호입니다. 시뮬레이션은 `hardware.lidar` 실측
장착값으로 2D LiDAR를 레이캐스트해 main.py의 `ParkingHardMission`에 그대로
공급합니다. 장애물 차량 박스는 `world_parking_case<case>.xml`에 생성됩니다.

main.py의 현재 주차 상수(`PARKING_INIT_STEER=7`로 후진)는 수직주차 기하에서
회전이 거의 없어 `REVERSE_RIGHT`에서 교착됩니다. 시뮬레이션 전용 오버라이드로
검증한 동작 조합은 다음과 같습니다 (main.py는 수정하지 않음).

- 북쪽 열 마커(2, 4): `--parking-reverse-steer 100 --parking-search-overrun-s 4.5`
- 남쪽 열 마커(1, 3): `--parking-reverse-steer 85 --parking-search-overrun-s 5.6`
- 공통: `--parking-reverse-straight-max-s 0.5`

`--parking-search-overrun-s`는 두 번째 장애물 차량 통과 후 정지 전에 추가
전진하는 시간, `--parking-reverse-straight-max-s`는 전방 장착 LiDAR로는 양측
클리어 조건이 차량 전체가 베이를 관통할 때까지 후진시키므로 직선 후진을
시간으로 제한하는 프로토타입입니다. 실차 적용은 main.py에 같은 로직을
반영해야 합니다.

`--parking-strategy timed`는 실차의 `main.py --mode parking_sim`이 사용하는
`ctrl_zero/parking_timed.py`의 `TimedParkingMission`을 그대로 구동합니다:
장애물 차량 두 대를 카운트해 두 번째 차량 감지를 기준점으로 → 정해진
시간만큼 직진 접근 → 풀락 좌회전 전진(θ₁) → 정지(조향 우측 선회 예비) →
풀락 우회전 후진(θ₂=90°−θ₁) → 직선 후진을 LiDAR 양측(±90°) 감지로 정지.
두 번째 차량 기준이라 dead-reckoning 구간이 남쪽 열 1.0 m, 북쪽 열 0.3 m
수준으로 짧아 속도 오차에 강합니다.

구간 시간은 main.py의 `PARKING_SIM_APPROACH_S`/`SWING_S`/`ARC_S` 단일
상수에서 옵니다. 시작 마커 4곳 모두 같은 값을 씁니다: 마커의 세로(행)
오프셋은 두 번째 차량 감지 트리거가, 가로(열) 오프셋은 90° 회전을 거치며
아크 종료 깊이 차이로 바뀌어 LiDAR 정지 직선 후진이 흡수합니다. 스윙
각도(θ₁≈17°)는 경계선에 가장 가까운 남쪽 열 기준으로 잡았습니다. 추첨
후 프로그램 수정이 불가한 대회 규정에서도 마커 입력이 필요 없습니다.
시뮬레이션은 실측 기하로 재유도한 값(`timed_plan()`)을 나란히 출력해
상수와의 어긋남을 드러냅니다. 감지 디바운스는 시간 기반(0.2 s hold)이라
실차 카메라 루프 주기와 시뮬레이션 스텝 주기에서 같은 동작을 합니다.
8개 조합(마커 4 × 케이스 2) 검증 결과 최종 요는 수직 ±1.3° 이내, 베이
중심 대비 x 오차 3.8 cm 이내입니다. 실차 트림은 main.py 상수 주석의
규칙(치우친 거리 D cm → approach_s를 D/29 s 조정)을 따릅니다.

```powershell
.\sim\run.ps1 --mode parking --parking-strategy timed --parking-case <1|2> --parking-start <1..4>
```

뷰어 없이 정해진 step만 검증할 때는 명시적인 수동 명령을 함께 줍니다.

```powershell
python -B .\sim\sim_run.py --headless --steps <step수> --command-steer <명령> --command-pwm <명령>
```

각 꺾쇠 항목에는 검증하려는 값을 직접 넣습니다. 위 표기는 실행 가능한 기본값을 뜻하지 않습니다.

## 모델 범위

현재 차량 운동은 측정한 축간거리, 바퀴 크기, 조향각, PWM-속도 표, 가속·감속 응답을 사용하는 kinematic bicycle 모델입니다. Ackermann 좌우 앞바퀴 각도와 바퀴 회전을 계산하고, 결과를 MuJoCo mocap 시각 요소에 동기화합니다.

질량, 관성, 무게중심, 모터 토크, 타이어 마찰 데이터는 제공되지 않았으므로 동역학 값으로 꾸며 넣지 않았습니다. `vehicle.urdf`는 현재 시각 확인용이며, 실제 운동은 측정 기반 kinematic 모델이 담당합니다.
차체 primitive는 제공된 전체 길이·폭·높이의 외곽 envelope를 보여 주는
시각 proxy이며 실제 차체 표면을 복원한 mesh는 아닙니다.

## 실행 환경과 테스트

Windows Python 환경에서 MuJoCo 3.9와 OpenGL 뷰어를 사용할 수 있습니다. 별도 환경이 필요하면 가상환경과 캐시를 `sim` 안에만 만듭니다.

```powershell
.\sim\setup.ps1
```

자동 모드의 YOLO/PyTorch 의존성까지 같은 `sim/.venv`에 설치하려면 다음을
사용합니다. 새 환경과 다운로드 캐시는 계속 `sim` 아래에만 생성됩니다.

```powershell
.\sim\setup.ps1 -Auto
```

테스트와 쓰기 범위 검사는 다음 명령으로 실행합니다.

```powershell
.\sim\test.ps1
python -B .\sim\scope_guard.py
```

테스트는 `sim` 밖 경로 차단, 누락값 hard error, 차량 보정 파일 누락·불완전 상태, 맵 crop·정합·보간, 사진 투영 계산, 실측 PWM-속도 보간, 조향 slew, Ackermann, 가속·제동·관성 감속을 검사합니다.
