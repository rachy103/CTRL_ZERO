// CTRL_ZERO Arduino motor controller for Arduino Mega 2560.
//
// Serial protocol from Python:
//   steer,speed\n
//   steer: -100..100, positive means right target angle
//   speed: -255..255, positive means forward
// Diagnostic / tuning (send one token per line in the Serial Monitor):
//   ?\n  - print one STATUS line (pot, target, err, ...)
//   p\n  - toggle live Serial-Plotter streaming: "target pot err ctrl"
// Serial runs at 115200 baud -> set the Serial Monitor/Plotter to 115200
// (and main.py ARDUINO_BAUDRATE = 115200).

const int DRIVE_L_IN1 = 4;
const int DRIVE_L_IN2 = 5;
const int DRIVE_R_IN1 = 6;
const int DRIVE_R_IN2 = 7;
const int STEER_IN1 = 2;
const int STEER_IN2 = 3;
const int STEER_POT = A0;

const bool DRIVE_L_INVERTED = true;
const bool DRIVE_R_INVERTED = true;
const bool STEER_INVERTED = true;

// 가변저항 값 범위: 사용자가 실측한 딱 엣지값.
const int POT_LEFT = 575;
const int POT_RIGHT = 429;
const int POT_MIN_SAFE = 429;
const int POT_MAX_SAFE = 575;

const float STEER_KP = 4.4f; //2.2f
const float STEER_KI = 60.0f; //0.0f
const float STEER_KD = 0.05f; // 0.35f
const int STEER_FF_PWM = 10; //55
const int STEER_MAX_PWM = 255;
const int STEER_DEADBAND_CNT = 1; //6
const float STEER_SLEW_PER_MS = 1.0f;

// --- PID structure (time-correct integral + anti-windup + hold) -------------
const float STEER_I_LIMIT_PWM = 40.0f;   // anti-windup: cap on PWM the I-term may contribute
const int   STEER_I_BAND_CNT  = 15;      // integrate ONLY when |err| <= this (counts), i.e. near target
const int   STEER_POT_SAMPLES = 4;       // analogRead averaging to cut ADC noise
const bool  STEER_BRAKE_ON_HOLD = true;  // hold via active brake (both pins HIGH); false = coast

const int DRIVE_MAX_PWM = 255;
const int DRIVE_DEADBAND_PWM = 8;
const unsigned long COMMAND_TIMEOUT_MS = 500;

char inputBuf[40];
uint8_t inputLen = 0;

float targetSteer = 0.0f;
float desiredSteer = 0.0f;
int driveCmdPwm = 0;

float steerIntegral = 0.0f;
int lastPotErr = 0;
unsigned long lastCommandMs = 0;
unsigned long lastLoopMs = 0;

// Live tuning plot (Serial Plotter). Default OFF so main.py runs safely; send
// 'p' in the Serial Monitor to start/stop streaming while tuning.
// (Streaming while Python is connected overflows the serial buffer -> the PC's
//  write times out, exactly the crash seen in --mode parking_sim.)
const unsigned long PLOT_INTERVAL_MS = 20;   // 50 Hz stream when enabled
bool plotEnabled = false;
unsigned long lastPlotMs = 0;
int g_potTarget = 0, g_pot = 0, g_err = 0, g_ctrl = 0;  // snapshot for plotStatus()

void setup() {
  Serial.begin(115200);
  pinMode(DRIVE_L_IN1, OUTPUT);
  pinMode(DRIVE_L_IN2, OUTPUT);
  pinMode(DRIVE_R_IN1, OUTPUT);
  pinMode(DRIVE_R_IN2, OUTPUT);
  pinMode(STEER_IN1, OUTPUT);
  pinMode(STEER_IN2, OUTPUT);
  pinMode(STEER_POT, INPUT);
  lastLoopMs = millis();
  holdAll();
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      inputBuf[inputLen] = '\0';
      if (strcmp(inputBuf, "?") == 0) {
        printStatus();
      } else if (strcmp(inputBuf, "p") == 0) {
        plotEnabled = !plotEnabled;
      } else {
        parseCommand(inputBuf);
      }
      inputLen = 0;
    } else if (c != '\r' && inputLen < sizeof(inputBuf) - 1) {
      inputBuf[inputLen++] = c;
    } else if (inputLen >= sizeof(inputBuf) - 1) {
      inputLen = 0;
    }
  }

  unsigned long now = millis();
  float dt = now - lastLoopMs;
  if (dt < 1) {
    return;
  }
  lastLoopMs = now;

  if (now - lastCommandMs > COMMAND_TIMEOUT_MS) {
    desiredSteer = 0.0f;
    driveCmdPwm = 0;
  }

  float maxStep = STEER_SLEW_PER_MS * dt;
  float diff = desiredSteer - targetSteer;
  if (diff > maxStep) diff = maxStep;
  if (diff < -maxStep) diff = -maxStep;
  targetSteer += diff;

  updateSteerPID(dt);
  applyDrive(driveCmdPwm);

  if (plotEnabled && (now - lastPlotMs >= PLOT_INTERVAL_MS)) {
    lastPlotMs = now;
    plotStatus();
  }
}

void parseCommand(char* line) {
  char* comma = strchr(line, ',');
  if (comma == NULL) return;
  *comma = '\0';
  int steer = atoi(line);
  int speed = atoi(comma + 1);
  desiredSteer = constrain(steer, -100, 100);
  driveCmdPwm = constrain(speed, -DRIVE_MAX_PWM, DRIVE_MAX_PWM);
  lastCommandMs = millis();
}

int steerToPotTarget(float steer) {
  float t = (steer + 100.0f) / 200.0f;
  float raw = POT_LEFT + t * (POT_RIGHT - POT_LEFT);
  return (int)constrain(raw, POT_MIN_SAFE, POT_MAX_SAFE);
}

int readSteerPot() {
  long sum = 0;
  for (int i = 0; i < STEER_POT_SAMPLES; i++) sum += analogRead(STEER_POT);
  return (int)(sum / STEER_POT_SAMPLES);
}

void steerHold() {
  // Hold the wheel: active brake (both inputs HIGH) if enabled, else coast.
  // If the driver misbehaves with both-HIGH, set STEER_BRAKE_ON_HOLD = false.
  if (STEER_BRAKE_ON_HOLD) {
    analogWrite(STEER_IN1, 255);
    analogWrite(STEER_IN2, 255);
  } else {
    hBridgeHold(STEER_IN1, STEER_IN2);
  }
}

void updateSteerPID(float dt) {
  int pot = readSteerPot();
  int potTarget = steerToPotTarget(targetSteer);
  int err = potTarget - pot;
  g_pot = pot; g_potTarget = potTarget; g_err = err;  // snapshot for Serial Plotter

  // Close enough: stop correcting and hold position.
  if (abs(err) <= STEER_DEADBAND_CNT) {
    steerIntegral = 0.0f;
    lastPotErr = err;
    g_ctrl = 0;
    steerHold();
    return;
  }

  float dt_s = dt / 1000.0f;  // dt arrives in milliseconds; integrate in seconds

  // Conditional integration + anti-windup: only accumulate near the target, and
  // clamp so the I-term can never exceed STEER_I_LIMIT_PWM.  This is what closes
  // the stiction dead-zone (what the constant feedforward used to cover, badly).
  if (STEER_KI > 0.0f && abs(err) <= STEER_I_BAND_CNT) {
    steerIntegral += err * dt_s;
    float iLimit = STEER_I_LIMIT_PWM / STEER_KI;
    steerIntegral = constrain(steerIntegral, -iLimit, iLimit);
  } else {
    steerIntegral = 0.0f;  // far from target (or KI off): no integral action
  }

  float deriv = (dt_s > 0.0f) ? (err - lastPotErr) / dt_s : 0.0f;
  lastPotErr = err;

  float u = STEER_KP * err + STEER_KI * steerIntegral + STEER_KD * deriv;
  int dir = (u >= 0) ? 1 : -1;
  int pwm = (int)fabs(u) + STEER_FF_PWM;  // FF_PWM is 0 now; kept as optional stiction breakaway
  pwm = constrain(pwm, 0, STEER_MAX_PWM);
  g_ctrl = dir * pwm;  // signed control effort for the plot

  driveSteer(dir, pwm);
}

void plotStatus() {
  // Serial Plotter format: "label:value" pairs, space-separated, one per line.
  // target & pot share the pot-count scale (watch them converge); err -> 0;
  // ctrl is signed drive effort. Toggle series in the IDE legend to zoom.
  Serial.print("target:"); Serial.print(g_potTarget);
  Serial.print(" pot:");   Serial.print(g_pot);
  Serial.print(" err:");   Serial.print(g_err);
  Serial.print(" ctrl:");  Serial.println(g_ctrl);
}

void printStatus() {
  int pot = analogRead(STEER_POT);
  int potTarget = steerToPotTarget(targetSteer);
  Serial.print("STATUS pot=");
  Serial.print(pot);
  Serial.print(" target=");
  Serial.print(potTarget);
  Serial.print(" err=");
  Serial.print(potTarget - pot);
  Serial.print(" desired_steer=");
  Serial.print(desiredSteer, 1);
  Serial.print(" target_steer=");
  Serial.print(targetSteer, 1);
  Serial.print(" drive_pwm=");
  Serial.print(driveCmdPwm);
  Serial.print(" edges_left=");
  Serial.print(POT_LEFT);
  Serial.print(" edges_right=");
  Serial.println(POT_RIGHT);
}

void driveSteer(int dir, int pwm) {
  if (STEER_INVERTED) dir = -dir;
  if (dir > 0) {
    hBridgeDrive(STEER_IN1, STEER_IN2, pwm);
  } else {
    hBridgeDrive(STEER_IN1, STEER_IN2, -pwm);
  }
}

void applyDrive(int pwm) {
  applyOneDrive(DRIVE_L_IN1, DRIVE_L_IN2, pwm, DRIVE_L_INVERTED);
  applyOneDrive(DRIVE_R_IN1, DRIVE_R_IN2, pwm, DRIVE_R_INVERTED);
}

void applyOneDrive(int in1, int in2, int pwm, bool inverted) {
  if (inverted) pwm = -pwm;
  if (abs(pwm) <= DRIVE_DEADBAND_PWM) {
    hBridgeHold(in1, in2);
    return;
  }
  hBridgeDrive(in1, in2, pwm);
}

void hBridgeDrive(int in1, int in2, int pwm) {
  pwm = constrain(pwm, -255, 255);
  if (pwm > 0) {
    analogWrite(in1, pwm);
    analogWrite(in2, 0);
  } else if (pwm < 0) {
    analogWrite(in1, 0);
    analogWrite(in2, -pwm);
  } else {
    analogWrite(in1, 0);
    analogWrite(in2, 0);
  }
}

void hBridgeHold(int in1, int in2) {
  analogWrite(in1, 0);
  analogWrite(in2, 0);
}

void holdAll() {
  hBridgeHold(DRIVE_L_IN1, DRIVE_L_IN2);
  hBridgeHold(DRIVE_R_IN1, DRIVE_R_IN2);
  hBridgeHold(STEER_IN1, STEER_IN2);
}
