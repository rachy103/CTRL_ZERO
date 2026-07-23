// CTRL_ZERO Arduino motor controller for Arduino Mega 2560.
//
// Serial protocol from Python:
//   steer,speed\n
//   steer: -100..100, positive means right target angle
//   speed: -255..255, positive means forward
// Diagnostic query:
//   ?\n
//   prints current potentiometer count, target count, steer target, and drive PWM.

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
const int POT_LEFT = 573;
const int POT_RIGHT = 436;
const int POT_MIN_SAFE = 436;
const int POT_MAX_SAFE = 573;

const float STEER_KP = 2.2f;
const float STEER_KI = 0.0f;
const float STEER_KD = 0.35f;
const int STEER_FF_PWM = 55;
const int STEER_MAX_PWM = 255;
const int STEER_DEADBAND_CNT = 6;
const float STEER_SLEW_PER_MS = 0.9f;

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

void setup() {
  Serial.begin(9600);
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

void updateSteerPID(float dt) {
  int pot = analogRead(STEER_POT);
  int potTarget = steerToPotTarget(targetSteer);
  int err = potTarget - pot;

  if (abs(err) <= STEER_DEADBAND_CNT) {
    steerIntegral = 0.0f;
    lastPotErr = err;
    hBridgeHold(STEER_IN1, STEER_IN2);
    return;
  }

  steerIntegral += err * dt;
  steerIntegral = constrain(steerIntegral, -20000.0f, 20000.0f);
  float deriv = (err - lastPotErr) / dt;
  lastPotErr = err;

  float u = STEER_KP * err + STEER_KI * steerIntegral + STEER_KD * deriv;
  int dir = (u >= 0) ? 1 : -1;
  int pwm = (int)fabs(u) + STEER_FF_PWM;
  pwm = constrain(pwm, 0, STEER_MAX_PWM);

  driveSteer(dir, pwm);
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
