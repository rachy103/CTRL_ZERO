// CTRL_ZERO Arduino motor controller.
//
// Serial protocol from Python:
//   steer,speed\n
//   steer: -100..100, positive means right
//   speed: -DRIVE_MAX_PWM..DRIVE_MAX_PWM, positive means forward

const int MOTOR1_IN1 = 2;
const int MOTOR1_IN2 = 3;
const int MOTOR2_IN1 = 4;
const int MOTOR2_IN2 = 5;
const int STEER_IN1 = 6;
const int STEER_IN2 = 7;

const bool MOTOR1_INVERTED = true;
const bool MOTOR2_INVERTED = true;
const bool STEER_INVERTED = false;

const int DRIVE_MAX_PWM = 160;
const int STEER_MAX_PWM = 180;
const int STEER_MIN_PWM = 140;
const int DRIVE_DEADBAND_PWM = 8;
const int STEER_DEADBAND = 8;
const unsigned long COMMAND_TIMEOUT_MS = 350;

String inputLine = "";
unsigned long lastCommandMs = 0;

void setup() {
  Serial.begin(9600);
  pinMode(MOTOR1_IN1, OUTPUT);
  pinMode(MOTOR1_IN2, OUTPUT);
  pinMode(MOTOR2_IN1, OUTPUT);
  pinMode(MOTOR2_IN2, OUTPUT);
  pinMode(STEER_IN1, OUTPUT);
  pinMode(STEER_IN2, OUTPUT);
  holdAll();
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      parseCommand(inputLine);
      inputLine = "";
    } else if (c != '\r' && inputLine.length() < 32) {
      inputLine += c;
    }
  }

  if (millis() - lastCommandMs > COMMAND_TIMEOUT_MS) {
    holdAll();
  }
}

void parseCommand(String line) {
  int comma = line.indexOf(',');
  if (comma < 0) {
    return;
  }

  int steer = constrain(line.substring(0, comma).toInt(), -100, 100);
  int speed = constrain(line.substring(comma + 1).toInt(), -DRIVE_MAX_PWM, DRIVE_MAX_PWM);

  setDriveMotor(MOTOR1_IN1, MOTOR1_IN2, speed, MOTOR1_INVERTED);
  setDriveMotor(MOTOR2_IN1, MOTOR2_IN2, speed, MOTOR2_INVERTED);
  setSteerMotor(steer);
  lastCommandMs = millis();
}

void setDriveMotor(int in1, int in2, int pwm, bool inverted) {
  if (inverted) {
    pwm = -pwm;
  }

  if (pwm > DRIVE_DEADBAND_PWM) {
    hBridgeForward(in1, in2, pwm);
  } else if (pwm < -DRIVE_DEADBAND_PWM) {
    hBridgeBackward(in1, in2, -pwm);
  } else {
    hBridgeHold(in1, in2);
  }
}

void setSteerMotor(int steer) {
  if (STEER_INVERTED) {
    steer = -steer;
  }

  if (abs(steer) <= STEER_DEADBAND) {
    hBridgeHold(STEER_IN1, STEER_IN2);
    return;
  }

  int steerPwm = map(abs(steer), STEER_DEADBAND, 100, STEER_MIN_PWM, STEER_MAX_PWM);
  steerPwm = constrain(steerPwm, STEER_MIN_PWM, STEER_MAX_PWM);

  if (steer > 0) {
    hBridgeForward(STEER_IN1, STEER_IN2, steerPwm);
  } else {
    hBridgeBackward(STEER_IN1, STEER_IN2, steerPwm);
  }
}

void hBridgeForward(int in1, int in2, int pwm) {
  analogWrite(in1, constrain(pwm, 0, 255));
  digitalWrite(in2, LOW);
}

void hBridgeBackward(int in1, int in2, int pwm) {
  digitalWrite(in1, LOW);
  analogWrite(in2, constrain(pwm, 0, 255));
}

void hBridgeHold(int in1, int in2) {
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
}

void holdAll() {
  hBridgeHold(MOTOR1_IN1, MOTOR1_IN2);
  hBridgeHold(MOTOR2_IN1, MOTOR2_IN2);
  hBridgeHold(STEER_IN1, STEER_IN2);
}
