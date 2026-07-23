// CTRL_ZERO steering calibration tool for Arduino Mega 2560.
//
// Purpose:
//   Manually drive the steering motor left/right from the Serial Monitor and
//   read the potentiometer (A0) value at the mechanical limits, so you can set
//   POT_LEFT / POT_RIGHT in CTRL_ZERO_Controller.ino.
//
// Serial Monitor settings:
//   - Baud: 9600
//   - Line ending: "Newline" (each command is one line)
//
// Commands (send one letter per line):
//   a : steer toward one side (labeled LEFT) and hold
//   d : steer toward the other side (labeled RIGHT) and hold
//   s : stop / hold (motor off)
//   + : increase drive PWM (moves faster / more torque)
//   - : decrease drive PWM
//   r : reset the tracked min/max
//   ? : print status immediately
//
// It continuously prints:  pot=<adc> min=<adc> max=<adc> pwm=<n> dir=<...>
// When you drive into a mechanical stop the pot stops changing; a stall guard
// then auto-stops the motor and prints the edge value.  Note that value:
//   - the reading at the LEFT stop  -> POT_LEFT
//   - the reading at the RIGHT stop -> POT_RIGHT
// If a/d feel physically reversed, flip STEER_INVERTED below (or swap them).

// --- pins / convention (match CTRL_ZERO_Controller.ino) ---------------------
const int STEER_IN1 = 2;
const int STEER_IN2 = 3;
const int STEER_POT = A0;
const bool STEER_INVERTED = true;

// --- drive / print tuning ---------------------------------------------------
const int PWM_MIN = 40;
const int PWM_MAX = 255;
const int PWM_STEP = 15;
const int DEFAULT_PWM = 90;
const unsigned long PRINT_INTERVAL_MS = 150;

// Stall guard: if the pot barely moves while driving, we have hit the stop.
const int STALL_DELTA_CNT = 2;
const unsigned long STALL_MS = 700;

int drivePwm = DEFAULT_PWM;   // PWM magnitude
// dir sign matches CTRL_ZERO_Controller.ino's driveSteer: +1 raises the pot
// toward POT_LEFT (LEFT), -1 lowers it toward POT_RIGHT (RIGHT).
int dir = 0;                  // +1 = LEFT, -1 = RIGHT, 0 = hold

int potMin = 1023;
int potMax = 0;

unsigned long lastPrintMs = 0;
int stallRefPot = 0;
unsigned long stallSinceMs = 0;

char inputBuf[16];
uint8_t inputLen = 0;

void setup() {
  Serial.begin(9600);
  pinMode(STEER_IN1, OUTPUT);
  pinMode(STEER_IN2, OUTPUT);
  pinMode(STEER_POT, INPUT);
  hold();
  Serial.println("check_steering ready.");
  Serial.println("Commands: a=LEFT  d=RIGHT  s=stop  +/-=pwm  r=reset  ?=status");
  Serial.println("Serial Monitor: 9600 baud, line ending = Newline.");
}

void loop() {
  readSerial();

  int pot = analogRead(STEER_POT);
  if (pot < potMin) potMin = pot;
  if (pot > potMax) potMax = pot;

  applySteer(pot);

  unsigned long now = millis();
  if (now - lastPrintMs >= PRINT_INTERVAL_MS) {
    lastPrintMs = now;
    printStatus(pot);
  }
}

void readSerial() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (inputLen > 0) {
        handleCommand(inputBuf[0]);
        inputLen = 0;
      }
    } else if (inputLen < sizeof(inputBuf) - 1) {
      inputBuf[inputLen++] = c;
    }
  }
}

void handleCommand(char c) {
  switch (c) {
    case 'a': case 'A':
      startDrive(+1);  // LEFT: raise pot toward POT_LEFT
      break;
    case 'd': case 'D':
      startDrive(-1);  // RIGHT: lower pot toward POT_RIGHT
      break;
    case 's': case 'S':
      dir = 0;
      hold();
      Serial.println("stop");
      break;
    case '+': case '=':
      drivePwm = min(drivePwm + PWM_STEP, PWM_MAX);
      Serial.print("pwm="); Serial.println(drivePwm);
      break;
    case '-': case '_':
      drivePwm = max(drivePwm - PWM_STEP, PWM_MIN);
      Serial.print("pwm="); Serial.println(drivePwm);
      break;
    case 'r': case 'R':
      potMin = 1023;
      potMax = 0;
      Serial.println("min/max reset");
      break;
    case '?':
      printStatus(analogRead(STEER_POT));
      break;
    default:
      break;
  }
}

void startDrive(int newDir) {
  dir = newDir;
  stallRefPot = analogRead(STEER_POT);
  stallSinceMs = millis();
  Serial.print("drive "); Serial.println(dir > 0 ? "LEFT" : "RIGHT");
}

void applySteer(int pot) {
  if (dir == 0) {
    hold();
    return;
  }

  // Stall guard: hitting the mechanical stop = pot stops changing.
  unsigned long now = millis();
  if (abs(pot - stallRefPot) > STALL_DELTA_CNT) {
    stallRefPot = pot;
    stallSinceMs = now;
  } else if (now - stallSinceMs >= STALL_MS) {
    Serial.print("LIMIT reached, holding. edge pot=");
    Serial.println(pot);
    dir = 0;
    hold();
    return;
  }

  driveSteer(dir, drivePwm);
}

void driveSteer(int d, int pwm) {
  if (STEER_INVERTED) d = -d;
  hBridge(d > 0 ? pwm : -pwm);
}

void hBridge(int pwm) {
  pwm = constrain(pwm, -255, 255);
  if (pwm > 0) {
    analogWrite(STEER_IN1, pwm);
    analogWrite(STEER_IN2, 0);
  } else if (pwm < 0) {
    analogWrite(STEER_IN1, 0);
    analogWrite(STEER_IN2, -pwm);
  } else {
    hold();
  }
}

void hold() {
  analogWrite(STEER_IN1, 0);
  analogWrite(STEER_IN2, 0);
}

void printStatus(int pot) {
  Serial.print("pot=");
  Serial.print(pot);
  Serial.print(" min=");
  Serial.print(potMin);
  Serial.print(" max=");
  Serial.print(potMax);
  Serial.print(" pwm=");
  Serial.print(drivePwm);
  Serial.print(" dir=");
  Serial.println(dir > 0 ? "LEFT" : (dir < 0 ? "RIGHT" : "hold"));
}
