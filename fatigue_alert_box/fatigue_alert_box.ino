/*
  Driver Fatigue Alert Box - Arduino Uno

  Receives one character from the PC over USB serial at 9600 baud:
    '0' = Normal      -> green LED on
    '1' = Warning     -> yellow LED on
    '2' = Alarm       -> red LED on
    '3' = Fleet alert -> red LED flashing

  If no valid command arrives for 3 seconds, the yellow LED flashes
  to indicate that the PC connection may be lost.

  Wiring:
    D8  -> 220 ohm resistor -> green LED -> GND
    D9  -> 220 ohm resistor -> yellow LED -> GND
    D10 -> 220 ohm resistor -> red LED -> GND
*/

const int GREEN_PIN = 8;
const int YELLOW_PIN = 9;
const int RED_PIN = 10;

const unsigned long LINK_TIMEOUT_MS = 3000;
const unsigned long FAST_BLINK_MS = 120;
const unsigned long SLOW_BLINK_MS = 500;

char level = '0';
unsigned long lastRx = 0;
unsigned long lastBlink = 0;
bool blinkState = false;

void setLeds(bool green, bool yellow, bool red) {
  digitalWrite(GREEN_PIN, green ? HIGH : LOW);
  digitalWrite(YELLOW_PIN, yellow ? HIGH : LOW);
  digitalWrite(RED_PIN, red ? HIGH : LOW);
}

void setup() {
  Serial.begin(9600);

  pinMode(GREEN_PIN, OUTPUT);
  pinMode(YELLOW_PIN, OUTPUT);
  pinMode(RED_PIN, OUTPUT);

  // Startup self-test: green -> yellow -> red.
  setLeds(true, false, false);
  delay(300);
  setLeds(false, true, false);
  delay(300);
  setLeds(false, false, true);
  delay(300);
  setLeds(false, false, false);

  lastRx = millis();
}

void loop() {
  while (Serial.available() > 0) {
    char command = Serial.read();

    if (command >= '0' && command <= '3') {
      level = command;
      lastRx = millis();
    }
  }

  unsigned long now = millis();

  if (now - lastRx > LINK_TIMEOUT_MS) {
    if (now - lastBlink >= SLOW_BLINK_MS) {
      lastBlink = now;
      blinkState = !blinkState;
    }

    setLeds(false, blinkState, false);
    return;
  }

  switch (level) {
    case '0':
      setLeds(true, false, false);
      break;

    case '1':
      setLeds(false, true, false);
      break;

    case '2':
      setLeds(false, false, true);
      break;

    case '3':
      if (now - lastBlink >= FAST_BLINK_MS) {
        lastBlink = now;
        blinkState = !blinkState;
      }

      setLeds(false, false, blinkState);
      break;
  }
}
