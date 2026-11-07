// Grove Loudness Sensor -> serial stream (one reading per line)
// The Grove loudness sensor outputs an ANALOG voltage on SIG.
// D2 is NOT an analog-capable pin on the Uno form factor, so analogRead(D2)
// won't work. Plug the sensor into an analog port (A0..A3) and set the pin here.
#define SENSOR_PIN A0
// Set to 1 only if you are stuck with D2: reads 0/1 instead of a level.
#define USE_DIGITAL 0

void setup() {
  Serial.begin(115200);  // must match BAUD in plot.py / test.py
  pinMode(SENSOR_PIN, INPUT);
}

void loop() {
#if USE_DIGITAL
  int v = digitalRead(2);
#else
  int v = analogRead(SENSOR_PIN);
#endif
  Serial.println(v);
  delay(1);  // up to ~1000 samples/sec (fast enough to catch tap transients)
}
