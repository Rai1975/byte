"""
Jaw-tap detection from the Grove loudness sensor on the Arduino (arduino/loudness/loudness.ino).

This module only detects taps and groups them into gestures (1, 2 or 3 taps). What a gesture
does (click, double click, action menu...) lives in actions.py; app.py wires the two together
and blocks everything while the user is talking.

Detector (per 1 ms tick, adapted from the original sensor/plot.py):
  1. dev  = |tick - baseline|           baseline = very slow average (removes DC / drift)
  2. fast = quick envelope of dev       follows sharp transients (a tooth tap)
     slow = slow envelope of dev        follows sustained sound (humming, voice)
  3. signal = fast - SLOW_GAIN * slow   sustained sound cancels itself; a tap jumps `fast` first
  4. tap when signal > THRESHOLD while armed; re-arm once the signal falls below REARM_LEVEL.
     Taps closer together than CLICK_GAP_MS form one gesture.

The sketch streams readings as fast as it can (tens of thousands per second on one laptop,
a few hundred on another), so raw readings are averaged into ~1 ms ticks and every tick carries
its real length. The tuning below is therefore in real milliseconds on any machine, and Python
never falls behind the board.
"""

import threading
import time
from collections import deque

import serial
import serial.tools.list_ports

BAUD = 115200            # must match Serial.begin() in loudness.ino
ARDUINO_VID = 0x2341
TICK_MS = 1.0            # target tick length; slower boards give longer ticks (handled)

# --- tuning knobs (alphas are per 1 ms; times in ms). Tune live with: python tools/tap_plot.py
BASE_ALPHA = 0.002       # baseline speed (~0.5 s time constant)
FAST_ALPHA = 0.3         # fast envelope rise (higher = reacts to shorter transients)
FAST_RELEASE = 0.7       # fast envelope fall (higher = spike drops off faster)
SLOW_ALPHA = 0.005       # slow envelope (~200 ms time constant); tracks humming / voice
SLOW_GAIN = 2.0          # how much of the slow envelope to subtract
THRESHOLD = 15           # tap trigger level, ADC counts. RAISE if false taps, LOWER if missed
REARM_LEVEL = 4          # signal must fall to this before the next tap can register
REFRACTORY_MS = 0        # optional extra dead time after a tap
TAP_HOLD_MS = 50         # slow envelope is frozen this long after a tap
CLICK_GAP_MS = 250       # taps closer together than this belong to the same gesture
FLOOR = 0                # signal never goes below this
# ------------------------------------------------------------------------------------------

GESTURES = {1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE"}


def find_port():
    """The Arduino's COM port differs per laptop (COM5 on one, COM10 on another)."""
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if p.vid == ARDUINO_VID or "arduino" in (p.description or "").lower():
            return p.device
    for p in ports:
        if "usb" in (p.description or "").lower():
            return p.device
    return None


def _per(alpha, dt_ms):
    """Per-1ms smoothing factor -> factor for a tick of dt_ms (same time constant)."""
    return alpha if dt_ms == 1.0 else 1.0 - (1.0 - alpha) ** dt_ms


class TapDetector:
    """Feed one tick at a time; step() returns the finished gesture (1/2/3) or 0."""

    def __init__(self):
        self.baseline = None
        self.fast = self.slow = 0.0
        self.armed = True
        self.n = 0
        self.t = 0.0               # ms since start (sum of tick lengths)
        self.last_tap = -1e9       # ms
        self.group = 0
        self.sig = 0.0
        self.taps = 0
        self.tap_now = False   # True on the tick a tap fired (for plots)

    def step(self, v, dt_ms=TICK_MS):
        self.baseline = v if self.baseline is None else self.baseline + _per(BASE_ALPHA, dt_ms) * (v - self.baseline)
        dev = abs(v - self.baseline)
        a = FAST_ALPHA if dev > self.fast else FAST_RELEASE
        self.fast += _per(a, dt_ms) * (dev - self.fast)
        self.n += 1
        self.t += dt_ms
        since = self.t - self.last_tap
        if since > TAP_HOLD_MS:     # taps must not teach the hum filter to ignore the next tap
            self.slow += _per(SLOW_ALPHA, dt_ms) * (dev - self.slow)
        self.sig = max(FLOOR, self.fast - SLOW_GAIN * self.slow)

        self.tap_now = False
        if self.sig <= REARM_LEVEL:
            self.armed = True
        if self.armed and self.sig > THRESHOLD and since > REFRACTORY_MS:
            self.armed = False
            self.last_tap = self.t
            self.taps += 1
            self.tap_now = True
            self.group += 1
            if self.group >= 3:     # triple is the max: report right away
                self.group = 0
                return 3
        elif self.group and since > CLICK_GAP_MS:
            g, self.group = self.group, 0
            return g
        return 0


class SerialTicks:
    """Reads the sensor in bulk and turns raw readings into 1 ms ticks (block averages)."""

    def __init__(self, port):
        self.ser = serial.Serial(port, BAUD, timeout=0.05)
        time.sleep(2)               # the board resets when the port opens
        self.ser.reset_input_buffer()
        self.buf = b""
        self.raw_rate = 0.0         # raw readings per second (measured)
        self._rate_t, self._rate_n = time.time(), 0
        self._block = []
        self.per_tick = None        # raw readings per tick, set after the first rate measurement
        self.tick_ms = TICK_MS      # real length of each returned tick

    def read(self):
        """Return the ticks completed since the last call (may be empty)."""
        self.buf += self.ser.read(max(1, self.ser.in_waiting))
        *lines, self.buf = self.buf.split(b"\n")
        vals = []
        for line in lines:
            try:
                vals.append(int(line))
            except ValueError:
                pass
        now = time.time()
        self._rate_n += len(vals)
        if now - self._rate_t >= 1.0:
            self.raw_rate = self._rate_n / (now - self._rate_t)
            self._rate_t, self._rate_n = now, 0
            if self.raw_rate > 0:
                self.per_tick = max(1, round(self.raw_rate * TICK_MS / 1000))
                self.tick_ms = self.per_tick * 1000.0 / self.raw_rate
        if self.per_tick is None:   # first second: just measure the rate
            return []
        ticks = []
        self._block.extend(vals)
        k = self.per_tick
        full = len(self._block) // k * k
        for i in range(0, full, k):
            ticks.append(sum(self._block[i:i + k]) / k)
        del self._block[:full]
        return ticks

    @property
    def backlog(self):
        return self.ser.in_waiting

    def close(self):
        self.ser.close()


class JawClicker(threading.Thread):
    """Background thread: sensor -> TapDetector -> on_event(kind, gesture, wall_time).

    kind "tap":     a single tap was detected (gesture = taps so far in this group)
    kind "gesture": the group finished (gesture = 1, 2 or 3)
    """

    def __init__(self, port=None, on_event=None):
        super().__init__(daemon=True)
        self.port_arg = port
        self.port = None
        self.on_event = on_event or (lambda *e: None)
        self.alive = True
        self.status = "starting"
        self.det = TapDetector()
        self.raw_rate = 0.0
        self.sig_trail = deque(maxlen=600)   # recent signal, for the HUD sparkline

    def run(self):
        while self.alive:
            self.port = self.port_arg or find_port()
            if not self.port:
                self.status = "no Arduino found (plug it in, or --port COMx)"
                time.sleep(2)
                continue
            try:
                src = SerialTicks(self.port)
            except serial.SerialException as e:
                self.status = f"{self.port}: {e.__class__.__name__} (in use by another program?)"
                time.sleep(2)
                continue
            self.status = "running"
            try:
                self._loop(src)
            except serial.SerialException:
                self.status = f"{self.port} disconnected"
            finally:
                src.close()
            time.sleep(1)

    def _loop(self, src):
        while self.alive:
            ticks = src.read()
            self.raw_rate = src.raw_rate
            peak = 0.0
            for v in ticks:
                g = self.det.step(v, src.tick_ms)
                peak = max(peak, self.det.sig)
                if self.det.tap_now:
                    self.on_event("tap", self.det.group or 3, time.time())
                if g:
                    self.on_event("gesture", g, time.time())
            if ticks:
                self.sig_trail.append(peak)

    def close(self):
        self.alive = False
