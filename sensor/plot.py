# pip install pyserial matplotlib
# Live tap detector for the Grove loudness sensor on the Uno Q (teeth tapping via jaw).
# Close the window to stop.
#
# Pipeline (per sample):
#   1. dev  = |raw - baseline|            baseline = very slow average (removes DC / drift)
#   2. fast = quick average of dev        follows sharp transients
#      slow = slow average of dev         follows sustained sounds (humming, voice)
#   3. signal = fast - SLOW_GAIN * slow   sustained hum raises `slow`, so it cancels itself;
#                                         a tap jumps `fast` before `slow` can react
#   4. tap when signal > THRESHOLD and REFRACTORY seconds passed since the last tap
import collections
import threading
import time

import matplotlib.pyplot as plt
import serial
from matplotlib.animation import FuncAnimation

PORT = 'COM5'
BAUD = 115200          # must match Serial.begin() in loudness.ino

# --- tuning knobs ---------------------------------------------------------
WINDOW = 1500          # samples on screen (~1.5-3 s depending on sample rate)
BASE_ALPHA = 0.002     # baseline speed (lower = slower, ignores more drift)
FAST_ALPHA = 0.3       # fast envelope (higher = reacts to shorter transients)
SLOW_ALPHA = 0.005     # slow envelope (~200 sample time constant); tracks humming
SLOW_GAIN = 2.0        # how much of the slow envelope to subtract (higher = hum rejected harder)
THRESHOLD = 15         # tap trigger level, in ADC counts. RAISE if false taps, LOWER if missed
REFRACTORY = 0.08      # seconds to ignore after a tap
FLOOR = 0              # signal never goes below this (stops the dip after each tap)
Y_MAX = 150            # y-axis top
# --------------------------------------------------------------------------

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)  # the board resets when the port opens
ser.reset_input_buffer()

data = collections.deque([0.0] * WINDOW, maxlen=WINDOW)
taps = collections.deque(maxlen=200)   # absolute sample indices of detected taps
state = {'n': 0, 'count': 0, 'rate': 0.0}
running = True


def reader():
    baseline = None
    fast = slow = 0.0
    last_tap = 0.0
    rate_t, rate_n = time.time(), 0
    while running:
        try:
            line = ser.readline().decode(errors='ignore').strip()
            if not line:
                continue
            v = int(line)
        except (ValueError, serial.SerialException):
            continue

        baseline = v if baseline is None else baseline + BASE_ALPHA * (v - baseline)
        dev = abs(v - baseline)
        fast += FAST_ALPHA * (dev - fast)
        slow += SLOW_ALPHA * (dev - slow)
        sig = max(FLOOR, fast - SLOW_GAIN * slow)  # no dip below the floor after a tap

        now = time.time()
        state['n'] += 1
        data.append(sig)
        if sig > THRESHOLD and now - last_tap > REFRACTORY:
            last_tap = now
            state['count'] += 1
            taps.append(state['n'])

        rate_n += 1
        if now - rate_t >= 1.0:
            state['rate'] = rate_n / (now - rate_t)
            rate_t, rate_n = now, 0


threading.Thread(target=reader, daemon=True).start()

fig, ax = plt.subplots()
(line,) = ax.plot(range(WINDOW), list(data), lw=1.2, label='tap signal')
ax.axhline(THRESHOLD, color='red', ls='--', lw=1, label=f'threshold ({THRESHOLD})')
dots, = ax.plot([], [], 'o', color='limegreen', ms=9, label='tap')
ax.set_ylim(FLOOR - Y_MAX * 0.03, Y_MAX)
ax.set_xlim(0, WINDOW - 1)
ax.set_xlabel('samples (newest on right)')
ax.set_ylabel('transient strength')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)


def update(_):
    vals = list(data)
    line.set_ydata(vals)
    start = state['n'] - WINDOW
    xs = [i - start for i in list(taps) if i >= start]
    dots.set_data(xs, [THRESHOLD] * len(xs))
    ax.set_title(f"Taps: {state['count']}   ({state['rate']:.0f} samples/s)")
    return line, dots


anim = FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)
try:
    plt.show()
finally:
    running = False
    ser.close()
