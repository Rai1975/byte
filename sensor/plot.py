# pip install pyserial matplotlib pynput
# Live tap detector for the Grove loudness sensor on the Uno Q (teeth tapping via jaw).
# Close the window to stop.
#
# Pipeline (per sample):
#   1. dev  = |raw - baseline|            baseline = very slow average (removes DC / drift)
#   2. fast = quick average of dev        follows sharp transients
#      slow = slow average of dev         follows sustained sounds (humming, voice)
#   3. signal = fast - SLOW_GAIN * slow   sustained hum raises `slow`, so it cancels itself;
#                                         a tap jumps `fast` before `slow` can react
#   4. tap when signal > THRESHOLD while armed; after a tap it re-arms only once the signal
#      has fallen back to REARM_LEVEL (near baseline), so a new tap can fire as soon as
#      the previous spike is over, however close together they are
import collections
import threading
import time

import matplotlib.pyplot as plt
import serial
from pynput import keyboard
from pynput.mouse import Button, Controller
from matplotlib.animation import FuncAnimation

PORT = 'COM5'
BAUD = 115200          # must match Serial.begin() in loudness.ino

# --- tuning knobs ---------------------------------------------------------
WINDOW = 1500          # samples on screen (~1.5-3 s depending on sample rate)
BASE_ALPHA = 0.002     # baseline speed (lower = slower, ignores more drift)
FAST_ALPHA = 0.3       # fast envelope (higher = reacts to shorter transients)
FAST_RELEASE = 0.7     # how fast the spike falls (higher = drops off faster; 0.3 = old behavior)
SLOW_ALPHA = 0.005     # slow envelope (~200 sample time constant); tracks humming
SLOW_GAIN = 2.0        # how much of the slow envelope to subtract (higher = hum rejected harder)
THRESHOLD = 15         # tap trigger level, in ADC counts. RAISE if false taps, LOWER if missed
REARM_LEVEL = 4        # after a tap, signal must drop to this (near baseline/floor) before the next tap can register
REFRACTORY = 0         # optional extra SAMPLES to ignore after a tap (0 = rely on re-arm only)
TAP_HOLD = 50          # SAMPLES after a tap during which the slow (hum) envelope is frozen
CLICK_GAP = 33        # SAMPLES: taps closer together than this belong to the same click group
FLOOR = 0              # signal never goes below this (stops the dip after each tap)
Y_MAX = 150            # y-axis top
# --------------------------------------------------------------------------

# --- OS mouse control -----------------------------------------------------
# single  -> left click            double -> double click
# triple  -> start drag (hold left button); next single -> drop (release)
# Clicks land wherever the cursor currently is. F8 turns OS clicking on/off.
mouse = Controller()
os_state = {'enabled': True, 'dragging': False}


def do_gesture(name):
    if not os_state['enabled']:
        return
    if os_state['dragging']:
        if name == 'SINGLE click':
            mouse.release(Button.left)
            os_state['dragging'] = False
            print('  -> DROP (drag off)', flush=True)
        return  # while dragging, only a single click does anything
    if name == 'SINGLE click':
        mouse.click(Button.left, 1)
    elif name == 'DOUBLE click':
        mouse.click(Button.left, 2)
    elif name == 'TRIPLE click':
        mouse.press(Button.left)
        os_state['dragging'] = True
        print('  -> DRAG ON (single click to drop)', flush=True)


def toggle_enabled():
    os_state['enabled'] = not os_state['enabled']
    if not os_state['enabled'] and os_state['dragging']:
        mouse.release(Button.left)
        os_state['dragging'] = False
    print(f"OS clicking {'ON' if os_state['enabled'] else 'OFF'}", flush=True)


keyboard.GlobalHotKeys({'<f8>': toggle_enabled}).start()
# --------------------------------------------------------------------------

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)  # the board resets when the port opens
ser.reset_input_buffer()

data = collections.deque([0.0] * WINDOW, maxlen=WINDOW)
taps = collections.deque(maxlen=200)   # absolute sample indices of detected taps
state = {'n': 0, 'count': 0, 'rate': 0.0, 'label': '-', 'label_t': 0.0, 'history': []}
CLICK_NAMES = {1: 'SINGLE click', 2: 'DOUBLE click', 3: 'TRIPLE click'}

running = True


def reader():
    baseline = None
    fast = slow = 0.0
    armed = True
    last_tap = -10**9   # sample index of the last tap
    group = 0  # taps in the current click group

    def finish_group():
        nonlocal group
        name = CLICK_NAMES[min(group, 3)]
        state['label'], state['label_t'] = name, time.time()
        state['history'] = (state['history'] + [name])[-5:]
        print(name, flush=True)
        do_gesture(name)
        group = 0
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
        fast += (FAST_ALPHA if dev > fast else FAST_RELEASE) * (dev - fast)
        n = state['n'] + 1   # index of this sample; all timing below counts samples, not seconds
        if n - last_tap > TAP_HOLD:   # taps must not teach the hum filter to ignore the next tap
            slow += SLOW_ALPHA * (dev - slow)
        sig = max(FLOOR, fast - SLOW_GAIN * slow)  # no dip below the floor after a tap

        state['n'] = n
        now = time.time()    # only used for the samples/s readout
        data.append(sig)
        if sig <= REARM_LEVEL:
            armed = True            # previous spike has come back down
        if armed and sig > THRESHOLD and n - last_tap > REFRACTORY:
            armed = False
            last_tap = n
            state['count'] += 1
            taps.append(state['n'])
            group += 1
            if group >= 3:          # triple is the max: report right away
                finish_group()
        elif group and n - last_tap > CLICK_GAP:
            finish_group()          # gap expired: report what we got (1 or 2)

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
label_txt = ax.text(0.98, 0.95, '', transform=ax.transAxes, ha='right', va='top',
                    fontsize=22, fontweight='bold', color='tab:green')
hist_txt = ax.text(0.98, 0.78, '', transform=ax.transAxes, ha='right', va='top', fontsize=9)


def update(_):
    vals = list(data)
    line.set_ydata(vals)
    start = state['n'] - WINDOW
    xs = [i - start for i in list(taps) if i >= start]
    dots.set_data(xs, [THRESHOLD] * len(xs))
    mode = 'DRAGGING' if os_state['dragging'] else ('mouse ON' if os_state['enabled'] else 'mouse OFF (F8)')
    ax.set_title(f"Taps: {state['count']}   ({state['rate']:.0f} samples/s)   [{mode}]")
    fresh = time.time() - state['label_t'] < 1.5   # show the latest result for 1.5 s
    label_txt.set_text(state['label'] if fresh else '')
    hist_txt.set_text(chr(10).join(reversed(state['history'])))
    return line, dots


anim = FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)
try:
    plt.show()
finally:
    running = False
    if os_state['dragging']:
        mouse.release(Button.left)   # never leave the button held on exit
    ser.close()
