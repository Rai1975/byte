# pip install pyserial pynput
# Headless version of plot.py: no graph, no console output. Taps -> real OS mouse clicks.
#   single -> left click      double -> double click
#   triple -> start drag (hold left button); next single -> drop (release)
# Clicks land wherever the cursor is.  F8 = clicking on/off,  F9 = quit.
# Run silently with:  pythonw sensor/clicker.py   (or python sensor/clicker.py)
# Keep tuning values in sync with plot.py (the debug/visual version).
import time

import serial
from pynput import keyboard
from pynput.mouse import Button, Controller

PORT = 'COM5'
BAUD = 115200          # must match Serial.begin() in loudness.ino

# --- tuning knobs (same meaning as plot.py) --------------------------------
BASE_ALPHA = 0.002
FAST_ALPHA = 0.3
FAST_RELEASE = 0.7
SLOW_ALPHA = 0.005
SLOW_GAIN = 2.0
THRESHOLD = 15
REARM_LEVEL = 4
REFRACTORY = 0         # SAMPLES
TAP_HOLD = 50          # SAMPLES
CLICK_GAP = 33         # SAMPLES
FLOOR = 0
# --------------------------------------------------------------------------

mouse = Controller()
state = {'enabled': True, 'dragging': False, 'running': True}


def release_drag():
    if state['dragging']:
        mouse.release(Button.left)
        state['dragging'] = False


def do_gesture(group):
    if not state['enabled']:
        return
    if state['dragging']:
        if group == 1:      # only a single click ends a drag
            release_drag()
        return
    if group == 1:
        mouse.click(Button.left, 1)
    elif group == 2:
        mouse.click(Button.left, 2)
    else:                   # triple
        mouse.press(Button.left)
        state['dragging'] = True


def toggle_enabled():
    state['enabled'] = not state['enabled']
    if not state['enabled']:
        release_drag()


def quit_app():
    state['running'] = False


keyboard.GlobalHotKeys({'<f8>': toggle_enabled, '<f9>': quit_app}).start()


def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # the board resets when the port opens
    ser.reset_input_buffer()

    baseline = None
    fast = slow = 0.0
    armed = True
    n = 0
    last_tap = -10**9
    group = 0

    try:
        while state['running']:
            try:
                v = int(ser.readline().decode(errors='ignore').strip())
            except ValueError:
                continue
            except serial.SerialException:
                time.sleep(0.5)
                continue

            baseline = v if baseline is None else baseline + BASE_ALPHA * (v - baseline)
            dev = abs(v - baseline)
            fast += (FAST_ALPHA if dev > fast else FAST_RELEASE) * (dev - fast)
            n += 1
            if n - last_tap > TAP_HOLD:
                slow += SLOW_ALPHA * (dev - slow)
            sig = max(FLOOR, fast - SLOW_GAIN * slow)

            if sig <= REARM_LEVEL:
                armed = True
            if armed and sig > THRESHOLD and n - last_tap > REFRACTORY:
                armed = False
                last_tap = n
                group += 1
                if group >= 3:
                    do_gesture(3)
                    group = 0
            elif group and n - last_tap > CLICK_GAP:
                do_gesture(group)
                group = 0
    finally:
        release_drag()   # never leave the button held on exit
        ser.close()


if __name__ == '__main__':
    main()
