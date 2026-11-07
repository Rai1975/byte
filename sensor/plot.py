# pip install pyserial matplotlib
# Live graph of Grove loudness sensor readings from the Uno Q. Close the window to stop.
import collections
import threading
import time

import matplotlib.pyplot as plt
import serial
from matplotlib.animation import FuncAnimation

PORT = 'COM5'
BAUD = 9600
WINDOW = 500  # samples shown (~10 s at 50 samples/sec)
# High-pass filter: removes slow/low-frequency drift and the DC offset so only
# fast spikes remain. Baseline is a one-pole low-pass (exponential average);
# filtered = raw - baseline. Smaller ALPHA = slower baseline = lower cutoff.
# Cutoff ~= ALPHA * 50 Hz / (2*pi): 0.02 -> ~0.16 Hz, 0.1 -> ~0.8 Hz.
ALPHA = 0.05
Y_RANGE = 300  # +/- display range of the filtered signal

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)  # the board resets when the port opens
ser.reset_input_buffer()

data = collections.deque([0] * WINDOW, maxlen=WINDOW)
running = True
baseline = None


def reader():
    global baseline
    while running:
        try:
            line = ser.readline().decode(errors='ignore').strip()
            if line:
                v = int(line)
                baseline = v if baseline is None else baseline + ALPHA * (v - baseline)
                data.append(v - baseline)
        except (ValueError, serial.SerialException):
            continue


threading.Thread(target=reader, daemon=True).start()

fig, ax = plt.subplots()
(line,) = ax.plot(range(WINDOW), list(data), lw=1.2)
avg_line = ax.axhline(0, color='orange', ls='--', lw=1, label='baseline (0)')
ax.set_ylim(-Y_RANGE, Y_RANGE)
ax.set_xlim(0, WINDOW - 1)
ax.set_xlabel('samples (newest on right)')
ax.set_ylabel('spike size (raw - baseline)')
ax.set_title(f'Loudness sensor on {PORT}')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)


def update(_):
    vals = list(data)
    line.set_ydata(vals)
    ax.set_title(f'Loudness spikes on {PORT} - now {vals[-1]:+.0f}')
    return line, avg_line


anim = FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)
try:
    plt.show()
finally:
    running = False
    ser.close()
