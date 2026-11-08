"""
Raw check that the Arduino is streaming loudness readings. Ctrl+C to stop.

    python tools/sensor_stream.py            # auto-detect the port
    python tools/sensor_stream.py --port COM10

Prints a reading + bar about 20 times a second, plus the total readings/s. If it says no data,
the sketch (arduino/loudness/loudness.ino) isn't uploaded or another program has the port open
(close the Arduino Serial Monitor and any running main.py / tap_plot.py).
"""

import argparse
import os
import sys
import time

import serial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from handsfree.jaw_clicker import BAUD, find_port   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    a = ap.parse_args()
    port = a.port or find_port()
    if not port:
        sys.exit("No Arduino found. Plug it in, or pass --port COMx.")

    ser = serial.Serial(port, BAUD, timeout=0.1)
    time.sleep(2)  # the board resets when the port opens
    ser.reset_input_buffer()
    print(f"Listening on {port}... (Ctrl+C to stop)")

    buf, n, peak = b"", 0, 0
    t_print = t_rate = t_data = time.time()
    try:
        while True:
            buf += ser.read(max(1, ser.in_waiting))
            *lines, buf = buf.split(b"\n")
            for line in lines:
                try:
                    v = int(line)
                except ValueError:
                    continue
                n += 1
                peak = max(peak, v)
                t_data = time.time()
            now = time.time()
            if now - t_data > 3:
                print("No data for 3 s - is loudness.ino uploaded and the Serial Monitor closed?")
                t_data = now
            if now - t_print >= 0.05 and n:
                print(f"peak {peak:4d} |{'#' * (peak * 50 // 1023)}")
                peak, t_print = 0, now
            if now - t_rate >= 1.0:
                print(f"--- {n / (now - t_rate):.0f} readings/s")
                n, t_rate = 0, now
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()


if __name__ == "__main__":
    main()
