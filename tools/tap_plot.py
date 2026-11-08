"""
Live graph of the jaw-tap detector, for testing and tuning the clicker on its own.
Uses exactly the detector in handsfree/jaw_clicker.py, so tuning changes there show up here.
Close the window to stop.

    python tools/tap_plot.py             # graph + gesture labels, NO real clicks
    python tools/tap_plot.py --click     # also send real OS clicks (F8 toggles them)
    python tools/tap_plot.py --port COM10

Watch the blue signal: a tap should spike well above the red threshold line, while sitting
still, breathing or talking should stay below it. Adjust THRESHOLD (and friends) in
handsfree/jaw_clicker.py, then rerun.
"""

import argparse
import collections
import os
import sys
import threading
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pynput import keyboard
from pynput.mouse import Button, Controller

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from handsfree import jaw_clicker as jc   # noqa: E402

WINDOW = 3000          # ticks on screen (~1 ms each on a fast board, longer on a slow one)
Y_MAX = 150


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="Arduino COM port (default: auto-detect)")
    ap.add_argument("--click", action="store_true", help="send real OS clicks")
    a = ap.parse_args()

    port = a.port or jc.find_port()
    if not port:
        sys.exit("No Arduino found. Plug it in, or pass --port COMx.")
    print(f"Opening {port} ...")
    src = jc.SerialTicks(port)
    det = jc.TapDetector()

    mouse = Controller()
    st = {"click": a.click, "dragging": False, "label": "", "label_t": 0.0, "history": [], "n": 0}
    data = collections.deque([0.0] * WINDOW, maxlen=WINDOW)
    taps = collections.deque(maxlen=200)
    running = True

    def release():
        if st["dragging"]:
            mouse.release(Button.left)
            st["dragging"] = False

    def gesture(g):
        name = jc.GESTURES[g]
        if st["click"]:
            if st["dragging"]:
                if g == 1:
                    release()
                    name = "DROP"
            elif g == 1:
                mouse.click(Button.left, 1)
            elif g == 2:
                mouse.click(Button.left, 2)
            else:
                mouse.press(Button.left)
                st["dragging"] = True
                name = "DRAG"
        st["label"], st["label_t"] = name, time.time()
        st["history"] = (st["history"] + [name])[-5:]
        print(name, flush=True)

    def toggle():
        st["click"] = not st["click"]
        if not st["click"]:
            release()
        print(f"OS clicks {'ON' if st['click'] else 'OFF'}", flush=True)

    keyboard.GlobalHotKeys({"<f8>": toggle}).start()

    def reader():
        while running:
            for v in src.read():
                g = det.step(v, src.tick_ms)
                st["n"] += 1
                data.append(det.sig)
                if det.tap_now:
                    taps.append(st["n"])
                if g:
                    gesture(g)

    threading.Thread(target=reader, daemon=True).start()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    (line,) = ax.plot(range(WINDOW), list(data), lw=1.0, label="tap signal")
    ax.axhline(jc.THRESHOLD, color="red", ls="--", lw=1, label=f"threshold ({jc.THRESHOLD})")
    dots, = ax.plot([], [], "o", color="limegreen", ms=9, label="tap")
    ax.set_ylim(-Y_MAX * 0.03, Y_MAX)
    ax.set_xlim(0, WINDOW - 1)
    ax.set_xlabel("ticks (newest on right)")
    ax.set_ylabel("transient strength")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    label_txt = ax.text(0.98, 0.95, "", transform=ax.transAxes, ha="right", va="top",
                        fontsize=22, fontweight="bold", color="tab:green")
    hist_txt = ax.text(0.98, 0.78, "", transform=ax.transAxes, ha="right", va="top", fontsize=9)

    def update(_):
        line.set_ydata(list(data))
        start = st["n"] - WINDOW
        xs = [i - start for i in list(taps) if i >= start]
        dots.set_data(xs, [jc.THRESHOLD] * len(xs))
        mode = "DRAGGING" if st["dragging"] else ("OS clicks ON (F8)" if st["click"] else "OS clicks OFF (F8)")
        ax.set_title(f"{port}   taps: {det.taps}   raw {src.raw_rate / 1000:.0f}k readings/s   "
                     f"tick {src.tick_ms:.1f} ms   backlog {src.backlog} B   [{mode}]")
        label_txt.set_text(st["label"] if time.time() - st["label_t"] < 1.5 else "")
        hist_txt.set_text("\n".join(reversed(st["history"])))
        return line, dots

    anim = FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)  # noqa: F841
    try:
        plt.show()
    finally:
        running = False
        release()          # never leave the button held on exit
        src.close()


if __name__ == "__main__":
    main()
