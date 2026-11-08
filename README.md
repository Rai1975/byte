# Hands-Free Mouse

Control your computer's mouse without your hands:

- **Move the cursor with your eyes.** A webcam tracks where you look.
- **Click by tapping your teeth.** A loudness sensor on your jaw, read by an Arduino, picks up the taps. One tap clicks, two double-click, and three open an action menu with right click, drag/select, scroll and zoom.
- **The cursor snaps onto nearby buttons and links**, so small targets are easier to hit.
- **Talking pauses everything.** While you speak, the cursor freezes and taps don't click, so you can talk or dictate without the mouse going off on its own. Presage SmartSpectra detects the talking, and a lip-movement detector takes over if Presage isn't available.

Everything runs as one program: `python main.py`.

## Repo layout

```text
main.py                    start the app (eye cursor + jaw clicks + talk pause)
handsfree/
  app.py                   eye tracking, cursor driver, calibration, debug window; wires it all up
  jaw_clicker.py           jaw-tap detector (reads the Arduino over USB serial)
  actions.py               what taps do: click, double click, action menu, drag, scroll, zoom
  snap.py                  snapping to nearby buttons/links (Windows UI Automation)
  overlay.py               transparent on-screen layer: menu, snap highlight, zoom view
  talk_detect.py           talking detection (Presage + lip-movement fallback)
presage/talk_bridge.mjs    Node bridge to the Presage SDK (the SDK is Node-only)
arduino/loudness/          Arduino sketch that streams the loudness sensor
tools/
  sensor_stream.py         check the Arduino is sending data
  tap_plot.py              live graph of the tap detector, for testing and tuning
models/face_landmarker.task  MediaPipe face model
environment.yml            conda environment (makeuc-eye)
```

## 1. Setup (one time)

You need [Miniconda](https://docs.conda.io/en/latest/miniconda.html), and also [Node.js](https://nodejs.org) 20 or newer if you want Presage. Run these from the repo root:

```bash
conda env create -f environment.yml        # creates the "makeuc-eye" env
conda activate makeuc-eye
```

If `environment.yml` changes later, update the env with `conda env update -f environment.yml --prune`.

**Presage (optional, recommended).** Install the SDK, then put your API key from [physiology.presagetech.com](https://physiology.presagetech.com) in a `.env` file in the repo root. Git ignores this file.

```bash
cd presage && npm install && cd ..
```
```text
PRESAGE_API_KEY=your-key-here
```

Without a key, the app falls back to the lip-movement detector automatically.

**Arduino (for clicks).**
1. Plug the Grove loudness sensor into **A0** on the Arduino.
2. Upload `arduino/loudness/loudness.ino` (Arduino IDE or App Lab).
3. Close the Serial Monitor and App Lab afterwards. Only one program can use the port at a time.

The app finds the Arduino's COM port automatically. Pass `--port COMx` to choose one yourself.

Every command below assumes `conda activate makeuc-eye` and the repo root as the working directory.

## 2. Test each part on its own

### a) Eye cursor (no Arduino needed)

```bash
python main.py --no-clicks
```

1. A debug window opens. Check that your eyes show cyan outlines with green iris rings. If it says "NO FACE", fix your lighting or camera angle.
2. Press **F8** to calibrate. Sit how you normally will, keep your head still, press SPACE, and follow the 13 dots with your eyes.
3. An accuracy check (9 dots) runs straight after, then cursor control turns on. Under 3° of error is good for a webcam.
4. **F7** turns the cursor on and off. **F10** or `q` quits.

The calibration is saved to `calibration.json` and reloaded next time. Recalibrate after you move the laptop or chair, or when the lighting changes.

### b) Arduino sensor

```bash
python tools/sensor_stream.py
```

You should see `Listening on COMx`, a bar that jumps when you tap your teeth, and a `readings/s` line every second.

| Problem | Fix |
|---------|-----|
| `No Arduino found` | Check the USB cable, or pass `--port COMx`. |
| `No data` or `Access denied` | Close App Lab, the Serial Monitor, and any other running `main.py` or `tap_plot.py`. |

### c) Jaw-tap detection

```bash
python tools/tap_plot.py              # graph only: no real clicks
```

1. With the sensor on your jaw, **sit still and quiet for 10 s**. The blue signal should stay below the red threshold line, with no green tap dots.
2. Then try each gesture:
   - **1 tap:** the label shows `SINGLE`.
   - **2 quick taps:** `DOUBLE`.
   - **3 quick taps:** `TRIPLE`.
3. Run `python tools/tap_plot.py --click` to also send real clicks. **F8** toggles them.

Tuning lives at the top of [handsfree/jaw_clicker.py](handsfree/jaw_clicker.py):

| Symptom | Change |
|---------|--------|
| Taps fire while you sit still | Raise `THRESHOLD` (default 15). |
| Real taps are missed | Lower `THRESHOLD`. |
| Doubles come out as two singles | Raise `CLICK_GAP_MS` (default 250). |
| Two separate clicks merge into a double | Lower `CLICK_GAP_MS`. |

Rerun after each change. The main app uses the same settings.

### d) Pause while talking

Run `python main.py --no-clicks` and talk.
- The camera view should get a red border and a **TALKING - INPUT PAUSED** banner.
- The cursor should freeze, then resume about 0.7 s after you stop.
- The `talk detect [...]` line shows which detector is active (`presage` or `local`) and the Presage status, for example `running`, `no API key` or `auth failed`.

## 3. Run everything

```bash
python main.py
```

Calibrate with **F8** if you haven't yet, then:

| Do this | Result |
|---------|--------|
| Look somewhere | Cursor moves there, and snaps onto a nearby button or link (green box) |
| 1 tap | Left click |
| 2 quick taps | Double click |
| 3 quick taps | **Action menu** at the cursor (see below) |
| Talk | Cursor freezes, taps are ignored, and any drag is dropped |

### Action menu (3 taps)

A ring of four big buttons appears around the cursor. **Look at one and tap once.** Looking at the centre, tapping 2 or 3 times, or waiting 8 s cancels the menu. A blue dot shows where the app thinks you're looking.

| Action | What happens |
|--------|-------------|
| **Right click** (top) | Right-clicks where you opened the menu |
| **Drag / select** (right) | Presses the left button there. Move with your eyes, then any tap drops. Use it for drag-and-drop and for **selecting text**. |
| **Scroll** (bottom) | Look above the menu point to scroll up, below it to scroll down. Looking further scrolls faster. Any tap stops. |
| **Zoom** (left) | Shows a 3× magnified view of the area. Look at the exact spot: 1 tap clicks it, 2 taps double-click, 3 cancel. Use it for tiny or crowded targets. |

### Accuracy helpers

- **Clicks land where you were looking just before the tap.** The app averages your gaze over the third of a second before the first tap, so the head jolt from tapping doesn't move the click.
- **The cursor freezes** from the first tap until the click is done.
- **Snapping (F4 toggles it):**
  - The cursor locks onto a button, link, checkbox, tab, menu item or icon once your gaze comes within about 45 px of it.
  - It stays locked until you look more than about 90 px away, so jitter doesn't unsnap it.
  - Over plain text it only locks when you're right on a link, so pointing at text stays free.
  - It's off while dragging, scrolling, zooming or in the menu, so text selection is never affected.
  - Tune it with `SNAP_IN_PX`, `SNAP_OUT_PX` and `SNAP_DWELL_S` in [handsfree/snap.py](handsfree/snap.py).

**Check the integration.** Talk while tapping. Taps then appear in the debug window as `last: SINGLE (blocked: talking)` and the `blocked` count goes up, but no click happens.

| Key (global) | Action |
|-----|--------|
| F4  | Snap to buttons on/off |
| F5  | Jaw-tap clicks on/off |
| F6  | Pause-while-talking on/off |
| F7  | Cursor control on/off |
| F8  | Calibrate |
| F9  | Accuracy check |
| F10 | Quit |
| `[` / `]` | Less / more cursor smoothing (debug window focused) |
| `-` / `=` | Less / more responsiveness during fast moves |

Options: `--no-clicks` (no Arduino), `--port COM10` (pick the Arduino port), `--camera 1` (another webcam).

The function keys still reach the window you're in (F5 also refreshes a browser page, for example). Click on the debug window first if that matters.

### Debug window

- **Camera:**
  - Eye outlines, irises and a head-direction arrow.
  - `BLINK` means the cursor is holding still.
  - A green `TAP: ...` banner shows what each gesture did (`CLICK`, `MENU`, `RIGHT CLICK`, `DROP`...).
  - A red border means talking.
- **Eyes (zoomed) / iris offset:** the raw signal that drives the cursor.
- **Screen:** a minimap of your monitor. Red is the raw gaze estimate, green is smoothed, and the white cross is the actual cursor.
- **Status lines:**
  - State, cursor on/off, fps, `snap:` (`on`, `LOCKED` on a target, or `OFF`) and `mode:` (`normal`, `menu`, `drag`, `scroll` or `zoom`).
  - Calibration error.
  - Smoothing.
  - The talk detector.
  - A **clicks** line: port, readings/s, on/off, total taps, taps blocked by talking, and the last gesture, with a mini graph of the tap signal (the red line is the threshold).

## How it works

**Eye cursor**
1. MediaPipe FaceLandmarker returns 478 face points, including the irises, plus head pose.
2. The features are each iris's position inside its eye (normalized for head tilt and distance), eye openness, and head yaw and pitch.
3. A regression fitted from the 13-point calibration maps those features to screen coordinates.
4. A One Euro filter smooths the estimate, and a 120 Hz thread glides the cursor toward it.
5. The webcam runs at 640×480, because many laptop cameras only give 30 fps at that size.

**Jaw clicks**
1. The Arduino streams loudness readings as fast as it can. That's anywhere from a few hundred to tens of thousands per second, depending on the board and laptop.
2. The readings are averaged into ~1 ms ticks.
3. A fast envelope (minus a slow one, so humming or a voice cancels out) has to jump above the threshold to count as a tap.
4. Taps within 250 ms of each other form a single, double or triple gesture.
5. Each gesture is aimed at your average gaze from just before the first tap, or at the snapped target.

**Snapping**
1. A background thread asks Windows UI Automation what's at the gaze point and at 8 points around it, about 10 times a second.
2. Clickable elements become candidates. Huge elements, like full-width list rows, are skipped.
3. The cursor locks onto the nearest candidate after a short dwell, using the in/out zones described above.

**Talk pause**
1. Webcam frames are piped to a small Node process that runs the Presage SmartSpectra `TALKING` metric.
2. The bridge restarts Presage automatically if it hits a recoverable error.
3. If Presage isn't available, a detector that counts lip open/close cycles is used instead.
4. Either way, `App.inputs_paused` gates the cursor and the clicks.

## Tips for accuracy

- Light your face from the front and avoid a bright window behind you.
- Put the webcam at eye level if you can, and sit fairly close.
- Calibrate in the posture you'll actually use.
- Webcam eye tracking typically reaches 1.5–3° of error. Use large targets and smoothing.
