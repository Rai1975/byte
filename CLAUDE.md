# Hands-Free Mouse (MakeUC / MLH Hackathon)

An assistive, hands-free mouse inspired by Augmental's MouthPad. One piece of software that emulates a mouse from three inputs:

1. **Cursor movement**: webcam eye tracking (current focus)
2. **Clicks**: subtle facial gestures, specifically **biting / jaw clenching** (not mouth opening). Movements should feel as natural and subtle as possible.
3. **Speech-to-text**: use existing software. Not built yet.

## Environment

- Windows 11, PowerShell + Git Bash, VS Code / Claude desktop
- System Python 3.13 (`C:\Python313`), Miniconda (`C:\Users\abhis\miniconda3`), Node.js 22.17.1
- **No Visual Studio C++ build tools**, so npm modules that compile with node-gyp fail (robotjs did). Use Python `pynput` for mouse/click injection.
- Hardware: webcam, Arduino Uno, touch sensor (no piezo)
- Sponsor tool: **Presage** (presagetech.com)

## Repo

- Remote: `https://github.com/Rai1975/makeuc.git`. Current branch is `feat/presage`.
- Ask before pushing.
- Restructured on 2026-11-07 into one app (`python main.py`). The teammate's `sensor/` folder is gone:
  - `plot.py` became `tools/tap_plot.py`.
  - `test.py` became `tools/sensor_stream.py`.
  - `loudness.ino` moved to `arduino/loudness/`.
  - `clicker.py` was replaced by `handsfree/jaw_clicker.py`.
  - Tell the teammate if they ask where their files went.

| Path | Purpose |
|------|---------|
| `main.py` | Entry point. Flags: `--no-clicks`, `--port COMx`, `--camera N`. |
| `handsfree/app.py` | Eye cursor + debug window + `App` (MediaPipe Tasks FaceLandmarker → features → ridge regression from a 13-point calibration → One Euro → 120 Hz pynput thread). `App.inputs_paused` gates the cursor AND the clicks. Hotkeys: F5 clicks, F6 talk-pause, F7 cursor, F8 calibrate, F9 accuracy, F10 quit. Camera runs at 640x480: this laptop only gives 30 fps at ≤640x480, and 720p drops to 5 fps, which broke Presage. |
| `handsfree/jaw_clicker.py` | `find_port()` (Arduino VID 0x2341), `SerialTicks` (bulk serial read → ~1 ms block-average ticks carrying their real `tick_ms`), `TapDetector` (teammate's fast/slow envelope algorithm with ms-based timing: THRESHOLD 15, CLICK_GAP_MS 250), `JawClicker` thread (gestures 1/2/3 = click/double/drag; blocked + drag released while `is_paused()`). |
| `handsfree/talk_detect.py` | `PresageTalk` (Node bridge), `LocalTalk` (lip-gap fallback), `TalkGate` (auto mode, 0.7 s hold). Reads the key from `.env` at the repo root. |
| `presage/talk_bridge.mjs` | Node bridge: stdin frames (16-byte `<IId` header + BGR) → `sdk.sendFrame` → stdout JSON. On retryable errors it restarts with reset()+start(), max 5 per minute. `npm install` in `presage/`. |
| `tools/tap_plot.py`, `tools/sensor_stream.py` | Testing/tuning the sensor on its own. `tap_plot.py` sends no clicks unless `--click`. |
| `arduino/loudness/loudness.ino` | Streams analogRead(A0) with no delay. |
| `models/face_landmarker.task`, `environment.yml` | Model and conda env spec. |

Python env: conda env **`makeuc-eye`** (Python 3.12) from `environment.yml`. Pip deps: mediapipe, opencv-python, numpy, pynput, screeninfo, pyserial, matplotlib. Call `C:\Users\abhis\miniconda3\envs\makeuc-eye\python.exe` directly (`conda run` breaks on multi-line `-c`). MediaPipe 1.2.2 has no `mp.solutions`.

Sensor facts:
- The Arduino is on **COM10** on this laptop (COM5 on the teammate's).
- The raw rate varies a lot: about 85k readings/s with bulk reads, but about 350/s once Arduino App Lab was open. That's why ticks carry their own duration.
- On idle data at THRESHOLD 15, a few spikes still cross the threshold. The user has not tested taps yet.

## Key decisions

- **Presage**: the SmartSpectra SDK works from an ordinary camera. Face metrics: `FACE_LANDMARKS` (11), `BLINKING` (12), `TALKING` (13), `EXPRESSIONS` (14), plus vitals. It is available for Node.js/Electron (`@smartspectra/node-sdk`, Node ≥20, API key from physiology.presagetech.com) and C++. **No Python SDK.** It has no jaw/bite metric, so bite detection uses MediaPipe. Verified facts (v3.4.0 types):
  - Decode with `decodeMetrics` from `@smartspectra/node-sdk/messages`. The main-package version returns a raw Buffer.
  - Talking data is at `metrics.face.talking[]` = `{detected, stable, timestamp}`.
  - `useCustomInput()` + `sendFrame(buf,w,h,stride,PixelFormat.kBGR,tsUs)` lets Python keep the webcam, since Windows allows one camera owner.
  - Auth quirks: an empty key makes `start()` throw. A bad key lets `start()` succeed, and the session sits at "starting" until frames arrive. Then it emits `error` code 2 and status 5, and logs a 401 to stderr.
  - The API key is in `.env` (gitignored) and works. Talking messages arrive about 5 times per second. Error 8 "processing failed" (retryable) is caused by a low camera fps.
- **Click injection**: robotjs failed and `@nut-tree/nut-js` returned a 404, so use Python `pynput`.
- **Click gesture**: the user rejected mouth-open in favor of a subtle bite. Wink/blink (false positives), eyebrow raise (tiring), and an Arduino touch-sensor switch (serial → pynput) were considered and set aside.

## Next steps

1. The user follows the README section "2. Test each part on its own": eye calibration (F8), `tools/sensor_stream.py`, `tools/tap_plot.py` (tune THRESHOLD / CLICK_GAP_MS), then talking.
2. Run `python main.py` and confirm that taps while talking show `blocked`.
3. Speech-to-text is not built yet.

## Working preferences

- When brainstorming, give the general idea first and leave the tech stack out unless asked.
