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

- Remote: `https://github.com/Rai1975/makeuc.git`, working branch `feat/eye_tracking`
- `eye-track/`: this work. `sensor/`: teammate's work, **do not modify**.
- Ask before pushing.

### `eye-track/` files

| File | Purpose |
|------|---------|
| `eye_cursor.py` | Eye-tracking cursor + live debug window. MediaPipe **Tasks** `FaceLandmarker` (VIDEO mode, blendshapes + pose matrix). Features: eye-axis-normalized iris offset, openness, head yaw/pitch → ridge quadratic regression from a 13-point calibration → One Euro filter → 120 Hz pynput cursor thread. Auto accuracy check (9 points, px/cm/deg). Hotkeys F7 toggle, F8 calibrate, F9 validate, F10 quit. Saves `calibration.json` (gitignored). |
| `talk_detect.py` | Pause-while-talking. `PresageTalk` (spawns the Node bridge, streams frames, parses JSON lines), `LocalTalk` (MediaPipe lip-gap fallback: range > 0.028 and ≥3 open/close reversals in 0.7 s), `TalkGate` (auto: Presage if live else local; 0.7 s hold). `App.inputs_paused` gates ALL input; F6 toggles. |
| `presage/talk_bridge.mjs` | Node bridge: stdin frames (16-byte header `<IId` w,h,tsUs + BGR) → `sdk.sendFrame` (custom input) → stdout JSON (`talking`/`status`/`error`). `presage/package.json` pins `@smartspectra/node-sdk@3.4.0`; `npm install` there (node_modules gitignored). |
| `environment.yml` | Conda env spec (`makeuc-eye`) |
| `face_landmarker.task` | MediaPipe model (copy of `sensor/face_landmarker.task`) |

Python env: conda env **`makeuc-eye`** (Python 3.12), built from `eye-track/environment.yml` (pip deps: mediapipe, opencv-python, numpy, pynput, screeninfo). Use `conda activate makeuc-eye`, or call `C:\Users\abhis\miniconda3\envs\makeuc-eye\python.exe` directly (`conda run` breaks on multi-line `-c` scripts). MediaPipe is **1.2.2, which has no `mp.solutions`** (legacy Face Mesh is gone), so use `mediapipe.tasks.python.vision`.
The old `bite_click.py`, `mouth_click.mjs` and `package.json` were deleted on 2026-11-07 (still in git history).

## Key decisions

- **Presage**: the SmartSpectra SDK works from an ordinary camera. Face metrics: `FACE_LANDMARKS` (11), `BLINKING` (12), `TALKING` (13), `EXPRESSIONS` (14), plus vitals. It is available for Node.js/Electron (`@smartspectra/node-sdk`, Node ≥20, API key from physiology.presagetech.com) and C++. **No Python SDK.** It has no jaw/bite metric, so bite detection uses MediaPipe. Verified facts (v3.4.0 types):
  - Decode with `decodeMetrics` from `@smartspectra/node-sdk/messages`. The main-package version returns a raw Buffer.
  - Talking data is at `metrics.face.talking[]` = `{detected, stable, timestamp}`.
  - `useCustomInput()` + `sendFrame(buf,w,h,stride,PixelFormat.kBGR,tsUs)` lets Python keep the webcam, since Windows allows one camera owner.
  - Auth quirks: an empty key makes `start()` throw. A bad key lets `start()` succeed, and the session sits at "starting" until frames arrive. Then it emits `error` code 2 and status 5, and logs a 401 to stderr.
  - **No API key yet** (2026-11-07). The bridge's real metrics path (actual `talking` messages, their latency and frequency) is untested. Only auth failures have been exercised. The local fallback is what runs now.
- **Click injection**: robotjs failed and `@nut-tree/nut-js` returned a 404, so use Python `pynput`.
- **Click gesture**: the user rejected mouth-open in favor of a subtle bite. Wink/blink (false positives), eyebrow raise (tiring), and an Arduino touch-sensor switch (serial → pynput) were considered and set aside.

## Next steps

1. **Current: eye → cursor only, with no clicks.** `eye_cursor.py` is built and smoke-tested (detection + debug render), but full calibration has not been run by the user yet. Run it, check the validated error, and tune it.
2. Pause-while-talking is built (Presage + local fallback). Next, get a Presage API key into `eye-track/.env`. Run `node presage/talk_bridge.mjs --debug` to see the raw metrics, then check the talking latency. Tune the `LOCAL_*` constants with real speech.
3. Merge cursor, bite/tap clicks (teammate's `sensor/plot.py` tap detector over serial), and speech into one app. Clicks must respect `App.inputs_paused`.

## Working preferences

- When brainstorming, give the general idea first and leave the tech stack out unless asked.
