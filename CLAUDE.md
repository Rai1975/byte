# Hands-Free Mouse (MakeUC / MLH Hackathon)

An assistive, hands-free mouse inspired by Augmental's MouthPad. One piece of software that emulates a mouse from three inputs:

1. **Cursor movement**: webcam eye tracking (current focus)
2. **Clicks**: subtle facial gestures, specifically **biting / jaw clenching** (not mouth opening). Movements should feel as natural and subtle as possible.
3. **Voice commands**: the teammate's `voice_agent/` ("byte" wake word → local Whisper → Claude tool use). Integrated on the `integration` branch.

## Environment

- Windows 11, PowerShell + Git Bash, VS Code / Claude desktop
- System Python 3.13 (`C:\Python313`), Miniconda (`C:\Users\abhis\miniconda3`), Node.js 22.17.1
- **No Visual Studio C++ build tools**, so npm modules that compile with node-gyp fail (robotjs did). Use Python `pynput` for mouse/click injection.
- Hardware: webcam, Arduino Uno, touch sensor (no piezo)
- Sponsor tool: **Presage** (presagetech.com)

## Repo

- Remote: `https://github.com/Rai1975/makeuc.git`. Work happens on `integration`, branched from `main` on 2026-11-08. `feat/presage` is merged. `feat/eye_tracking` is stale (old `eye-track/` layout): the user said to ignore it, don't merge it.
- Ask before pushing.
- Restructured on 2026-11-07 into one app (`python main.py`). The teammate's `sensor/` folder is gone:
  - `plot.py` became `tools/tap_plot.py`.
  - `test.py` became `tools/sensor_stream.py`.
  - `loudness.ino` moved to `arduino/loudness/`.
  - `clicker.py` was replaced by `handsfree/jaw_clicker.py`.
  - Tell the teammate if they ask where their files went.

| Path | Purpose |
|------|---------|
| `main.py` | Entry point. Flags: `--no-clicks`, `--no-voice`, `--port COMx`, `--camera N`. |
| `handsfree/app.py` | Eye cursor + debug window + `App` (MediaPipe Tasks FaceLandmarker → features → ridge regression from a 13-point calibration → One Euro → 120 Hz pynput thread). Per frame: `update_gaze` → `handle_taps` (events from JawClicker, aim = avg gaze 350–30 ms before the first tap or the snap centre) → `steer` (cursor target = freeze hold > snap centre > gaze; overlay state). Two pause gates: `App.clicks_paused` (talking OR voice agent busy) blocks gestures; `App.cursor_paused` (voice agent busy, incl. cooldown) freezes the cursor and cancels any mode. Talking never freezes the cursor (user's choice). Hotkeys: F4 snap, F5 clicks, F6 talk-pause, F7 cursor, F8 calibrate, F9 accuracy, F10 quit. Camera runs at 640x480: this laptop only gives 30 fps at ≤640x480, and 720p drops to 5 fps, which broke Presage. |
| `handsfree/jaw_clicker.py` | `find_port()` (Arduino VID 0x2341), `SerialTicks` (bulk serial read → ~1 ms block-average ticks carrying their real `tick_ms`), `TapDetector` (teammate's fast/slow envelope algorithm with ms-based timing: THRESHOLD 15, CLICK_GAP_MS 250), `JawClicker` thread (pure detector: emits `('tap'|'gesture', n, t)` via `on_event`). |
| `handsfree/actions.py` | Gesture map: 1 = click, 2 = double, 3 = radial action menu (Right click / Drag-select / Scroll / Zoom). In the menu, taps only (user's spec, 2026-11-08): 1 tap = highlight next clockwise (Right → Drag → Scroll → Zoom → Cancel → …, starts on Right click), 2 taps = choose, 3 taps = cancel, 10 s timeout from the last tap. Gaze does not select. `MENU_ENABLED = False` makes 3 taps hold the left button directly instead. Freezes the cursor from the first tap until the click, plus 0.2 s. |
| `handsfree/snap.py` | `Snapper` thread (uiautomation ControlFromPoint at the gaze + 8 points at r=50, about 65 ms per scan) + `SnapLogic` (in 60 px with 0.12 s dwell and 0.15 s grace; release only after >130 px for 0.3 s; switching to another target needs 0.3 s continuously on it; over text, capture only inside the target; skips elements bigger than 420x160). Single-line `EditControl`s (h <= 90 px, any width) are targets too; `SnapLogic.point(gaze)` returns the button centre, or for a text box the gaze clamped onto its middle line. `caption_targets()`: Electron/Chrome/VS Code don't expose min/max/close to UIA (ControlFromPoint returns a full-window Pane), so they're synthesised for every visible, uncloaked top-level window (EnumWindows) whose button strip is within SNAP_OUT_PX of the gaze, from WS_SYSMENU/MINIMIZEBOX/MAXIMIZEBOX (VS Code has no WS_SYSMENU!), DWMWA_EXTENDED_FRAME_BOUNDS and the MONITOR dpi (DPI-unaware apps report 96 but are stretched) at 46x32 px per 100%; each rect kept only if WindowFromPoint at its centre belongs to that window (not covered); a real UIA button at the same spot wins (Notepad exposes real ones). Verified pixel-aligned on VS Code at 150%. |
| `handsfree/overlay.py` | Tk thread with a full-screen transparent-colour-key window, click-through + no-activate exstyles. Gives focus back to the previous window on start. Tk objects are torn down in its own thread (otherwise you get `Tcl_AsyncDelete`). |
| `handsfree/talk_detect.py` | `PresageTalk` (Node bridge), `LocalTalk` (lip-gap fallback), `TalkGate` (auto mode, 0.7 s hold). Reads the key from `.env` at the repo root. |
| `handsfree/voice_link.py` | `VoiceLink` thread: starts `voice_agent/agent.py` with `voice_agent/.venv/Scripts/python.exe` (CREATE_NO_WINDOW; reuses an already-running agent), polls `GET 127.0.0.1:8765/status` every 0.1 s. `busy` = agent working OR within the cooldown after it's completely done: `HOLD_AFTER_S` 1 s, and `HOLD_AFTER_DICTATION_S` also 1 s if the agent flagged `dictation` (user lowered it from 4 s to 1 s on 2026-11-08; the dictation flag still shows in the HUD). The agent flags dictation from the transcript wording (`agent.DICTATION_RX`) or when `type_text` runs (`tools.ON_TOOL` hook). `working` is False if no answer for 1 s or busy for over 180 s (failsafes; teammate's dictation, `voice_agent/dictate.py`, records up to 2 min and sets dictation mode + phase "dictating"). Debug-window keys are ignored while busy. Terminates the agent on exit only if it started it. |
| `voice_agent/` | Teammate's voice agent (own Python 3.12 venv: faster-whisper on CUDA, sherpa-onnx wake word "byte", anthropic). `Agent.phase` (recording/transcribing/acting) is set in `handle_utterance`; `/status` returns `busy`. Never sends mouse input or F4–F10 (`winapi.BLOCKED_KEYS`). Its venv is NOT set up on this laptop yet. |
| `presage/talk_bridge.mjs` | Node bridge: stdin frames (16-byte `<IId` header + BGR) → `sdk.sendFrame` → stdout JSON. On retryable errors it restarts with reset()+start(), max 5 per minute. `npm install` in `presage/`. |
| `tools/tap_plot.py`, `tools/sensor_stream.py` | Testing/tuning the sensor on its own. `tap_plot.py` sends no clicks unless `--click`. |
| `arduino/loudness/loudness.ino` | Streams analogRead(A0) with no delay. |
| `models/face_landmarker.task`, `environment.yml` | Model and conda env spec. |

Python env: conda env **`makeuc-eye`** (Python 3.12) from `environment.yml`. Pip deps: mediapipe, opencv-python, numpy, pynput, screeninfo, pyserial, matplotlib, uiautomation. Call `C:\Users\abhis\miniconda3\envs\makeuc-eye\python.exe` directly (`conda run` breaks on multi-line `-c`). MediaPipe 1.2.2 has no `mp.solutions`.

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

1. The user tests per the README: calibration, sensor tools, tap tuning, talking. Then the gesture map, action menu and snapping in `python main.py`.
2. Tune the snap zones and `AIM_WINDOW_S` with real use.
3. Set up `voice_agent/.venv` on this laptop and test the voice pause for real (only tested against the agent's real control server with a faked phase).
4. Idea: the jaw clench could trigger the agent via `POST /trigger` (push-to-talk) instead of the wake word.

## Working preferences

- When brainstorming, give the general idea first and leave the tech stack out unless asked.
