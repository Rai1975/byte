# Eye Track: webcam eye-tracking cursor

Moves the mouse cursor to where you look, using only a webcam. A live debug window shows the camera feed, zoomed eyes, the raw iris signal, and where the tracker thinks you're looking on screen.

## Setup

```bash
conda env create -f eye-track/environment.yml   # one time
conda activate makeuc-eye
python eye-track/eye_cursor.py                 # --camera 1 to pick another webcam
```

To refresh dependencies after `environment.yml` changes: `conda env update -f eye-track/environment.yml --prune`.

`face_landmarker.task` (the MediaPipe model) must sit next to the script.

## Using it

1. Press **F8** to calibrate. Sit how you normally will, keep your head still, press SPACE, and follow the 25 dots, then a 30 s moving dot, with your eyes.
2. An **accuracy check** (9 dots) runs straight after and shows the error in px, cm, and degrees. If the error is above `TARGET_PX` (35), press SPACE for another round: a new 30 s moving-dot path is added to the pooled data, the check's own points are added too, the model is refit, and it is tested again. It stops at the target, after `MAX_ROUNDS` (5), or on ESC. Cursor control then turns on.
3. To recalibrate, press F8 again, especially after you move your chair or laptop or the lighting changes.

| Key | Action |
|-----|--------|
| F6  | Pause-while-talking on/off (global) |
| F7  | Cursor control on/off (global) |
| F8  | Calibrate (global) |
| F9  | Accuracy check (global) |
| F11 | Add another moving-dot round to the existing data, then test (global) |
| F10 | Quit (global) |
| `[` / `]` | Less / more smoothing (debug window focused) |
| `-` / `=` | Less / more responsiveness during fast moves |
| `q` | Quit (debug window focused) |

The calibration is saved to `calibration.json` and reloaded on the next start (control starts OFF).

## Pause while talking (Presage)

When you start talking, all mouse input stops: the cursor freezes where it is, and future click inputs must check `App.inputs_paused` too. Input resumes about 0.7 s after you stop talking. While paused, the camera view gets a red border and a "TALKING - INPUT PAUSED" banner.

Two detectors are available ([talk_detect.py](talk_detect.py)):

- **Presage SmartSpectra** (preferred): its `TALKING` metric. The SDK is Node-only, so [presage/talk_bridge.mjs](presage/talk_bridge.mjs) runs as a child process, and `eye_cursor.py` streams its webcam frames to it over stdin. Python keeps the only handle to the camera. One-time setup:
  ```bash
  cd eye-track/presage && npm install     # pulls ~few hundred MB of native runtimes
  ```
  Then put your key (from [physiology.presagetech.com](https://physiology.presagetech.com)) in `eye-track/.env`, which is gitignored:
  ```text
  PRESAGE_API_KEY=your-key-here
  ```
- **Local fallback**: when Presage isn't available, the lip gap from MediaPipe landmarks is used instead. Three or more open/close cycles within 0.7 s count as talking, so a yawn or a single mouth opening doesn't.

The debug window shows which detector is active, the Presage status (for example `no API key`, `auth failed`, or `running`), and the fallback's live `lips range/reversals` values for tuning. Set `TALK_MODE` in `talk_detect.py` to `auto` (default: Presage if live, else local), `both`, `presage`, or `local`.

## Debug window

- **Camera**: eye contours (cyan), iris (green ring, red center), and head direction (yellow arrow). "BLINK" means the cursor is frozen.
- **Eyes (zoomed)**: crops of each eye. The cyan × is the eye-corner midpoint and the red + is the iris center. Their offset is the signal.
- **Iris offset**: normalized iris position (the "joystick") with a short trail. The purple box is the range covered during calibration.
- **Screen**: minimap of your monitor. Red = raw prediction, green = smoothed, white cross = actual cursor.

## How it works

1. **MediaPipe FaceLandmarker** returns 478 landmarks (including 10 iris points), a head-pose matrix, and blendshapes.
2. **Features** (per frame):
   - The iris center relative to the eye-corner midpoint, rotated into the eye's own axis and divided by eye width. This makes it independent of head roll and distance.
   - Eye openness, which helps with vertical gaze.
   - Head yaw and pitch.
   - Both eyes are averaged, weighted by eye width.
3. **Calibration**: 25 dots (5×5 grid), about 1.8 s of samples each after a 1.4 s settle delay. Saccade and glitch frames are rejected per dot with a median/MAD filter.
3b. **Eyes and head are trained separately.**
    - *Eyes only (head still)*: the 25 dots and a 30 s moving dot. The eyes trail the dot by about 100-250 ms, so the lag is estimated from the data and the dot position is matched to the eye features that much earlier. The worst 10% of samples (catch-up saccades) are dropped.
    - *Head only (eyes fixed)*: 3 static dots, 20 s each, with prompts every 4 s: move CLOSER, TILT, SPIN, move BACK, NOD. Keep your eyes on the dot. The gaze target never changes, so every shift of the iris signal here is caused by the head. A small linear model learns "iris shift per unit of head movement" (`HeadComp`) and subtracts it, so the eye model only sees the iris position as if the head were at its neutral pose. The console prints how much of the iris shift the head explains.
4. **Model**: one ridge regression per screen axis on `[1, features, iris polynomial terms]`. Each axis picks its polynomial order (linear, quadratic or cubic in the iris offset) and ridge strength by leave-one-dot-out cross-validation, so extra capacity is only used if it helps held-out dots. The debug window shows the fit error next to the held-out error; a large gap means overfitting. A second stage then fits a smooth *residual field* over the predicted screen position (the remaining error depends on where on the screen you look) and subtracts it; it is kept only if it improves held-out error. Including head pose lets small head movements count instead of breaking the mapping.
5. **Smoothing**: a One Euro filter (steady when you fixate, low lag when you move). A 120 Hz cursor thread glides between camera frames, and the cursor holds still during blinks.

## Tips for accuracy

- Light your face from the front and avoid a bright window behind you.
- Raise the webcam to eye level if you can, so your eyes aren't half hidden by your lids.
- Sit closer: a bigger iris in the image means less jitter.
- Calibrate in the posture you'll use. If the error is high on one side, recalibrate.
- Typical webcam eye tracking reaches about 1.5–3° of error. Lean on smoothing and large targets.
