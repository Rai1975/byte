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

1. Press **F8** to calibrate (about 3.5 minutes). Sit how you normally will and press SPACE at each intro. The instructions are **read aloud** (Windows speech, `--no-voice` to turn off), so you don't have to read text while looking at a dot. The phases are:
   - 25 dots, eyes only, head still
   - a 25 s moving dot, eyes only, head still
   - a 20 s moving dot while you **move your head freely** (mixed phase)
   - 3 static dots × 15 s while you move your head and keep your eyes on the dot (head phase)
2. An **accuracy check** (9 dots) runs straight after and shows the error in px, cm, and degrees. If the error is above `TARGET_PX` (35), press SPACE for another round: a new 30 s moving-dot path is added to the pooled data, the check's own points are added too, the model is refit, and it is tested again. It stops at the target, after `MAX_ROUNDS` (5), or on ESC. Cursor control then turns on.
3. To recalibrate, press F8 again, especially after you move your chair or laptop or the lighting changes.

| Key | Action |
|-----|--------|
| F6  | Pause-while-talking on/off (global) |
| F7  | Cursor control on/off (global) |
| F8  | Calibrate (global) |
| F9  | Accuracy check (global) |
| F11 | Add another round to the existing data (also works after a restart, see Sessions), then test (global) |
| F4  | **Recenter**: look at the middle dot; the cursor jumps to the screen center and the gaze zero-point is re-aligned (global) |
| F12 | Posture-shift check: lean closer / sit sideways, scored in px, never used for fitting (global) |
| F10 | Quit (global) |
| `[` / `]` | Less / more smoothing (debug window focused) |
| `-` / `=` | Less / more responsiveness during fast moves |
| `q` | Quit (debug window focused) |

The calibration is saved to `calibration.json` and reloaded on the next start (control starts OFF). Flags: `--camera N`, `--no-voice`, `--no-fixation`, `--online-refine` (experimental, see below).

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

## Recenter (F4)

When you sit down, shift in your seat, or the screen/camera moves a little, the whole cursor tends to sit a constant distance off. Press **F4** and look at the dot in the middle of the screen (about 2 s): the cursor jumps to the center and the gaze zero-point is shifted so that where you are looking right now maps to the center. It works whether or not cursor control (F7) is on. It does not retrain the model, the shift is limited to 40% of the screen, it is per sitting (reset by any calibration, never saved), and it is not applied to the accuracy and posture checks, which score the raw model. Press it again any time the cursor has drifted.

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
   - Head yaw, pitch, roll and x/y/z position.
   - MediaPipe's eye-look blendshapes (`bs_h`, `bs_up`, `bs_down`), learned from the whole eye image.
   - The iris offset expressed in the head frame (`ih_x/y/z`): the 3D eye-corner → iris vector rotated by the inverse head rotation.
   - Both eyes are averaged, weighted by eye width.
3. **Calibration**: 25 dots (5×5 grid), 2 s of samples each after a 0.7 s wait. All frames are stored raw. At fit time, per dot, the frames before the iris settles near its final position (saccade latency) and outliers (>3 robust SD) are dropped (`clean_mask`). Moving-dot frames that the linear fit explains worst (10%) are dropped too.
3b. **Eyes and head are trained separately.**
    - *Eyes only (head still)*: the 25 dots and a 30 s moving dot. The eyes trail the dot by about 100-250 ms, so the lag is estimated from the data and the dot position is matched to the eye features that much earlier. The worst 10% of samples (catch-up saccades) are dropped.
    - *Head only (eyes fixed)*: 3 static dots, 20 s each, with prompts every 4 s: move CLOSER, TILT, SPIN, move BACK, NOD. Keep your eyes on the dot. The gaze target never changes, so every shift of the iris signal here is caused by the head. A small linear model learns "iris shift per unit of head movement" (`HeadComp`) and subtracts it, so the eye model only sees the iris position as if the head were at its neutral pose. The console prints how much of the iris shift the head explains.
4. **Model**: one ridge regression per screen axis on `[1, features, iris polynomial terms]`. **Each axis independently** picks its feature set (eye only, + head pose, + blendshapes, + head-frame iris, or combinations), its polynomial kind (linear, quadratic, cubic, or linear/quadratic **with iris × head/extra-feature interaction terms**) and its ridge strength by grouped cross-validation (see Validation), so extra capacity or a new feature is only used where it lowers held-out error. The model summary shows what each axis chose, e.g. `x: eye cubic@0.001 | y: eye+head+bs lin_int@0.3`. A second stage then fits a smooth *residual field* over the predicted screen position (the remaining error depends on where on the screen you look) and subtracts it; it is kept only if it improves held-out error. Including head pose lets small head movements count instead of breaking the mapping.
5. **Smoothing**: a One Euro filter (steady when you fixate, low lag when you move). A velocity-threshold **fixation hold** (`runtime_filters.py`) then pins the cursor to the running mean of the gaze while you fixate, and releases it the moment you make a saccade (`--no-fixation` to disable; thresholds are tuned with `eval_runtime.py`). A 120 Hz cursor thread glides between camera frames, and the cursor holds still during blinks.

## Tips for accuracy

- Light your face from the front and avoid a bright window behind you.
- Raise the webcam to eye level if you can, so your eyes aren't half hidden by your lids.
- Sit closer: a bigger iris in the image means less jitter.
- The camera is opened with the Media Foundation backend (30 fps at 1280×720 here); DirectShow only managed ~10 fps at that size, which means 3× fewer samples and 3× more lag. It is used only as a fallback.
- Calibrate in the posture you'll use. If the error is high on one side, recalibrate.
- Typical webcam eye tracking reaches about 1.5–3° of error. Lean on smoothing and large targets.

## Validation

Every number the app shows for "held-out" error comes from **grouped** cross-validation (`gaze_model.py`):

- A **group** is one calibration dot or one contiguous time block (2 s of the moving dot, 3 s of a head-phase prompt), never single frames, because frames within a group are nearly identical.
- When a time block is held out, its **temporal neighbours are left out of training too**. Neighbouring blocks share slow drift and the eye's lag behind the dot, so they would leak.
- **Selection CV** (`cv_err`) picks the feature set / polynomial kind / ridge strength / stage-2 field. Because it also chose the winner it is optimistic.
- **Nested CV** (`cv_nested`) repeats the *whole* selection inside each outer fold, so it is an honest estimate of the entire procedure. It is computed once per calibration round and shown on the results screen.
- On synthetic sessions the old leave-one-group-out CV reported 104 px when the true held-out error was 120 px (16 px too optimistic); nested CV reported 117 px.
- The **accuracy check** (9 dots) is the in-session test. The **posture check** (F12) is separate: after you lean closer and then sit sideways/tilt, it scores 5 dots per posture in px. Its samples are saved in the session (phase `posture`) but are **never** fitted. Compare it with the in-session number to see how much accuracy you lose when your posture changes.

## Sessions and offline refitting

Every calibration is written to `sessions/<timestamp>.npz` (gitignored): features, screen targets, CV group ids, timestamps, phase (`dots`, `pursuit`, `mixed`, `head`, `valid`, `posture`), round, and metadata. The newest matching session is restored at startup, so F11 can add rounds after a restart.

```bash
python refit.py                  # refit the newest session, print selection CV, nested CV, held-out valid-dot error
python refit.py --all            # pool every session
python refit.py --compare        # also run the old validation (no neighbour gap) side by side
python refit.py --sweep          # table of robustness variants (no clean / openness weights / Huber)
python refit.py --prior prior.json
python eval_runtime.py           # replay a session through One Euro + fixation hold: jitter vs lag
python build_prior.py userA/ userB/ ...   # one argument per user (a .npz or a folder of sessions)
```

`refit.py` reports the *valid-dot error*: the accuracy-check dots scored by a model trained without them, which is the cleanest single number for comparing feature or model changes without recalibrating. Fitting options: `gaze_model.fit_pool(..., weights="openness", robust=True, clean=False, masks=[...], kinds=(...), prior=...)`.

### Population prior (optional)

`build_prior.py` fits a linear model per user, converts the coefficients to screen-fraction per raw feature unit (so it transfers across screens) and writes the median over users, plus the head-compensation matrix, to `prior.json`. If `prior.json` exists, the ridge shrinks toward it instead of toward zero. The ridge strength is still chosen by CV, so a poor prior is simply ignored. With no file the behaviour is unchanged.

### Online refinement (experimental, off by default)

`--online-refine` adds a small recursive-least-squares correction to the gaze prediction, learned from `App.register_click(x, y)` events. It only helps if the click location is independent of the cursor (a physical mouse correction, a snap-to-target, a click on a known button): if the cursor already follows the prediction, a click at the cursor carries no information.

## Troubleshooting the signal

- **`python iris_check.py`**: prints your eye width in pixels, the image brightness and sharpness around the eyes, and how far the iris moves between the left and right dots (healthy: about 0.15 to 0.25 of an eye width). Run it first if accuracy suddenly gets much worse.
- **`python camera_check.py`**: compares the capture backends/resolutions the same way.
- **Camera drivers remember exposure settings across programs.** A tool that leaves manual exposure on makes the image nearly black (brightness ~30/255 instead of ~140) and the iris signal collapses. `eye_cursor.py` now forces auto exposure every time it opens the camera.
- The posture check (F12) is the number to watch for day-to-day robustness: on the first clean session the in-session error was about 130 px but the posture check about 550 px. Head position/rotation changes shift the eye features in ways the calibration poses did not cover, so the calibration's head phase now also asks for sideways slides and sitting tall / slouching.
- Safeguards in `gaze_model.py`: `CLAMP_MARGIN` clips the inputs to the range seen in training (bounded extrapolation) and `COMPLEXITY_TOL` prefers the simplest model within 10% of the best cross-validation error. Both lowered the posture error a little on the first clean session (748 -> 565 px) without hurting the in-session error.
- Head compensation fits all gaze features if you set `HeadComp.TARGETS = HeadComp.TARGETS_ALL`. With the gaze fixed, head pitch moves the eyelid and eye-look signals in different directions from the iris, but compensating them all did not improve held-out error on the first clean session, so the default is still iris-only.
