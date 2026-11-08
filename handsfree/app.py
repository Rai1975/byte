"""
Hands-free mouse: eye-tracking cursor + jaw-tap clicks + voice agent, with a live
debug visualization. Run from the repo root:  python main.py

Pipeline
  webcam -> MediaPipe FaceLandmarker (478 landmarks incl. irises, head pose, blendshapes)
         -> features: iris offset inside each eye (rotation/scale normalized), eye openness,
            head yaw/pitch
         -> ridge-regularized polynomial regression fitted on a 13-point calibration
         -> One Euro filter (low jitter when still, responsive when moving)
         -> 120 Hz cursor driver (pynput) that glides toward the filtered gaze point

Clicks come from jaw taps on the Arduino loudness sensor (jaw_clicker.py -> actions.py):
1 tap = click, 2 = double click, 3 = action menu (1 tap = next option clockwise, 2 = choose, 3 = cancel).
Clicks land where you looked just before the tap, and the cursor freezes during the tap.
The cursor snaps onto nearby buttons/links (snap.py); overlay.py draws the menu and highlights.

Pausing (any new input must respect these):
  App.clicks_paused  - the user is talking (talk_detect.py: Presage TALKING, lip-movement fallback)
                       OR the voice agent is busy. Jaw taps do nothing. Talking leaves the cursor live.
  App.cursor_paused  - the voice agent (voice_link.py -> voice_agent/) is handling a command, from
                       its wake word until it's completely done + a 1 s cooldown
                       (voice_link.HOLD_AFTER_S / HOLD_AFTER_DICTATION_S).
                       Cursor freezes, any drag/menu/scroll is dropped, debug-window keys are ignored.

Hotkeys (global, work even when another window has focus)
  F4  toggle snapping to buttons on/off
  F5  toggle jaw-tap clicks on/off
  F6  toggle pause-clicks-while-talking on/off
  F7  toggle cursor control on/off
  F8  recalibrate
  F9  run accuracy validation
  F10 quit
Debug-window keys (window must be focused)
  [ / ]  less / more smoothing     - / =  less / more responsiveness (beta)
  q      quit
"""

import argparse
import ctypes
import json
import math
import os
import queue
import threading
import time
from collections import deque

# Must run before any window is created, so OpenCV, pynput and the screen size all use
# physical pixels (otherwise Windows display scaling skews every coordinate).
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from pynput import keyboard as pkeyboard
from pynput import mouse as pmouse

from .actions import AIM_WINDOW_S, Actions
from .jaw_clicker import GESTURES, THRESHOLD as TAP_THRESHOLD, JawClicker
from .overlay import Overlay
from .snap import SnapLogic, Snapper
from .talk_detect import TalkGate
from .voice_link import VoiceLink

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
MODEL_PATH = os.path.join(ROOT, "models", "face_landmarker.task")
CALIB_PATH = os.path.join(ROOT, "calibration.json")

# ----------------------------------------------------------------------------- tuning
# 640x480 on purpose: this laptop webcam does 30 fps only at <= 640x480 (720p drops to 5 fps,
# which makes the cursor laggy and makes Presage fail with "processing failed").
CAM_W, CAM_H, CAM_FPS = 640, 480, 30
LOW_FPS_WARN = 15
SETTLE_S = 0.9          # time to let the eyes land on a calibration dot before sampling
COLLECT_S = 1.0         # sampling time per calibration dot
MIN_SAMPLES = 10        # minimum good frames per dot
RIDGE = 0.01            # regularization strength (standardized features)
BLINK_THRESH = 0.45     # blendshape eyeBlink score above which the cursor freezes
ONE_EURO_MIN_CUTOFF = 0.6   # Hz; lower = steadier when still
ONE_EURO_BETA = 0.004       # higher = less lag during fast movements
ONE_EURO_D_CUTOFF = 1.0
CURSOR_HZ = 120
CURSOR_GLIDE = 0.35     # fraction of remaining distance covered per cursor tick
CAM_HFOV_DEG = 70       # typical webcam horizontal field of view (only used for distance estimate)
IRIS_DIAMETER_MM = 11.7 # nearly constant across adults -> gives camera distance from iris size

# 3x3 grid near the edges + 4 inner points
CALIB_POINTS = [(x, y) for y in (0.06, 0.5, 0.94) for x in (0.04, 0.5, 0.96)]
CALIB_POINTS = CALIB_POINTS[0:3] + CALIB_POINTS[3:6][::-1] + CALIB_POINTS[6:9]  # serpentine
CALIB_POINTS += [(0.27, 0.28), (0.73, 0.28), (0.73, 0.72), (0.27, 0.72)]
VALID_POINTS = [(0.2, 0.2), (0.5, 0.15), (0.8, 0.2), (0.85, 0.5), (0.8, 0.8),
                (0.5, 0.85), (0.2, 0.8), (0.15, 0.5), (0.5, 0.5)]

# ----------------------------------------------------------------------------- landmarks
# Corner/lid/iris indices per eye. Corner order is resolved by x at runtime, so this works
# on the mirrored image too.
EYES = [
    dict(corners=(33, 133), lids=(159, 145), iris=(468, 469, 470, 471, 472),
         contour=[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]),
    dict(corners=(362, 263), lids=(386, 374), iris=(473, 474, 475, 476, 477),
         contour=[362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]),
]
NOSE_TIP = 1

# colors (BGR)
C_GREEN = (90, 220, 120)
C_RED = (80, 80, 255)
C_CYAN = (230, 210, 80)
C_YELLOW = (60, 220, 240)
C_WHITE = (240, 240, 240)
C_GRAY = (120, 120, 120)
C_DARK = (32, 32, 36)
C_PANEL = (22, 22, 26)
C_MAGENTA = (220, 90, 220)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ============================================================================ helpers
def screen_info():
    u32 = ctypes.windll.user32
    sw, sh = u32.GetSystemMetrics(0), u32.GetSystemMetrics(1)
    width_mm = None
    try:
        from screeninfo import get_monitors
        for m in get_monitors():
            if m.is_primary and m.width_mm:
                width_mm = m.width_mm
    except Exception:
        pass
    return sw, sh, width_mm


class OneEuro:
    """One Euro filter (Casiez et al. 2012) on a 2D point."""

    def __init__(self, min_cutoff, beta, d_cutoff):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None
        self.dx = np.zeros(2)
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x, self.t = None, None
        self.dx = np.zeros(2)

    def __call__(self, x, t):
        x = np.asarray(x, dtype=float)
        if self.x is None:
            self.x, self.t = x, t
            return x
        dt = max(t - self.t, 1e-3)
        self.t = t
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        return self.x


class GazeModel:
    """Ridge regression: [1, z, ix^2, iy^2, ix*iy] -> screen (x, y), z = standardized features."""

    FEATURES = ["iris_x", "iris_y", "openness", "head_yaw", "head_pitch"]

    def __init__(self):
        self.mu = self.sd = self.W = None

    @staticmethod
    def _expand(Z):
        ix, iy = Z[:, 0], Z[:, 1]
        return np.column_stack([np.ones(len(Z)), Z, ix * ix, iy * iy, ix * iy])

    def fit(self, F, S, alpha=RIDGE):
        F, S = np.asarray(F, float), np.asarray(S, float)
        self.mu = F.mean(0)
        self.sd = F.std(0) + 1e-9
        X = self._expand((F - self.mu) / self.sd)
        reg = np.eye(X.shape[1]) * alpha * len(X)
        reg[0, 0] = 0.0
        self.W = np.linalg.solve(X.T @ X + reg, X.T @ S)

    def predict(self, f):
        z = (np.asarray(f, float)[None, :] - self.mu) / self.sd
        return (self._expand(z) @ self.W)[0]

    @property
    def ready(self):
        return self.W is not None

    def to_dict(self):
        return dict(mu=self.mu.tolist(), sd=self.sd.tolist(), W=self.W.tolist())

    def from_dict(self, d):
        self.mu, self.sd, self.W = (np.array(d[k]) for k in ("mu", "sd", "W"))


def extract(result, W, H):
    """Turn a FaceLandmarker result into gaze features plus drawing info."""
    if not result.face_landmarks:
        return None
    lms = result.face_landmarks[0]
    pts = np.array([(p.x * W, p.y * H) for p in lms], dtype=np.float64)
    eyes = []
    for e in EYES:
        a, b = pts[e["corners"][0]], pts[e["corners"][1]]
        if a[0] > b[0]:
            a, b = b, a
        w = float(np.linalg.norm(b - a)) + 1e-6
        u = (b - a) / w                     # along the eye (left -> right in image)
        v = np.array([-u[1], u[0]])         # perpendicular (downwards in image)
        c = (a + b) / 2
        iris = pts[list(e["iris"])].mean(axis=0)
        r = float(np.mean([np.linalg.norm(pts[i] - pts[e["iris"][0]]) for i in e["iris"][1:]]))
        d = iris - c
        openness = float(np.linalg.norm(pts[e["lids"][1]] - pts[e["lids"][0]])) / w
        eyes.append(dict(ix=float(d @ u) / w, iy=float(d @ v) / w, open=openness, w=w,
                         c=c, iris=iris, r=r, contour=pts[e["contour"]]))
    wsum = eyes[0]["w"] + eyes[1]["w"]      # weight by eye width: the foreshortened eye is noisier
    ix = sum(e["ix"] * e["w"] for e in eyes) / wsum
    iy = sum(e["iy"] * e["w"] for e in eyes) / wsum
    op = sum(e["open"] * e["w"] for e in eyes) / wsum

    hx = hy = 0.0
    dist_cm = None
    if result.facial_transformation_matrixes:
        M = np.asarray(result.facial_transformation_matrixes[0])
        hx, hy = float(M[0, 2]), float(M[1, 2])   # face forward vector, x/y components (y up)
    # distance from apparent iris size, assuming a typical webcam field of view
    focal_px = (W / 2) / math.tan(math.radians(CAM_HFOV_DEG / 2))
    r_px = max(e["r"] for e in eyes)
    dist_cm = focal_px * (IRIS_DIAMETER_MM / 2) / max(r_px, 1e-3) / 10
    blink = 0.0
    if result.face_blendshapes:
        bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
        blink = (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2

    return dict(feat=np.array([ix, iy, op, hx, hy]), ix=ix, iy=iy, eyes=eyes, pts=pts,
                hx=hx, hy=hy, dist_cm=dist_cm, blink=blink)


def robust_filter(F):
    """Drop frames whose iris features are far from the per-dot median (saccades, glitches)."""
    F = np.asarray(F)
    med = np.median(F, axis=0)
    mad = np.median(np.abs(F - med), axis=0) * 1.4826 + 1e-6
    keep = np.all(np.abs(F[:, :2] - med[:2]) <= 3 * mad[:2], axis=1)
    return F[keep]


def put(img, text, org, scale=0.5, color=C_WHITE, thick=1, bg=None):
    if bg is not None:
        (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
        x, y = org
        cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + base + 2), bg, -1)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


# ============================================================================ cursor driver
class CursorDriver(threading.Thread):
    """Moves the OS cursor at CURSOR_HZ, gliding toward the latest gaze target."""

    def __init__(self, sw, sh):
        super().__init__(daemon=True)
        self.mouse = pmouse.Controller()
        self.sw, self.sh = sw, sh
        self.target = None
        self.enabled = False
        self.paused = False     # True while the voice agent acts: cursor stays exactly where it is
        self.alive = True
        self._pos = None

    def set_enabled(self, on):
        if on and not self.enabled:
            self._pos = np.array(self.mouse.position, dtype=float)
        self.enabled = on

    def run(self):
        period = 1.0 / CURSOR_HZ
        while self.alive:
            t = self.target
            if self.enabled and not self.paused and t is not None:
                if self._pos is None:
                    self._pos = np.array(self.mouse.position, dtype=float)
                self._pos += (np.asarray(t) - self._pos) * CURSOR_GLIDE
                x = int(min(max(self._pos[0], 0), self.sw - 1))
                y = int(min(max(self._pos[1], 0), self.sh - 1))
                try:
                    self.mouse.position = (x, y)
                except Exception:
                    pass
            time.sleep(period)


# ============================================================================ app
class App:
    DBG = "Hands-Free Mouse - debug"
    CAL = "Hands-Free Mouse - calibration"

    def __init__(self, cam_index, port=None, clicks=True, voice=True):
        self.sw, self.sh, self.width_mm = screen_info()
        self.px_per_mm = (self.sw / self.width_mm) if self.width_mm else None

        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open camera {cam_index}")

        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self.last_ts = -1

        self.model = GazeModel()
        self.filter = OneEuro(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
        self.cursor = CursorDriver(self.sw, self.sh)
        self.cursor.start()

        # Talking blocks jaw-tap clicks (the cursor keeps following your eyes).
        self.talk = TalkGate()
        self.talking = False
        # The voice agent handling a command pauses everything (cursor + clicks).
        self.voice = VoiceLink(launch=voice)
        self.voice.start()
        self.voice_busy = False

        self.actions = Actions(self.sw, self.sh, self.cursor)
        self.overlay = Overlay(self.sw, self.sh)
        self.overlay.start()
        self.snapper = Snapper()
        self.snapper.start()
        self.snap = SnapLogic()
        self.snap_enabled = True
        self.gaze_pt = None              # latest smoothed gaze on screen (None = unknown)
        self.gaze_hist = deque(maxlen=90)   # (t, x, y) for aiming clicks
        self.tap_events = queue.Queue()
        self.tap_aim = None              # aim point captured at the first tap of a gesture
        self.clicks_enabled = True
        self.blocked = 0
        self.last_action, self.last_action_t = "", 0.0
        self.clicker = None
        if clicks:
            self.clicker = JawClicker(port, on_event=lambda *e: self.tap_events.put(e))
            self.clicker.start()

        self.keys = queue.Queue()
        self.listener = pkeyboard.Listener(on_press=self._on_global_key)
        self.listener.start()

        self.state = "idle"     # idle | calib | valid | results | run
        self.raw = None         # latest raw gaze prediction (screen px)
        self.smooth = None      # latest filtered gaze prediction
        self.gaze_trail = deque(maxlen=45)
        self.iris_trail = deque(maxlen=45)
        self.iris_range = None  # (min, max) of iris features seen in calibration
        self.fps = 0.0
        self.calib_err = None
        self.valid_stats = None
        self.message = ""
        self.message_until = 0
        self.dbg_open = False
        self.cal_bg = np.zeros((self.sh, self.sw, 3), np.uint8)
        self.cal_bg[:] = C_DARK

        if self._load_calibration():
            self.state = "run"
            self.flash("Loaded saved calibration. F7 = cursor on/off, F8 = recalibrate", 8)
        else:
            self.flash("Not calibrated yet. Press F8 to calibrate.", 3600)

    # ------------------------------------------------------------------ input
    def _on_global_key(self, key):
        names = {pkeyboard.Key.f4: "snap", pkeyboard.Key.f5: "clicks", pkeyboard.Key.f6: "talk", pkeyboard.Key.f7: "toggle", pkeyboard.Key.f8: "calib", pkeyboard.Key.f9: "valid",
                 pkeyboard.Key.f10: "quit", pkeyboard.Key.space: "space", pkeyboard.Key.esc: "esc"}
        if key in names:
            self.keys.put(names[key])

    def flash(self, msg, seconds=4):
        self.message, self.message_until = msg, time.time() + seconds

    # ------------------------------------------------------------------ persistence
    def _save_calibration(self):
        d = dict(model=self.model.to_dict(), screen=[self.sw, self.sh],
                 iris_range=[list(map(float, r)) for r in self.iris_range] if self.iris_range else None,
                 calib_err=self.calib_err, saved=time.ctime())
        with open(CALIB_PATH, "w") as f:
            json.dump(d, f, indent=1)

    def _load_calibration(self):
        try:
            with open(CALIB_PATH) as f:
                d = json.load(f)
            if d["screen"] != [self.sw, self.sh]:
                return False
            self.model.from_dict(d["model"])
            if d.get("iris_range"):
                self.iris_range = tuple(np.array(r) for r in d["iris_range"])
            self.calib_err = d.get("calib_err")
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ frame processing
    def grab(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        frame = cv2.flip(frame, 1)  # mirror: looks natural and makes "look right" = "iris right"
        ts = int(time.perf_counter() * 1000)
        if ts <= self.last_ts:
            ts = self.last_ts + 1
        self.last_ts = ts
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = self.landmarker.detect_for_video(img, ts)
        info = extract(res, frame.shape[1], frame.shape[0])
        now = time.time()
        self.talk.feed(frame, info, now)
        self.talking = self.talk.talking(now)
        self.voice_busy = self.voice.busy
        return frame, info

    @property
    def cursor_paused(self):
        """True while the voice agent is handling a command: the cursor must not move."""
        return self.voice_busy

    @property
    def clicks_paused(self):
        """True while talking or while the voice agent is busy: jaw taps must not click."""
        return self.talking or self.voice_busy

    def _pause_reason(self):
        return "voice agent" if self.voice_busy else "talking"

    def update_gaze(self, info, now):
        self.cursor.paused = self.cursor_paused
        if info is None or not self.model.ready:
            return
        self.iris_trail.append((info["ix"], info["iy"]))
        if self.cursor_paused:
            self._resume_clean = True
            return  # voice agent acting (or cooling down): freeze the cursor
        if getattr(self, "_resume_clean", False):    # restart smoothing from here, no lurch from old state
            self._resume_clean = False
            self.filter.reset()
        if info["blink"] > BLINK_THRESH:
            return  # hold position through blinks
        raw = self.model.predict(info["feat"])
        raw = np.clip(raw, [-0.05 * self.sw, -0.05 * self.sh], [1.05 * self.sw, 1.05 * self.sh])
        self.raw = raw
        self.smooth = self.filter(raw, now)
        self.gaze_trail.append(tuple(self.smooth))
        self.gaze_pt = (float(np.clip(self.smooth[0], 0, self.sw - 1)),
                        float(np.clip(self.smooth[1], 0, self.sh - 1)))
        self.gaze_hist.append((now, *self.gaze_pt))

    # ------------------------------------------------------------------ taps, snapping, actions
    def _aim(self, t):
        """Where a gesture whose first tap happened at time t should land."""
        t0, t1 = t - AIM_WINDOW_S[0], t - AIM_WINDOW_S[1]
        pts = [(x, y) for ts, x, y in self.gaze_hist if t0 <= ts <= t1]
        looked = tuple(np.mean(pts, axis=0)) if pts else self.gaze_pt
        if self.actions.mode == "normal":
            if self.snap.current is not None:    # button centre, or along a text box where you looked
                return self.snap.point(looked)
            if not self.cursor.enabled:          # eye cursor off: click wherever the mouse is
                return tuple(self.cursor.mouse.position)
        if looked is not None:
            return looked
        return tuple(self.cursor.mouse.position)

    def _set_last(self, label):
        self.last_action, self.last_action_t = label, time.time()

    def handle_taps(self, now):
        while not self.tap_events.empty():
            kind, g, t = self.tap_events.get_nowait()
            blocked = self.clicks_paused or not self.clicks_enabled
            if kind == "tap":
                if not blocked and self.tap_aim is None:
                    self.tap_aim = self._aim(t)
                    self.actions.tap_started(self.tap_aim, now)
                continue
            aim, self.tap_aim = self.tap_aim, None
            name = GESTURES[g]
            if not self.clicks_enabled:
                self.actions.hold = None
                self._set_last(f"{name} (clicks off)")
                continue
            if self.clicks_paused:
                self.actions.hold = None
                self.blocked += 1
                self._set_last(f"{name} (blocked: {self._pause_reason()})")
                continue
            if aim is None:
                aim = self._aim(t)
            label = self.actions.gesture(g, aim, now)
            if self.actions.mode != "normal":
                self.snap.reset()
            self._set_last(label)

    def steer(self, now):
        """Pick the cursor target (freeze > snap > gaze), run modes, update the overlay."""
        paused = self.cursor_paused
        if paused and self.actions.mode != "normal":     # voice agent took over: leave any mode
            was = self.actions.mode
            self.actions.cancel()
            self._set_last(f"{'DROP' if was == 'drag' else 'CANCEL ' + was.upper()} (voice agent)")
        gaze = self.gaze_pt if self.model.ready else None
        snap_on = (self.snap_enabled and self.cursor.enabled and gaze is not None and not paused
                   and self.actions.mode == "normal")
        if snap_on:
            self.snapper.query = gaze
            if self.actions.hold is None:        # don't re-snap while frozen for a click
                self.snap.update(gaze, self.snapper.result, now)
        else:
            self.snapper.query = None
            self.snap.reset()
        hold = self.actions.hold_point(now)
        target = hold or self.snap.point(gaze) or gaze
        if target is not None:
            self.cursor.target = target
        self.actions.frame(gaze, now, paused)
        st = self.actions.overlay_state(gaze)
        if self.snap.current is not None:
            st["snap"] = self.snap.current[0]
        self.overlay.set(st)

    def reset_inputs(self):
        """Before calibration etc.: leave any mode, clear the overlay, drop queued taps."""
        self.actions.cancel()
        self.snap.reset()
        self.snapper.query = None
        self.tap_aim = None
        self.overlay.set({})
        while not self.tap_events.empty():
            self.tap_events.get_nowait()

    # ------------------------------------------------------------------ windows
    def open_debug(self, canvas_shape):
        if self.dbg_open:
            return
        cv2.namedWindow(self.DBG, cv2.WINDOW_AUTOSIZE)
        h, w = canvas_shape[:2]
        cv2.moveWindow(self.DBG, self.sw - w - 30, self.sh - h - 110)
        try:
            cv2.setWindowProperty(self.DBG, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass
        self.dbg_open = True

    def close_debug(self):
        if self.dbg_open:
            cv2.destroyWindow(self.DBG)
            self.dbg_open = False

    def open_fullscreen(self):
        self.close_debug()
        cv2.namedWindow(self.CAL, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.CAL, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        try:
            cv2.setWindowProperty(self.CAL, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass

    def close_fullscreen(self):
        try:
            cv2.destroyWindow(self.CAL)
        except cv2.error:
            pass

    # ------------------------------------------------------------------ target sequences
    def run_targets(self, points, mode):
        """Show dots one by one. mode='calib' collects features, mode='valid' collects predictions.
        Returns list of (target_px, samples) or None if cancelled."""
        self.open_fullscreen()
        was_enabled = self.cursor.enabled
        self.cursor.set_enabled(False)
        title = "CALIBRATION" if mode == "calib" else "ACCURACY CHECK"
        sub = ("Keep your head still and follow the dot with your EYES only."
               if mode == "calib" else "Look at each dot. This measures how accurate tracking is.")

        # intro screen: wait for SPACE
        while True:
            frame, info = self.grab()
            img = self.cal_bg.copy()
            cx, cy = self.sw // 2, self.sh // 2
            put(img, title, (cx - 220, cy - 120), 1.6, C_CYAN, 3)
            put(img, sub, (cx - 560, cy - 40), 0.9)
            put(img, "Press SPACE to start, ESC to cancel", (cx - 330, cy + 20), 0.9, C_YELLOW, 2)
            if frame is not None:
                self._draw_face_status(img, frame, info)
            cv2.imshow(self.CAL, img)
            k = cv2.waitKey(1) & 0xFF
            ev = self._drain_keys()
            if k == 27 or "esc" in ev:
                self.close_fullscreen()
                self.cursor.set_enabled(was_enabled)
                return None
            if k == 32 or "space" in ev:
                break

        out = []
        for (px, py) in points:
            target = np.array([px * self.sw, py * self.sh])
            samples = []
            t0 = time.time()
            while True:
                now = time.time()
                el = now - t0
                frame, info = self.grab()
                good = (info is not None and info["blink"] < BLINK_THRESH and not self.talking
                        and not self.voice_busy)
                collecting = el >= SETTLE_S
                if collecting and good:
                    if mode == "calib":
                        samples.append(info["feat"])
                    elif self.model.ready:
                        samples.append(self.model.predict(info["feat"]))
                done = collecting and el >= SETTLE_S + COLLECT_S and len(samples) >= MIN_SAMPLES
                if done or el > SETTLE_S + COLLECT_S + 4.0:
                    break

                img = self.cal_bg.copy()
                tx, ty = int(target[0]), int(target[1])
                if not collecting:
                    rad = int(12 + 40 * (1 - el / SETTLE_S))
                    cv2.circle(img, (tx, ty), rad, C_WHITE, 2, cv2.LINE_AA)
                    cv2.circle(img, (tx, ty), 6, C_WHITE, -1, cv2.LINE_AA)
                else:
                    frac = min((el - SETTLE_S) / COLLECT_S, 1.0)
                    cv2.circle(img, (tx, ty), 14, C_GREEN, -1, cv2.LINE_AA)
                    cv2.ellipse(img, (tx, ty), (26, 26), -90, 0, 360 * frac, C_GREEN, 3, cv2.LINE_AA)
                    cv2.circle(img, (tx, ty), 3, (0, 0, 0), -1, cv2.LINE_AA)
                if frame is not None and (info is None):
                    put(img, "FACE NOT DETECTED", (self.sw // 2 - 200, int(self.sh * 0.18)), 1.2, C_RED, 3)
                put(img, f"{len(out) + 1}/{len(points)}", (30, self.sh - 30), 0.8, C_GRAY, 2)
                cv2.imshow(self.CAL, img)
                k = cv2.waitKey(1) & 0xFF
                if k == 27 or "esc" in self._drain_keys():
                    self.close_fullscreen()
                    self.cursor.set_enabled(was_enabled)
                    return None
            out.append((target, samples))
        self.cursor.set_enabled(was_enabled)
        return out

    def _drain_keys(self):
        ev = []
        while not self.keys.empty():
            ev.append(self.keys.get_nowait())
        # re-queue control events so the main loop still sees them
        for e in ev:
            if e in ("quit",):
                self.keys.put(e)
        return ev

    def _draw_face_status(self, img, frame, info):
        """Small camera preview on the intro screen so the user can check framing."""
        prev = cv2.resize(frame, (480, int(480 * frame.shape[0] / frame.shape[1])))
        if info is not None:
            s = 480 / frame.shape[1]
            for e in info["eyes"]:
                cv2.circle(prev, tuple(int(v) for v in e["iris"] * s), max(2, int(e["r"] * s)), C_GREEN, 1)
        h = prev.shape[0]
        x0, y0 = self.sw // 2 - 240, self.sh // 2 + 70
        img[y0:y0 + h, x0:x0 + 480] = prev
        put(img, "face OK" if info else "no face", (x0, y0 + h + 30), 0.8, C_GREEN if info else C_RED, 2)

    def calibrate(self):
        res = self.run_targets(CALIB_POINTS, "calib")
        if res is None:
            self.flash("Calibration cancelled", 4)
            self.state = "run" if self.model.ready else "idle"
            return
        F, S, per_point = [], [], []
        for target, samples in res:
            if len(samples) < 5:
                continue
            kept = robust_filter(samples)
            F.extend(kept)
            S.extend([target] * len(kept))
            per_point.append((target, kept))
        if len(per_point) < 8:
            self.close_fullscreen()
            self.flash("Calibration failed: face not tracked on enough dots. Check lighting, F8 to retry.", 8)
            self.state = "run" if self.model.ready else "idle"
            return
        self.model.fit(F, S)
        F = np.array(F)
        self.iris_range = (F[:, :2].min(0), F[:, :2].max(0))
        errs = [np.linalg.norm(self.model.predict(np.mean(k, 0)) - t) for t, k in per_point]
        self.calib_err = float(np.mean(errs))
        self.filter.reset()
        self._save_calibration()
        self.validate(auto=True)

    def validate(self, auto=False):
        if not self.model.ready:
            self.flash("Calibrate first (F8)", 4)
            return
        res = self.run_targets(VALID_POINTS, "valid")
        if res is None:
            self.close_fullscreen()
            self.state = "run"
            if auto:
                self.cursor.set_enabled(True)
            return
        rows = []
        for target, preds in res:
            if len(preds) < 3:
                continue
            P = np.array(preds)
            m = P.mean(0)
            rows.append((target, m, float(np.linalg.norm(m - target)),
                         float(np.sqrt(np.mean(np.sum((P - m) ** 2, 1))))))
        if not rows:
            self.close_fullscreen()
            self.state = "run"
            return
        acc = float(np.mean([r[2] for r in rows]))
        prec = float(np.mean([r[3] for r in rows]))
        stats = dict(acc_px=acc, prec_px=prec)
        dist = None
        if self.px_per_mm:
            stats["acc_cm"] = acc / self.px_per_mm / 10
            _, info = self.grab()
            dist = info["dist_cm"] if info else None
            if dist:
                stats["acc_deg"] = math.degrees(math.atan(acc / self.px_per_mm / (dist * 10)))
        self.valid_stats = stats

        # results screen
        t0 = time.time()
        while time.time() - t0 < 12:
            img = self.cal_bg.copy()
            for target, m, err, pr in rows:
                t = tuple(int(v) for v in target)
                p = tuple(int(v) for v in m)
                cv2.circle(img, t, 10, C_WHITE, 2, cv2.LINE_AA)
                cv2.line(img, t, p, C_YELLOW, 1, cv2.LINE_AA)
                cv2.circle(img, p, max(4, int(pr)), C_GREEN, 1, cv2.LINE_AA)
                cv2.circle(img, p, 5, C_GREEN, -1, cv2.LINE_AA)
                put(img, f"{err:.0f}px", (p[0] + 10, p[1] - 10), 0.6, C_YELLOW)
            cx, cy = self.sw // 2, int(self.sh * 0.33)
            put(img, "ACCURACY", (cx - 120, cy - 50), 1.3, C_CYAN, 3)
            line = f"mean error {acc:.0f}px ({100 * acc / self.sw:.1f}% of screen width)"
            if "acc_cm" in stats:
                line += f" = {stats['acc_cm']:.1f} cm"
            if "acc_deg" in stats:
                line += f" = {stats['acc_deg']:.1f} deg"
            put(img, line + ("  (deg/cm approx.)" if "acc_deg" in stats else ""), (cx - 520, cy), 0.9)
            put(img, f"jitter (precision) {prec:.0f}px before smoothing", (cx - 520, cy + 40), 0.9)
            put(img, "white = target, green = where tracking thought you looked (circle = spread)",
                (cx - 520, cy + 80), 0.7, C_GRAY)
            put(img, "SPACE/ESC to continue   |   F8 to recalibrate", (cx - 520, cy + 130), 0.8, C_YELLOW, 2)
            cv2.imshow(self.CAL, img)
            k = cv2.waitKey(30) & 0xFF
            ev = self._drain_keys()
            if k in (27, 32) or "space" in ev or "esc" in ev:
                break
            if "calib" in ev:
                self.keys.put("calib")
                break
        self.close_fullscreen()
        self.state = "run"
        self.filter.reset()
        if auto:
            self.cursor.set_enabled(True)

    # ------------------------------------------------------------------ debug view
    def render_debug(self, frame, info):
        H, W = frame.shape[:2]
        cam_w = 640
        s = cam_w / W
        cam_h = int(H * s)
        panel_w = 380
        canvas_h = max(cam_h + 238, 584)
        canvas = np.zeros((canvas_h, cam_w + panel_w, 3), np.uint8)
        canvas[:] = C_PANEL

        clean = frame
        cam = cv2.resize(frame, (cam_w, cam_h))
        if info is not None:
            for e in info["eyes"]:
                cv2.polylines(cam, [(e["contour"] * s).astype(np.int32)], True, C_CYAN, 1, cv2.LINE_AA)
                ic = tuple(int(v) for v in e["iris"] * s)
                cv2.circle(cam, ic, max(2, int(e["r"] * s)), C_GREEN, 1, cv2.LINE_AA)
                cv2.circle(cam, ic, 2, C_RED, -1)
            # head direction arrow from nose tip
            nose = info["pts"][NOSE_TIP] * s
            tip = nose + np.array([info["hx"], -info["hy"]]) * 120
            cv2.arrowedLine(cam, tuple(nose.astype(int)), tuple(tip.astype(int)), C_YELLOW, 2, cv2.LINE_AA)
            if info["blink"] > BLINK_THRESH:
                put(cam, "BLINK - holding", (10, cam_h - 15), 0.7, C_YELLOW, 2, bg=(0, 0, 0))
        else:
            put(cam, "NO FACE", (cam_w // 2 - 70, cam_h // 2), 1.0, C_RED, 2, bg=(0, 0, 0))
        if self.voice_busy:
            cv2.rectangle(cam, (0, 0), (cam_w - 1, cam_h - 1), C_MAGENTA, 4)
            put(cam, f"VOICE AGENT ({self.voice.phase_label()}) - ALL INPUT PAUSED", (12, 34), 0.7, C_WHITE, 2,
                bg=(140, 40, 140))
        elif self.talking:
            cv2.rectangle(cam, (0, 0), (cam_w - 1, cam_h - 1), C_RED, 4)
            put(cam, "TALKING - CLICKS PAUSED", (12, 34), 0.8, C_WHITE, 2, bg=(40, 40, 200))
        if time.time() - self.last_action_t < 1.0:
            put(cam, f"TAP: {self.last_action}", (12, 72), 0.7, C_WHITE, 2, bg=(30, 110, 30))
        canvas[:cam_h, :cam_w] = cam

        px = cam_w + 10
        # --- zoomed eyes
        put(canvas, "eyes (zoomed)", (px, 18), 0.45, C_GRAY)
        ew, eh = 175, 98
        if info is not None:
            # image-left eye first
            for i, e in enumerate(sorted(info["eyes"], key=lambda e: e["c"][0])):
                c, w = e["c"], e["w"]
                x0, x1 = int(c[0] - 0.8 * w), int(c[0] + 0.8 * w)
                y0, y1 = int(c[1] - 0.45 * w), int(c[1] + 0.45 * w)
                x0, y0 = max(x0, 0), max(y0, 0)
                x1, y1 = min(x1, W), min(y1, H)
                if x1 - x0 > 4 and y1 - y0 > 4:
                    crop = cv2.resize(clean[y0:y1, x0:x1], (ew, eh))
                    sx, sy = ew / (x1 - x0), eh / (y1 - y0)
                    ic = (int((e["iris"][0] - x0) * sx), int((e["iris"][1] - y0) * sy))
                    cv2.circle(crop, ic, max(3, int(e["r"] * sx)), C_GREEN, 1, cv2.LINE_AA)
                    cv2.drawMarker(crop, ic, C_RED, cv2.MARKER_CROSS, 10, 1)
                    cc = (int((c[0] - x0) * sx), int((c[1] - y0) * sy))
                    cv2.drawMarker(crop, cc, C_CYAN, cv2.MARKER_TILTED_CROSS, 8, 1)
                    xo = px + i * (ew + 10)
                    canvas[24:24 + eh, xo:xo + ew] = crop

        # --- iris "joystick": normalized iris offset, the raw signal that drives the cursor
        jy, js = 24 + eh + 22, 140
        put(canvas, "iris offset", (px, jy - 6), 0.45, C_GRAY)
        cv2.rectangle(canvas, (px, jy), (px + js, jy + js), C_GRAY, 1)
        cv2.line(canvas, (px + js // 2, jy), (px + js // 2, jy + js), (60, 60, 60), 1)
        cv2.line(canvas, (px, jy + js // 2), (px + js, jy + js // 2), (60, 60, 60), 1)
        if self.iris_range is not None:
            lo, hi = self.iris_range
            pad = (hi - lo) * 0.15 + 1e-6
            lo, hi = lo - pad, hi + pad
        else:
            lo, hi = np.array([-0.12, -0.08]), np.array([0.12, 0.08])

        def jpt(ix, iy):
            nx = (ix - lo[0]) / (hi[0] - lo[0])
            ny = (iy - lo[1]) / (hi[1] - lo[1])
            return (int(px + np.clip(nx, 0, 1) * js), int(jy + np.clip(ny, 0, 1) * js))

        if self.iris_range is not None:
            a, b = jpt(*self.iris_range[0]), jpt(*self.iris_range[1])
            cv2.rectangle(canvas, a, b, (70, 70, 90), 1)
        tr = list(self.iris_trail)
        for i in range(1, len(tr)):
            cv2.line(canvas, jpt(*tr[i - 1]), jpt(*tr[i]), (60, 120, 80), 1, cv2.LINE_AA)
        if info is not None:
            cv2.circle(canvas, jpt(info["ix"], info["iy"]), 5, C_GREEN, -1, cv2.LINE_AA)

        # values next to joystick
        vx = px + js + 14
        vals = []
        if info is not None:
            vals = [f"iris x  {info['ix']:+.3f}", f"iris y  {info['iy']:+.3f}",
                    f"open    {info['feat'][2]:.3f}", f"head yaw {info['hx']:+.2f}",
                    f"head pit {info['hy']:+.2f}", f"blink   {info['blink']:.2f}"]
            if info["dist_cm"]:
                vals.append(f"dist   ~{info['dist_cm']:.0f} cm")
        for i, t in enumerate(vals):
            put(canvas, t, (vx, jy + 14 + i * 19), 0.45)

        # --- screen minimap
        my = jy + js + 28
        mw = panel_w - 20
        mh = int(mw * self.sh / self.sw)
        put(canvas, "screen  (red=raw  green=smoothed  white=cursor)", (px, my - 6), 0.4, C_GRAY)
        cv2.rectangle(canvas, (px, my), (px + mw, my + mh), (50, 50, 55), -1)
        cv2.rectangle(canvas, (px, my), (px + mw, my + mh), C_GRAY, 1)
        ms = mw / self.sw

        def mpt(p):
            return (int(px + np.clip(p[0], 0, self.sw) * ms), int(my + np.clip(p[1], 0, self.sh) * ms))

        for (cx, cy) in CALIB_POINTS:
            cv2.circle(canvas, mpt((cx * self.sw, cy * self.sh)), 2, (90, 90, 90), -1)
        gt = list(self.gaze_trail)
        for i in range(1, len(gt)):
            cv2.line(canvas, mpt(gt[i - 1]), mpt(gt[i]), (60, 130, 80), 1, cv2.LINE_AA)
        if self.raw is not None:
            cv2.circle(canvas, mpt(self.raw), 3, C_RED, -1, cv2.LINE_AA)
        if self.smooth is not None:
            cv2.circle(canvas, mpt(self.smooth), 6, C_GREEN, 2, cv2.LINE_AA)
        try:
            cp = self.cursor.mouse.position
            cv2.drawMarker(canvas, mpt(cp), C_WHITE, cv2.MARKER_CROSS, 10, 1)
        except Exception:
            pass

        # --- status text under camera
        ty = cam_h + 24
        ctl = "ON" if self.cursor.enabled else "OFF"
        snap = "OFF" if not self.snap_enabled else ("LOCKED" if self.snap.current else "on")
        status = (f"state: {self.state}   cursor: {ctl}   fps: {self.fps:.0f}   "
                  f"snap: {snap}   mode: {self.actions.mode}")
        put(canvas, status, (10, ty), 0.55, C_GREEN if self.cursor.enabled else C_YELLOW)
        if 0 < self.fps < LOW_FPS_WARN:
            put(canvas, f"LOW CAMERA FPS ({self.fps:.0f}) - lower CAM_W/CAM_H or add light",
                (10, cam_h - 12), 0.55, C_WHITE, 2, bg=(40, 40, 200))
        line2 = []
        if self.calib_err is not None:
            line2.append(f"calib fit err {self.calib_err:.0f}px")
        if self.valid_stats:
            v = self.valid_stats
            t = f"validated err {v['acc_px']:.0f}px"
            if "acc_deg" in v:
                t += f" ({v['acc_deg']:.1f} deg)"
            t += f", jitter {v['prec_px']:.0f}px"
            line2.append(t)
        put(canvas, "   ".join(line2) or "not validated", (10, ty + 24), 0.5)
        put(canvas, f"smoothing: min_cutoff {self.filter.min_cutoff:.2f} Hz  [ ]    "
                    f"beta {self.filter.beta:.4f}  - =", (10, ty + 48), 0.5, C_GRAY)
        lt = self.talk.local
        talk_line = (f"talk detect [{self.talk.source}]: {'TALKING' if self.talking else 'quiet'}   "
                     f"Presage: {self.talk.presage_status()}   "
                     f"lips {lt.level:.3f}/{lt.reversals}")
        put(canvas, talk_line, (10, ty + 72), 0.45, C_RED if self.talking else C_GRAY)
        self._draw_clicks(canvas, (10, ty + 96), canvas.shape[1])
        v = self.voice
        voice_line = f'voice agent: {v.label()}   commands {v.commands}   (say "byte", then a command)'
        put(canvas, voice_line, (10, ty + 120), 0.45,
            C_MAGENTA if self.voice_busy else (C_GREEN if v.status == "running" else C_GRAY))
        put(canvas, "F4 snap  F5 clicks  F6 talk-pause  F7 cursor  F8 calib  F9 accuracy  F10 quit",
            (10, ty + 146), 0.5, C_CYAN)
        if time.time() < self.message_until:
            put(canvas, self.message, (10, ty + 172), 0.5, C_YELLOW)
        return canvas

    def _draw_clicks(self, canvas, org, width):
        c = self.clicker
        if c is None:
            put(canvas, "clicks: disabled (--no-clicks)", org, 0.45, C_GRAY)
            return
        state = "ON" if self.clicks_enabled else "OFF"
        if c.status != "running":
            text = f"clicks [{c.port or '?'}]: {c.status}"
        else:
            rate = f"{c.raw_rate / 1000:.0f}k/s" if c.raw_rate >= 1000 else f"{c.raw_rate:.0f}/s"
            text = (f"clicks [{c.port} {rate}]: {state}   taps {c.det.taps}   "
                    f"blocked {self.blocked}   last: {self.last_action or '-'}")
        color = (C_RED if (c.status != "running" or self.clicks_paused)
                 else (C_GREEN if self.clicks_enabled else C_YELLOW))
        put(canvas, text, org, 0.45, color)
        # sparkline of the tap signal (red line = threshold), right-aligned on the same row
        trail = list(c.sig_trail)[-150:]
        x0, y0, w, h = width - 160, org[1] - 14, 150, 18
        if trail:
            top = max(TAP_THRESHOLD * 2, max(trail))
            pts = np.array([(x0 + i * w / 150, y0 + h - min(v, top) / top * h)
                            for i, v in enumerate(trail)], np.int32)
            cv2.polylines(canvas, [pts], False, C_CYAN, 1, cv2.LINE_AA)
            ly = int(y0 + h - TAP_THRESHOLD / top * h)
            cv2.line(canvas, (x0, ly), (x0 + w, ly), C_RED, 1)

    # ------------------------------------------------------------------ main loop
    def run(self):
        t_prev = time.time()
        try:
            while True:
                for ev in self._take_events():
                    if ev == "quit":
                        return
                    if ev == "snap":
                        self.snap_enabled = not self.snap_enabled
                        self.flash(f"Snap to buttons {'ON' if self.snap_enabled else 'OFF'}", 3)
                        if self.snapper.status not in ("ready", "starting"):
                            self.flash(f"Snap: {self.snapper.status}", 6)
                    elif ev == "clicks":
                        if self.clicker:
                            self.clicks_enabled = not self.clicks_enabled
                            if not self.clicks_enabled:
                                self.actions.cancel()
                            self.flash(f"Jaw-tap clicks {'ON' if self.clicks_enabled else 'OFF'}", 3)
                        else:
                            self.flash("Clicks disabled (--no-clicks)", 3)
                    elif ev == "talk":
                        self.talk.enabled = not self.talk.enabled
                        self.flash(f"Pause clicks while talking {'ON' if self.talk.enabled else 'OFF'}", 3)
                    elif ev == "toggle":
                        if self.model.ready:
                            self.cursor.set_enabled(not self.cursor.enabled)
                            self.filter.reset()
                        else:
                            self.flash("Calibrate first (F8)", 4)
                    elif ev == "calib":
                        self.reset_inputs()
                        self.state = "calib"
                        self.calibrate()
                        self.reset_inputs()
                    elif ev == "valid":
                        self.reset_inputs()
                        self.state = "valid"
                        self.validate()
                        self.reset_inputs()

                frame, info = self.grab()
                if frame is None:
                    self.flash("Camera frame failed", 2)
                    cv2.waitKey(10)
                    continue
                now = time.time()
                dt = now - t_prev
                t_prev = now
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(dt, 1e-3))
                self.update_gaze(info, now)
                self.handle_taps(now)
                self.steer(now)

                canvas = self.render_debug(frame, info)
                self.open_debug(canvas.shape)
                cv2.imshow(self.DBG, canvas)
                k = cv2.waitKey(1) & 0xFF
                if self.voice_busy:
                    k = 0xFF     # the agent may be typing into this window: never treat it as q/[/]/-/=
                if k == ord("q"):
                    return
                if k == ord("["):
                    self.filter.min_cutoff = max(0.05, self.filter.min_cutoff / 1.25)
                elif k == ord("]"):
                    self.filter.min_cutoff = min(10.0, self.filter.min_cutoff * 1.25)
                elif k == ord("-"):
                    self.filter.beta = max(0.0, self.filter.beta / 1.5)
                elif k == ord("="):
                    self.filter.beta = min(0.2, max(self.filter.beta, 1e-4) * 1.5)
        finally:
            self.cursor.alive = False
            self.listener.stop()
            self.talk.close()
            self.voice.close()
            if self.clicker:
                self.clicker.close()
            self.actions.cancel()        # never leave the left button held on exit
            self.snapper.close()
            self.overlay.close()
            self.cap.release()
            cv2.destroyAllWindows()

    def _take_events(self):
        ev = []
        while not self.keys.empty():
            e = self.keys.get_nowait()
            if e in ("quit", "toggle", "calib", "valid", "talk", "clicks", "snap"):
                ev.append(e)
        return ev


def main():
    ap = argparse.ArgumentParser(description="Hands-free mouse: eye cursor + jaw-tap clicks")
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--port", help="Arduino COM port (default: auto-detect)")
    ap.add_argument("--no-clicks", action="store_true", help="eye cursor only, no Arduino")
    ap.add_argument("--no-voice", action="store_true", help="don't start/use the voice agent")
    args = ap.parse_args()
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Missing model file: {MODEL_PATH}")
    App(args.camera, port=args.port, clicks=not args.no_clicks, voice=not args.no_voice).run()


if __name__ == "__main__":
    main()
