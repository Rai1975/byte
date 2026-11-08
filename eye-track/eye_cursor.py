"""
Eye-tracking cursor with a live debug visualization.

Pipeline
  webcam -> MediaPipe FaceLandmarker (478 landmarks incl. irises, head pose, blendshapes)
         -> features: iris offset inside each eye (rotation/scale normalized), eye openness,
            head yaw/pitch
         -> ridge-regularized polynomial regression fitted on a 13-point calibration
         -> One Euro filter (low jitter when still, responsive when moving)
         -> 120 Hz cursor driver (pynput) that glides toward the filtered gaze point

While the user is talking, ALL mouse input is paused (talk_detect.py: Presage TALKING metric,
with a lip-movement fallback). Any new input (e.g. clicks) must check `App.inputs_paused`.

Hotkeys (global, work even when another window has focus)
  F6  toggle pause-while-talking on/off
  F7  toggle cursor control on/off
  F8  full calibration: dots + moving dot, then test, repeated until error < TARGET_PX
  F11 add another moving-dot round to the existing data, then test
  F4  recenter: look at the middle dot, the cursor jumps to the center and the gaze zero-point is re-aligned
  F12 posture-shift check (lean closer / sit sideways), scored separately, never used for fitting
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

# Many small matrix solves: BLAS multithreading only adds overhead (3x slower fits measured), so use 1
# thread. Must run before numpy is imported.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
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

from gaze_model import GazeModel, HeadComp, RIDGE, fit_pool, load_prior, robust_mask
import samples as sessions
from runtime_filters import FixationHold, OneEuro, OnlineRefiner
from speech import Speaker
from talk_detect import TalkGate

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "face_landmarker.task")
CALIB_PATH = os.path.join(HERE, "calibration.json")
PRIOR_PATH = os.path.join(HERE, "prior.json")   # optional population prior (build_prior.py); absent = zeros

# ----------------------------------------------------------------------------- tuning
CAM_W, CAM_H, CAM_FPS = 1280, 720, 30
SETTLE_S = 0.7          # fixed wait before sampling a dot (late settling is dropped adaptively at fit time)
COLLECT_S = 2.0         # sampling time per calibration dot
MIN_SAMPLES = 20        # minimum good frames per dot
BLINK_THRESH = 0.45     # blendshape eyeBlink score above which the cursor freezes
ONE_EURO_MIN_CUTOFF = 0.6   # Hz; lower = steadier when still
ONE_EURO_BETA = 0.004       # higher = less lag during fast movements
ONE_EURO_D_CUTOFF = 1.0
TARGET_PX = 35          # tune(): stop when the measured mean error is below this
MAX_ROUNDS = 5          # tune(): hard cap on calibrate/test rounds
PURSUIT_S = 25.0        # length of the moving-dot calibration (head still)
MIXED_S = 20.0          # moving dot while the head moves freely
PURSUIT_RAMP_S = 2.0    # the dot eases up from rest over this long
PURSUIT_FX, PURSUIT_FY = 0.11, 0.16     # Lissajous frequencies (Hz) of the moving dot
PURSUIT_LAGS = np.arange(0.0, 0.41, 0.05)   # candidate eye-lag behind the dot (s)
HEAD_POINTS = [(0.5, 0.5), (0.2, 0.3), (0.8, 0.7)]   # static dots for the head-movement phase
HEAD_S = 17.5           # seconds per static dot
HEAD_PROMPT_S = 2.5     # seconds per head-movement instruction
HEAD_PROMPTS = ["Move your head CLOSER to the screen", "TILT your head left, then right",
                "SPIN (turn) your head left, then right", "Move your head BACK a little",
                "NOD your head up, then down", "SLIDE your head left, then right, without turning",
                "Sit TALL, then slouch down"]
CURSOR_HZ = 120
# fixation hold (px values are for a 1920-px-wide screen and scale with the actual width)
FIX_V_ENTER, FIX_V_EXIT, FIX_RADIUS, FIX_MIN_S = 200.0, 500.0, 90.0, 0.15   # tuned with eval_runtime.py
CURSOR_GLIDE = 0.35     # fraction of remaining distance covered per cursor tick
CAM_HFOV_DEG = 70       # typical webcam horizontal field of view (only used for distance estimate)
IRIS_DIAMETER_MM = 11.7 # nearly constant across adults -> gives camera distance from iris size

# 5x5 grid reaching near the edges, visited in a serpentine order (25 dots)
_CX = (0.04, 0.27, 0.5, 0.73, 0.96)
_CY = (0.06, 0.28, 0.5, 0.72, 0.94)
CALIB_POINTS = [(x, y) for j, y in enumerate(_CY) for x in (_CX if j % 2 == 0 else _CX[::-1])]
POSTURE_POINTS = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.5), (0.2, 0.8), (0.8, 0.8)]   # posture-shift check
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

    hx = hy = tx = ty = tz = roll = 0.0
    dist_cm = None
    if result.facial_transformation_matrixes:
        M = np.asarray(result.facial_transformation_matrixes[0])
        hx, hy = float(M[0, 2]), float(M[1, 2])   # face forward vector, x/y components (y up)
        tx, ty, tz = float(M[0, 3]), float(M[1, 3]), float(M[2, 3])   # head position (cm, camera-ish space)
        roll = math.atan2(float(M[1, 0]), float(M[0, 0]))             # head tilt (rad)
    # distance from apparent iris size, assuming a typical webcam field of view
    focal_px = (W / 2) / math.tan(math.radians(CAM_HFOV_DEG / 2))
    r_px = max(e["r"] for e in eyes)
    dist_cm = focal_px * (IRIS_DIAMETER_MM / 2) / max(r_px, 1e-3) / 10
    blink = 0.0
    bs_h = bs_up = bs_down = 0.0
    if result.face_blendshapes:
        bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
        blink = (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2
        g = bs.get
        # eyeLook*: how far each eye looks in/out/up/down, learned from the whole eye image
        bs_h = ((g("eyeLookInLeft", 0) + g("eyeLookOutRight", 0)) - (g("eyeLookOutLeft", 0) + g("eyeLookInRight", 0))) / 2
        bs_up = (g("eyeLookUpLeft", 0) + g("eyeLookUpRight", 0)) / 2
        bs_down = (g("eyeLookDownLeft", 0) + g("eyeLookDownRight", 0)) / 2

    # iris offset in the HEAD frame: the 3D eye-corner -> iris vector rotated by the inverse head rotation,
    # so head yaw/pitch no longer mix into the iris signal (needs the face transform matrix)
    ih = np.zeros(3)
    if result.facial_transformation_matrixes:
        Rm = np.asarray(result.facial_transformation_matrixes[0])[:3, :3]
        Rm = Rm / (np.linalg.norm(Rm, axis=0) + 1e-9)
        zs = np.array([p.z for p in lms]) * W
        for e, spec in zip(eyes, EYES):
            dz = zs[list(spec["iris"])].mean() - zs[list(spec["corners"])].mean()
            d_gl = np.array([e["iris"][0] - e["c"][0], -(e["iris"][1] - e["c"][1]), -dz])   # image -> OpenGL axes
            ih += (Rm.T @ d_gl) / e["w"] * (e["w"] / wsum)

    return dict(feat=np.array([ix, iy, op, hx, hy, tx, ty, tz, roll, bs_h, bs_up, bs_down, *ih]), ix=ix, iy=iy, eyes=eyes, pts=pts,
                hx=hx, hy=hy, tx=tx, ty=ty, tz=tz, roll=roll, dist_cm=dist_cm, blink=blink)




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
        self.paused = False     # True while the user is talking: cursor stays exactly where it is
        self.alive = True
        self._pos = None

    def set_enabled(self, on):
        if on and not self.enabled:
            self._pos = np.array(self.mouse.position, dtype=float)
        self.enabled = on

    def jump(self, x, y):
        """Place the OS cursor at (x, y) immediately (used by recenter)."""
        self._pos = np.array([x, y], dtype=float)
        self.target = (float(x), float(y))
        try:
            self.mouse.position = (int(x), int(y))
        except Exception:
            pass

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
    DBG = "Eye Tracker - debug"
    CAL = "Eye Tracker - calibration"

    def __init__(self, cam_index, voice=True, fixation=True, online_refine=False):
        self.voice = Speaker(enabled=voice)
        self.sw, self.sh, self.width_mm = screen_info()
        self.pool_F, self.pool_S, self.pool_G = [], [], []   # training data pooled over all rounds
        self.pool_K, self.pool_V = [], []                    # kind ('eye'/'head') and visit id per chunk
        self.pool_T, self.pool_P, self.pool_R = [], [], []   # time, phase code, round per chunk
        self.session_path = None
        self.contig_groups = set()                           # group ids that are consecutive time blocks
        self.headcomp = HeadComp()
        self._gid, self.rounds, self.history = 0, 0, []
        self.px_per_mm = (self.sw / self.width_mm) if self.width_mm else None

        # MSMF delivers 30 fps at 1280x720 here; DirectShow gave only ~10 fps at that size (measured),
        # which means 3x fewer calibration samples and 3x more lag. DirectShow stays as the fallback.
        self.cap = None
        for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
            cap = cv2.VideoCapture(cam_index, backend)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
            cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
            # Camera drivers REMEMBER exposure settings across programs: a tool that left manual exposure
            # on made the image nearly black (brightness 30/255) and the iris signal vanish. Force auto.
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            if cap.isOpened() and cap.read()[0]:
                self.cap = cap
                break
            cap.release()
        if self.cap is None:
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
        self.prior = load_prior(PRIOR_PATH)
        self.filter = OneEuro(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_D_CUTOFF)
        k = self.sw / 1920.0
        self.fixation = (FixationHold(FIX_V_ENTER * k, FIX_V_EXIT * k, FIX_RADIUS * k, FIX_MIN_S)
                         if fixation else None)
        self.refiner = OnlineRefiner(self.sw, self.sh) if online_refine else None   # OFF unless --online-refine
        self.raw0 = None        # latest raw prediction before the online correction
        self.offset = np.zeros(2)   # recenter zero-point shift (px), see recenter(); per sitting, never saved
        self.cursor = CursorDriver(self.sw, self.sh)
        self.cursor.start()

        # Talking pauses ALL mouse input (cursor now; clicks must check `inputs_paused` too).
        self.talk = TalkGate()
        self.talking = False

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
        self.posture_stats = None
        self.message = ""
        self.message_until = 0
        self.dbg_open = False
        self.cal_bg = np.zeros((self.sh, self.sw, 3), np.uint8)
        self.cal_bg[:] = C_DARK

        if self._load_calibration():
            self._load_session()
            self.state = "run"
            self.flash("Loaded saved calibration. F7 = cursor on/off, F8 = recalibrate", 8)
        else:
            self.flash("Not calibrated yet. Press F8 to calibrate.", 3600)

    # ------------------------------------------------------------------ input
    def _on_global_key(self, key):
        names = {pkeyboard.Key.f6: "talk", pkeyboard.Key.f7: "toggle", pkeyboard.Key.f8: "calib", pkeyboard.Key.f9: "valid",
                 pkeyboard.Key.f10: "quit", pkeyboard.Key.f11: "refine", pkeyboard.Key.f4: "recenter", pkeyboard.Key.f12: "posture", pkeyboard.Key.space: "space", pkeyboard.Key.esc: "esc"}
        if key in names:
            self.keys.put(names[key])

    def _reset_filters(self):
        self.filter.reset()
        if self.fixation is not None:
            self.fixation.reset()

    def flash(self, msg, seconds=4):
        self.message, self.message_until = msg, time.time() + seconds

    # ------------------------------------------------------------------ persistence
    def _save_session(self):
        """Write every raw sample of this session to eye-track/sessions/ (see samples.py)."""
        if not self.pool_F:
            return
        if self.session_path is None:
            self.session_path = sessions.new_path()
        try:
            sessions.save(self.session_path, self._pool_arrays(),
                          dict(screen=[self.sw, self.sh], features=GazeModel.FEATURES, rounds=self.rounds,
                               mediapipe=getattr(mp, "__version__", "?"), history_px=self.history))
        except Exception as e:                       # never break calibration over a disk problem
            print("could not save session:", e)

    def _load_session(self):
        """Restore the newest saved session into the pool, so F11 can add rounds after a restart."""
        path = sessions.latest()
        if not path:
            return False
        try:
            A, meta = sessions.load(path)
            if meta.get("screen") != [self.sw, self.sh] or A["F"].shape[1] != len(GazeModel.FEATURES):
                return False
            self.pool_F, self.pool_S, self.pool_G = [A["F"]], [A["S"]], [A["G"]]
            self.pool_V, self.pool_K, self.pool_T = [A["V"]], [A["K"]], [A["T"]]
            self.pool_P, self.pool_R = [A["PH"]], [A["R"]]
            self.contig_groups = set(int(g) for g in np.unique(A["G"][A["C"]]))
            self._gid = int(max(A["G"].max(), A["V"].max())) + 2
            self.rounds = int(meta.get("rounds", A["R"].max()))
            self.history = list(meta.get("history_px", []))
            self.session_path = path
            return True
        except Exception as e:
            print("could not load session:", e)
            return False

    def _save_calibration(self):
        d = dict(model=self.model.to_dict(), headcomp=self.headcomp.to_dict(), screen=[self.sw, self.sh],
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
            if d.get("headcomp"):
                self.headcomp.from_dict(d["headcomp"])
            if len(self.model.mu) != len(GazeModel.FEATURES):
                self.model.axes = None      # saved with a different feature set: needs recalibration
                return False
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
        return frame, info

    @property
    def inputs_paused(self):
        """True while the user is talking. Every input (cursor, clicks) must respect this."""
        return self.talking

    def update_gaze(self, info, now):
        self.cursor.paused = self.inputs_paused
        if info is None or not self.model.ready:
            return
        self.iris_trail.append((info["ix"], info["iy"]))
        if self.inputs_paused:
            return  # talking: freeze the cursor; mouth movement also distorts the gaze features
        if info["blink"] > BLINK_THRESH:
            return  # hold position through blinks
        raw = self.gaze(info["feat"])
        raw = np.clip(raw, [-0.05 * self.sw, -0.05 * self.sh], [1.05 * self.sw, 1.05 * self.sh])
        raw = raw + self.offset                  # recenter zero-point
        self.raw0 = raw
        if self.refiner is not None:
            raw = raw + self.refiner.correction(raw)
        self.raw = raw
        self.smooth = self.filter(raw, now)
        out = self.smooth
        if self.fixation is not None:
            out, _ = self.fixation.update(self.smooth, now)     # pin the cursor during fixations
        self.gaze_trail.append(tuple(out))
        self.cursor.target = (float(np.clip(out[0], 0, self.sw - 1)),
                              float(np.clip(out[1], 0, self.sh - 1)))

    def register_click(self, x, y):
        """Call when the user clicks at a location that is independent of the gaze cursor (see
        OnlineRefiner). Only does anything with --online-refine. Respects inputs_paused."""
        if self.refiner is not None and self.raw0 is not None and not self.inputs_paused:
            self.refiner.update(self.raw0, (x, y))

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
    def run_targets(self, points, mode, title=None, sub=None):
        """Show dots one by one. mode='calib' collects features, mode='valid' collects predictions.
        Returns list of (target_px, samples, times) or None if cancelled."""
        self.open_fullscreen()
        was_enabled = self.cursor.enabled
        self.cursor.set_enabled(False)
        title_ = "CALIBRATION" if mode == "calib" else "ACCURACY CHECK"
        sub_ = ("Keep your head still and follow the dot with your EYES only."
               if mode == "calib" else "Look at each dot. This measures how accurate tracking is.")
        title, sub = title or title_, sub or sub_

        if not self._intro(title, sub):
            self.close_fullscreen()
            self.cursor.set_enabled(was_enabled)
            return None

        out = []
        for (px, py) in points:
            target = np.array([px * self.sw, py * self.sh])
            samples, times = [], []
            t0 = time.time()
            while True:
                now = time.time()
                el = now - t0
                frame, info = self.grab()
                good = info is not None and info["blink"] < BLINK_THRESH and not self.talking
                collecting = el >= SETTLE_S
                if collecting and good:
                    samples.append(info["feat"])    # raw features; predictions are made later
                    times.append(now)
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
            out.append((target, samples, times))
        self.cursor.set_enabled(was_enabled)
        return out

    def _intro(self, title, sub):
        """Intro screen with a camera preview. True on SPACE, False on ESC. Read aloud."""
        self.voice.say(f"{title.title()}. {sub} Press space to start, or escape to cancel.")
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
                return False
            if k == 32 or "space" in ev:
                return True

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

    # ------------------------------------------------------------------ training pool
    def gaze(self, feat):
        """Raw feature vector -> screen px (head compensation, then the eye model)."""
        return self.model.predict(self.headcomp.apply(feat))

    def _pool_add(self, F, S, g, kind="eye", v=None, contiguous=False, T=None, phase="dots"):
        """Append samples to the training pool. g = local group ids (CV groups), v = local visit ids
        (head phase: one per static dot). kind = 'eye' (head still) or 'head' (static target, head moving).
        contiguous = the groups are consecutive time blocks (neighbours are excluded from training when
        one is held out in cross-validation)."""
        F, S, g = np.asarray(F, float), np.asarray(S, float), np.asarray(g, int)
        if len(F) == 0:
            return
        v = g if v is None else np.asarray(v, int)
        base = self._gid
        g = g - g.min() + base
        v = v - v.min() + base
        self._gid = int(max(g.max(), v.max())) + 1
        if contiguous:
            self.contig_groups.update(int(x) for x in np.unique(g))
            self._gid += 1          # keep the next chunk from looking adjacent
        self.pool_F.append(F)
        self.pool_S.append(S)
        self.pool_G.append(g)
        self.pool_V.append(v)
        self.pool_K.append(np.full(len(F), kind == "head"))
        self.pool_T.append(np.zeros(len(F)) if T is None else np.asarray(T, float))
        self.pool_P.append(np.full(len(F), sessions.PHASES.index(phase), np.int8))
        self.pool_R.append(np.full(len(F), self.rounds, np.int16))

    def _pool_add_dots(self, res, phase="dots"):
        """res = [(target_px, [feature vectors], [times])]. Returns number of usable dots."""
        F, S, G, T, n = [], [], [], [], 0
        for target, samples, times in res:
            if len(samples) < 5:
                continue
            samples = np.asarray(samples)
            keep = robust_mask(samples)
            F.extend(samples[keep])
            T.extend(np.asarray(times)[keep])
            S.extend([target] * int(keep.sum()))
            G.extend([n] * int(keep.sum()))
            n += 1
        if n:
            self._pool_add(F, S, G, T=T, phase=phase)
        return n

    def _pool_arrays(self):
        cat = np.concatenate
        G = cat(self.pool_G)
        return dict(F=np.vstack(self.pool_F), S=np.vstack(self.pool_S), G=G, V=cat(self.pool_V),
                    K=cat(self.pool_K), T=cat(self.pool_T), PH=cat(self.pool_P), R=cat(self.pool_R),
                    C=np.isin(G, list(self.contig_groups)))

    def refit(self, nested=False):
        A = self._pool_arrays()
        F, S, G = A["F"], A["S"], A["G"]
        self.headcomp, self.model = fit_pool(A, (self.sw, self.sh), nested=nested, model=self.model,
                                             prior=self.prior)
        self.iris_range = (F[:, :2].min(0), F[:, :2].max(0))
        sub = slice(None, None, max(1, len(F) // 400))
        pred = np.array([self.gaze(f) for f in F[sub]])
        self.calib_err = float(np.sqrt(np.mean(np.sum((pred - S[sub]) ** 2, 1))))
        self._reset_filters()
        self._save_calibration()
        self._save_session()

    # ------------------------------------------------------------------ head-movement phase
    def run_headphase(self):
        """Look at a STATIC dot while moving the head (closer, tilt, spin, back, nod). The gaze target
        never changes, so every change in the iris signal is caused by head movement.
        Returns (F, S, G, V) or None if cancelled."""
        self.open_fullscreen()
        was_enabled = self.cursor.enabled
        self.cursor.set_enabled(False)
        if not self._intro("HEAD MOVEMENT",
                           "Keep your EYES on the dot the whole time and move only your head, as told."):
            self.close_fullscreen()
            self.cursor.set_enabled(was_enabled)
            return None
        F, S, G, V, T = [], [], [], [], []
        lead = 1.5
        spoken = None
        for vi, (px, py) in enumerate(HEAD_POINTS):
            target = np.array([px * self.sw, py * self.sh])
            tx, ty = int(target[0]), int(target[1])
            t0 = time.time()
            while True:
                el = time.time() - t0
                if el >= lead + HEAD_S:
                    break
                frame, info = self.grab()
                t = el - lead
                good = info is not None and info["blink"] < BLINK_THRESH and not self.talking
                img = self.cal_bg.copy()
                if t < 0:
                    cv2.circle(img, (tx, ty), int(12 + 40 * (-t / lead)), C_WHITE, 2, cv2.LINE_AA)
                    cv2.circle(img, (tx, ty), 6, C_WHITE, -1, cv2.LINE_AA)
                    prompt = "Look at the dot..."
                    if spoken != ("lead", vi):
                        spoken = ("lead", vi)
                        self.voice.say("Look at the dot, and keep your eyes on it.")
                else:
                    if good:
                        seg = int(t // HEAD_PROMPT_S)
                        F.append(info["feat"])
                        T.append(time.time())
                        S.append(target)
                        G.append(vi * 100 + seg)
                        V.append(vi)
                    cv2.circle(img, (tx, ty), 14, C_GREEN, -1, cv2.LINE_AA)
                    cv2.circle(img, (tx, ty), 3, (0, 0, 0), -1, cv2.LINE_AA)
                    pi = min(int(t // HEAD_PROMPT_S), len(HEAD_PROMPTS) - 1)
                    prompt = HEAD_PROMPTS[pi]
                    if spoken != (vi, pi):
                        spoken = (vi, pi)
                        self.voice.say(prompt)
                    cv2.rectangle(img, (0, self.sh - 8), (int(self.sw * t / HEAD_S), self.sh), C_GREEN, -1)
                put(img, prompt, (self.sw // 2 - 380, int(self.sh * 0.1)), 1.3, C_YELLOW, 3)
                put(img, "keep your EYES on the dot", (self.sw // 2 - 200, int(self.sh * 0.1) + 50), 0.8, C_GRAY, 2)
                put(img, f"{vi + 1}/{len(HEAD_POINTS)}", (30, self.sh - 30), 0.8, C_GRAY, 2)
                if info is None:
                    put(img, "FACE NOT DETECTED", (self.sw // 2 - 200, int(self.sh * 0.25)), 1.2, C_RED, 3)
                cv2.imshow(self.CAL, img)
                k = cv2.waitKey(1) & 0xFF
                if k == 27 or "esc" in self._drain_keys():
                    self.close_fullscreen()
                    self.cursor.set_enabled(was_enabled)
                    return None
        self.cursor.set_enabled(was_enabled)
        if len(F) < 200:
            self.flash("Head phase failed: face not tracked enough.", 8)
            return None
        return np.array(F), np.array(S), np.array(G), np.array(V), np.array(T)

    # ------------------------------------------------------------------ moving-dot pursuit
    @staticmethod
    def _warp(t):
        """Time warp so the dot eases up from rest instead of jumping into motion."""
        T = PURSUIT_RAMP_S
        return t * t / (2 * T) if t < T else t - T / 2

    def _traj(self, tau, rnd):
        """Lissajous path over (nearly) the whole screen; a different phase every round."""
        ph = 1.7 * rnd
        x = 0.5 + 0.45 * math.sin(2 * math.pi * PURSUIT_FX * tau + ph)
        y = 0.5 + 0.43 * math.sin(2 * math.pi * PURSUIT_FY * tau + 2.1 * ph + 0.6)
        return np.array([x * self.sw, y * self.sh])

    def run_pursuit(self, rnd, mixed=False, lag_hint=None):
        """Follow a moving dot for PURSUIT_S seconds. Returns (F, S, G, lag) or None if cancelled.
        Eyes trail the dot by roughly 100-250 ms, so the lag is estimated from the data: the lag at
        which a simple fit of features -> dot position has the smallest error."""
        self.open_fullscreen()
        was_enabled = self.cursor.enabled
        self.cursor.set_enabled(False)
        dur = MIXED_S if mixed else PURSUIT_S
        rnd = rnd + (0.5 if mixed else 0.0)          # a different path from the head-still pursuit
        msg = (f"Follow the moving dot with your eyes, and also move your head freely and naturally: "
               f"nod, turn, lean ({dur:.0f} seconds)." if mixed else
               f"Follow the moving dot smoothly with your EYES only, head still ({dur:.0f} s).")
        if not self._intro("MOVING DOT, HEAD FREE" if mixed else "MOVING DOT", msg):
            self.close_fullscreen()
            self.cursor.set_enabled(was_enabled)
            return None
        rec = []
        t0 = time.time()
        lead = 2.0                                   # stationary "get ready" phase
        started = False
        while True:
            el = time.time() - t0
            if el >= lead + dur:
                break
            frame, info = self.grab()
            t = el - lead
            pos = self._traj(self._warp(max(t, 0.0)), rnd)
            good = info is not None and info["blink"] < BLINK_THRESH and not self.talking
            if t >= 0 and not started:
                started = True
                self.voice.say("Go. Follow the dot, and move your head naturally." if mixed else "Go. Follow the dot with your eyes.")
            if t >= 0 and good:
                rec.append((t, info["feat"]))
            img = self.cal_bg.copy()
            tx, ty = int(pos[0]), int(pos[1])
            if t < 0:
                cv2.circle(img, (tx, ty), int(12 + 40 * (-t / lead)), C_WHITE, 2, cv2.LINE_AA)
                cv2.circle(img, (tx, ty), 6, C_WHITE, -1, cv2.LINE_AA)
            else:
                cv2.circle(img, (tx, ty), 14, C_GREEN, -1, cv2.LINE_AA)
                cv2.circle(img, (tx, ty), 3, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.rectangle(img, (0, self.sh - 8), (int(self.sw * t / dur), self.sh), C_GREEN, -1)
            if info is None:
                put(img, "FACE NOT DETECTED", (self.sw // 2 - 200, int(self.sh * 0.18)), 1.2, C_RED, 3)
            cv2.imshow(self.CAL, img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or "esc" in self._drain_keys():
                self.close_fullscreen()
                self.cursor.set_enabled(was_enabled)
                return None
        self.cursor.set_enabled(was_enabled)
        if len(rec) < 200:
            self.flash("Pursuit failed: face not tracked enough. Check lighting.", 8)
            return None

        ts = np.array([r[0] for r in rec])
        Fm = np.array([r[1] for r in rec])
        Z = (Fm - Fm.mean(0)) / (Fm.std(0) + 1e-9)
        X = np.column_stack([np.ones(len(Z)), Z])
        best = None
        for lag in (PURSUIT_LAGS if lag_hint is None else [lag_hint]):
            S = np.array([self._traj(self._warp(max(t - lag, 0.0)), rnd) for t in ts])
            W = GazeModel._solve(X, S, 1e-3)
            r = np.linalg.norm(X @ W - S, axis=1)
            score = float(np.sqrt(np.mean(r ** 2)))
            if best is None or score < best[0]:
                best = (score, lag, S, r)
        _, lag, S, r = best
        keep = r <= np.quantile(r, 0.9)              # drop catch-up saccades / blink aftermath
        keep &= ts >= 0.5
        G = (ts // 2.0).astype(int)                  # 2 s segments = CV groups
        return Fm[keep], S[keep], G[keep], float(lag), (t0 + lead + ts)[keep]

    # ------------------------------------------------------------------ measuring
    def _measure(self):
        """Accuracy check with the current model. Returns (rows, stats, raw_result) or None."""
        res = self.run_targets(VALID_POINTS, "valid")
        if res is None:
            return None
        rows = []
        for target, feats, _ in res:
            if len(feats) < 3:
                continue
            P = np.array([self.gaze(f) for f in feats])
            m = P.mean(0)
            rows.append((target, m, float(np.linalg.norm(m - target)),
                         float(np.sqrt(np.mean(np.sum((P - m) ** 2, 1))))))
        if not rows:
            return None
        acc = float(np.mean([r[2] for r in rows]))
        prec = float(np.mean([r[3] for r in rows]))
        stats = dict(acc_px=acc, prec_px=prec)
        if self.px_per_mm:
            stats["acc_cm"] = acc / self.px_per_mm / 10
            _, info = self.grab()
            dist = info["dist_cm"] if info else None
            if dist:
                stats["acc_deg"] = math.degrees(math.atan(acc / self.px_per_mm / (dist * 10)))
        return rows, stats, res

    def _results_screen(self, rows, stats, footer, extra=(), timeout=None):
        """Draw the accuracy result. Returns 'next' (SPACE), 'stop' (ESC) or 'calib' (F8)."""
        acc, prec = stats["acc_px"], stats["prec_px"]
        self.voice.say(f"Accuracy check done. Mean error {acc:.0f} pixels. {footer.split('|')[0].replace(' = ', ' for ').strip()}")
        t0 = time.time()
        while timeout is None or time.time() - t0 < timeout:
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
            for i, (txt, col) in enumerate(extra):
                put(img, txt, (cx - 520, cy + 115 + i * 30), 0.7, col)
            put(img, footer, (cx - 520, cy + 125 + 30 * len(extra)), 0.8, C_YELLOW, 2)
            cv2.imshow(self.CAL, img)
            k = cv2.waitKey(30) & 0xFF
            ev = self._drain_keys()
            if k == 27 or "esc" in ev:
                return "stop"
            if k == 32 or "space" in ev:
                return "next"
            if "calib" in ev:
                return "calib"
        return "next"

    # ------------------------------------------------------------------ calibration rounds
    def tune(self, fresh=True):
        """Calibrate -> test -> calibrate again, pooling the data of every round, until the measured
        error is below TARGET_PX (or MAX_ROUNDS / ESC). Each round = a moving dot (eyes only, head
        still) + a static dot (head only, eyes fixed); round 1 also does the 25 dots. Each accuracy
        check's own points are added to the pool afterwards."""
        self.offset = np.zeros(2)               # a new model has a new zero-point
        if fresh or not self.pool_F:
            self.pool_F, self.pool_S, self.pool_G, self._gid, self.rounds = [], [], [], 0, 0
            self.pool_K, self.pool_V = [], []
            self.pool_T, self.pool_P, self.pool_R = [], [], []
            self.contig_groups = set()
            self.session_path = sessions.new_path()
            self.history = []
        first = not self.pool_F
        done_rounds = 0
        try:
            while True:
                self.rounds += 1
                done_rounds += 1
                if first:
                    res = self.run_targets(CALIB_POINTS, "calib")
                    if res is None or self._pool_add_dots(res, "dots") < 12:
                        self.flash("Calibration cancelled or face not tracked on enough dots", 6)
                        return
                    first = False
                # 1) EYES only, head still: moving dot
                pur = self.run_pursuit(self.rounds)
                if pur is None:
                    if self.pool_F:
                        self.refit()        # keep what the dots gave us
                    return
                F, S, G, lag, T = pur
                self._pool_add(F, S, G, contiguous=True, T=T, phase="pursuit")
                # 1b) EYES + free HEAD motion: the moving dot while the head moves naturally
                mix = self.run_pursuit(self.rounds, mixed=True, lag_hint=lag)
                if mix is None:
                    self.refit()
                    return
                self._pool_add(mix[0], mix[1], mix[2], contiguous=True, T=mix[4], phase="mixed")
                # 2) HEAD only, eyes fixed on a static dot: closer / tilt / spin / back / nod
                head = self.run_headphase()
                if head is None:
                    self.refit()
                    return
                self._pool_add(*head[:3], kind="head", v=head[3], contiguous=True, T=head[4], phase="head")
                self.refit()
                r2 = self.headcomp.r2
                print(f"[round {self.rounds}] pursuit lag {lag * 1000:.0f} ms, pool "
                      f"{sum(map(len, self.pool_F))} samples, head explains "
                      f"{'n/a' if r2 is None else f'{100 * r2:.0f}%'} of iris shift, held-out "
                      f"{self.model.cv_err} (stage 1 only {self.model.cv_base}), {self.model.summary}")

                m = self._measure()
                if m is None:
                    return
                rows, stats, res = m
                self.valid_stats = stats
                self.history.append(stats["acc_px"])
                reached = stats["acc_px"] < TARGET_PX
                # the accuracy check's own points become training data for the next fit
                self._pool_add_dots(res, "valid")
                self.refit(nested=True)     # final fit of the round: also compute the honest nested-CV error
                cv = self.model.cv_nested or self.model.cv_err
                print(f"[round {self.rounds}] selection-CV {self.model.cv_err}, nested-CV {self.model.cv_nested}")
                extra = [("rounds: " + "  ->  ".join(f"{h:.0f}px" for h in self.history), C_WHITE),
                         (f"model now: {self.model.summary}   held-out (nested CV) {cv:.0f}px" if cv else
                          f"model now: {self.model.summary}", C_GRAY)]
                if reached:
                    footer = f"TARGET REACHED (< {TARGET_PX:.0f}px).  SPACE = finish"
                elif self.rounds >= MAX_ROUNDS:
                    footer = f"Round limit reached ({MAX_ROUNDS}).  SPACE = finish"
                else:
                    footer = "SPACE = another round (adds data)   |   ESC = stop here"
                key = self._results_screen(rows, stats, footer, extra)
                if reached or self.rounds >= MAX_ROUNDS or key != "next":
                    if key == "calib":
                        self.keys.put("calib")
                    return
        finally:
            self.close_fullscreen()
            self.state = "run" if self.model.ready else "idle"
            self._reset_filters()
            if self.model.ready and done_rounds:
                self.cursor.set_enabled(True)
                self.flash(f"Tuned in {done_rounds} round(s): last error {self.history[-1]:.0f}px"
                           if self.history else "Calibrated", 10)

    def recenter(self):
        """Look at the dot in the middle of the screen and press F4: the cursor jumps to the center and the gaze
        zero-point is shifted so that where you are looking right now maps to the center. Fixes the constant
        offset that builds up when you shift in your seat or the screen/camera moves a little. Does not retrain
        anything; the shift is per sitting (it is reset by calibration and never saved)."""
        if not self.model.ready:
            self.flash("Calibrate first (F8)", 4)
            return
        c = np.array([self.sw / 2.0, self.sh / 2.0])
        self.open_fullscreen()
        was_enabled = self.cursor.enabled
        self.cursor.set_enabled(False)
        self.voice.say("Recenter. Look at the dot in the middle of the screen.")
        settle, collect = 1.0, 1.2
        P, t0, ok = [], time.time(), False
        try:
            while True:
                el = time.time() - t0
                frame, info = self.grab()
                if el >= settle and info is not None and info["blink"] < BLINK_THRESH and not self.talking:
                    P.append(self.gaze(info["feat"]))
                if el >= settle + collect:
                    ok = len(P) >= 8
                    break
                if el > settle + collect + 4.0:
                    break
                img = self.cal_bg.copy()
                rad = int(12 + 40 * max(0.0, 1 - el / settle))
                cv2.circle(img, (int(c[0]), int(c[1])), rad, C_WHITE, 2, cv2.LINE_AA)
                cv2.circle(img, (int(c[0]), int(c[1])), 12, C_GREEN if el >= settle else C_WHITE, -1, cv2.LINE_AA)
                put(img, "RECENTER: look at the dot", (self.sw // 2 - 230, int(self.sh * 0.2)), 1.1, C_CYAN, 3)
                if info is None:
                    put(img, "FACE NOT DETECTED", (self.sw // 2 - 200, int(self.sh * 0.3)), 1.2, C_RED, 3)
                cv2.imshow(self.CAL, img)
                k = cv2.waitKey(1) & 0xFF
                if k == 27 or "esc" in self._drain_keys():
                    self.flash("Recenter cancelled", 3)
                    return
            if not ok:
                self.flash("Recenter failed: face not tracked, try again", 5)
                return
            off = c - np.median(P, axis=0)                              # gaze() has no offset, so P is raw
            lim = np.array([0.4 * self.sw, 0.4 * self.sh])
            self.offset = np.clip(off, -lim, lim)
            self.flash(f"Recentered (shift {self.offset[0]:+.0f}, {self.offset[1]:+.0f} px)", 6)
            print(f"[recenter] gaze zero-point shifted by ({self.offset[0]:+.0f}, {self.offset[1]:+.0f}) px")
        finally:
            self.close_fullscreen()
            self.cursor.set_enabled(was_enabled)
            self._reset_filters()
            self.state = "run"
            if ok:
                self.cursor.jump(c[0], c[1])
                self.smooth = c.copy()

    def _score(self, res):
        """res from run_targets -> (rows, mean error px). Same scoring as the accuracy check."""
        rows = []
        for target, feats, _ in res:
            if len(feats) < 3:
                continue
            P = np.array([self.gaze(f) for f in feats])
            m = P.mean(0)
            rows.append((target, m, float(np.linalg.norm(m - target)),
                         float(np.sqrt(np.mean(np.sum((P - m) ** 2, 1))))))
        return rows, (float(np.mean([r[2] for r in rows])) if rows else None)

    def posture_check(self):
        """Honest evaluation: look at dots after deliberately changing posture/distance. Scored in px and
        reported SEPARATELY from the in-session accuracy check; saved to the session but never fitted."""
        if not self.model.ready:
            self.flash("Calibrate first (F8)", 4)
            return
        steps = [("POSTURE 1 OF 2", "Lean CLOSER to the screen, about ten centimeters, then look at each dot."),
                 ("POSTURE 2 OF 2", "Sit back, shift your chair to one side and tilt your head a little, "
                                    "then look at each dot.")]
        results = []
        try:
            for title, sub in steps:
                res = self.run_targets(POSTURE_POINTS, "valid", title=title, sub=sub)
                if res is None:
                    return
                results.append(res)
            scored = [self._score(r) for r in results]
            if any(e is None for _, e in scored):
                self.flash("Posture check: face not tracked well enough", 6)
                return
            for r in results:                                    # kept in the session file, excluded from fits
                self._pool_add_dots(r, "posture")
            self._save_session()
            rows = scored[0][0] + scored[1][0]
            errs = [e for _, e in scored]
            stats = dict(acc_px=float(np.mean(errs)), prec_px=float(np.mean([r[3] for r in rows])))
            base = self.valid_stats["acc_px"] if self.valid_stats else None
            extra = [(f"posture 1 (closer): {errs[0]:.0f}px    posture 2 (shifted/tilted): {errs[1]:.0f}px", C_WHITE),
                     ("in-session accuracy check: " + (f"{base:.0f}px" if base else "n/a")
                      + "   (these posture errors were NOT used for fitting)", C_GRAY)]
            print(f"[posture check] closer {errs[0]:.0f}px, shifted {errs[1]:.0f}px, in-session {base}")
            self.posture_stats = dict(closer=errs[0], shifted=errs[1])
            self._results_screen(rows, stats, "SPACE/ESC to continue", extra, timeout=30)
        finally:
            self.close_fullscreen()
            self.state = "run"
            self._reset_filters()

    def validate(self):
        if not self.model.ready:
            self.flash("Calibrate first (F8)", 4)
            return
        m = self._measure()
        if m is None:
            self.close_fullscreen()
            self.state = "run"
            return
        rows, stats, _ = m
        self.valid_stats = stats
        key = self._results_screen(rows, stats, "SPACE/ESC to continue   |   F8 to recalibrate", timeout=12)
        if key == "calib":
            self.keys.put("calib")
        self.close_fullscreen()
        self.state = "run"
        self._reset_filters()

    # ------------------------------------------------------------------ debug view
    def render_debug(self, frame, info):
        H, W = frame.shape[:2]
        cam_w = 640
        s = cam_w / W
        cam_h = int(H * s)
        panel_w = 380
        canvas_h = max(cam_h + 190, 560)
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
        if self.talking:
            cv2.rectangle(cam, (0, 0), (cam_w - 1, cam_h - 1), C_RED, 4)
            put(cam, "TALKING - INPUT PAUSED", (12, 34), 0.8, C_WHITE, 2, bg=(40, 40, 200))
        else:
            put(cam, "NO FACE", (cam_w // 2 - 70, cam_h // 2), 1.0, C_RED, 2, bg=(0, 0, 0))
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
                    f"head pit {info['hy']:+.2f}", f"blink   {info['blink']:.2f}",
                    f"head xyz {info['tx']:+.1f} {info['ty']:+.1f} {info['tz']:+.1f}"]
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
        status = f"state: {self.state}   cursor control: {ctl}   fps: {self.fps:.0f}"
        put(canvas, status, (10, ty), 0.55, C_GREEN if self.cursor.enabled else C_YELLOW)
        line2 = []
        if self.calib_err is not None:
            t = f"fit err {self.calib_err:.0f}px"
            if self.model.cv_err is not None:
                t += f", held-out {self.model.cv_err:.0f}px"
            line2.append(t + f" [{self.model.summary}]")
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
        put(canvas, "F6 talk-pause on/off  F7 cursor on/off  F8 calibrate  F9 accuracy  F11 +round  F12 posture  F4 recenter  F10 quit",
            (10, ty + 98), 0.5, C_CYAN)
        if time.time() < self.message_until:
            put(canvas, self.message, (10, ty + 124), 0.5, C_YELLOW)
        return canvas

    # ------------------------------------------------------------------ main loop
    def run(self):
        t_prev = time.time()
        try:
            while True:
                for ev in self._take_events():
                    if ev == "quit":
                        return
                    if ev == "talk":
                        self.talk.enabled = not self.talk.enabled
                        self.flash(f"Pause-while-talking {'ON' if self.talk.enabled else 'OFF'}", 3)
                    elif ev == "toggle":
                        if self.model.ready:
                            self.cursor.set_enabled(not self.cursor.enabled)
                            self._reset_filters()
                        else:
                            self.flash("Calibrate first (F8)", 4)
                    elif ev == "calib":
                        self.state = "calib"
                        self.tune(fresh=True)
                    elif ev == "refine":
                        self.state = "calib"
                        self.tune(fresh=False)
                    elif ev == "posture":
                        self.state = "calib"
                        self.posture_check()
                    elif ev == "recenter":
                        self.state = "calib"
                        self.recenter()
                    elif ev == "valid":
                        self.state = "valid"
                        self.validate()

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

                canvas = self.render_debug(frame, info)
                self.open_debug(canvas.shape)
                cv2.imshow(self.DBG, canvas)
                k = cv2.waitKey(1) & 0xFF
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
            self.cap.release()
            cv2.destroyAllWindows()

    def _take_events(self):
        ev = []
        while not self.keys.empty():
            e = self.keys.get_nowait()
            if e in ("quit", "toggle", "calib", "valid", "talk", "refine", "posture", "recenter"):
                ev.append(e)
        return ev


def main():
    ap = argparse.ArgumentParser(description="Webcam eye-tracking cursor")
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--no-fixation", action="store_true", help="disable the fixation hold")
    ap.add_argument("--online-refine", action="store_true",
                    help="EXPERIMENTAL: correct the gaze model online from App.register_click() events")
    ap.add_argument("--no-voice", action="store_true", help="do not read calibration instructions aloud")
    args = ap.parse_args()
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Missing model file: {MODEL_PATH}")
    App(args.camera, voice=not args.no_voice, fixation=not args.no_fixation,
        online_refine=args.online_refine).run()


if __name__ == "__main__":
    main()
