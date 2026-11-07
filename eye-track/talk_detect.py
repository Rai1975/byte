"""
Talking detection, used to pause all mouse input while the user speaks.

Two detectors:
  PresageTalk  Presage SmartSpectra TALKING metric. The SDK is Node-only, so a small Node process
               (presage/talk_bridge.mjs) receives our webcam frames over stdin and reports back.
               Needs PRESAGE_API_KEY (env var or eye-track/.env) and `npm install` in presage/.
  LocalTalk    Fallback built on the MediaPipe landmarks we already compute: the lips repeatedly
               opening and closing within a short window counts as talking.

TalkGate picks between them (Presage when it is live, otherwise local) and adds a short hold so
the pause doesn't flicker between words.
"""

import json
import os
import shutil
import struct
import subprocess
import threading
import time
from collections import deque

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BRIDGE_DIR = os.path.join(HERE, "presage")
BRIDGE = os.path.join(BRIDGE_DIR, "talk_bridge.mjs")

# ----------------------------------------------------------------------------- tuning
TALK_MODE = "auto"        # "auto": Presage if live, else local | "both": either one | "local" | "presage"
HOLD_S = 0.7              # keep input paused this long after talking stops (gaps between words)
PRESAGE_SEND_W = 640      # frames are downscaled to this width before going to Presage
PRESAGE_STALE_S = 3.0     # no talking update for this long -> treat Presage as not live
LOCAL_WINDOW_S = 0.7      # look-back window for lip movement
LOCAL_RANGE = 0.028       # min (max - min) lip gap, relative to eye-corner distance
LOCAL_STEP = 0.010        # min lip-gap change that counts as an open/close reversal...
LOCAL_STEP_FRAC = 0.3     # ...and at least this fraction of the window's range
LOCAL_REVERSALS = 3       # >= this many open/close direction changes in the window = talking

UPPER_LIP, LOWER_LIP = 13, 14
EYE_OUTER_L, EYE_OUTER_R = 33, 263


def load_api_key():
    key = os.environ.get("PRESAGE_API_KEY")
    if key:
        return key.strip()
    env = os.path.join(HERE, ".env")
    if os.path.exists(env):
        with open(env) as f:
            for line in f:
                k, _, v = line.partition("=")
                if k.strip() == "PRESAGE_API_KEY" and v.strip():
                    return v.strip().strip('"').strip("'")
    return None


# ============================================================================ Presage
class PresageTalk:
    def __init__(self):
        self.status = "off"
        self.detected = False
        self.stable = False
        self.last_msg = 0.0
        self.last_true = float("-inf")
        self.validation = ""
        self.messages = deque(maxlen=50)      # raw bridge messages, for debugging
        self.stderr_tail = deque(maxlen=20)   # SDK log lines (auth errors show up here)
        self.proc = None
        self._slot = None
        self._event = threading.Event()
        self._alive = False
        self._last_ts = 0.0

        key = load_api_key()
        node = shutil.which("node")
        if not key:
            self.status = "no API key (set PRESAGE_API_KEY or eye-track/.env)"
            return
        if not node:
            self.status = "node not found on PATH"
            return
        if not os.path.isdir(os.path.join(BRIDGE_DIR, "node_modules", "@smartspectra")):
            self.status = "SDK missing: run `npm install` in eye-track/presage"
            return

        env = dict(os.environ, PRESAGE_API_KEY=key)
        try:
            self.proc = subprocess.Popen(
                [node, BRIDGE], cwd=BRIDGE_DIR, env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as e:
            self.status = f"failed to start bridge: {e}"
            return
        self.status = "starting"
        self._alive = True
        threading.Thread(target=self._writer, daemon=True).start()
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._stderr, daemon=True).start()

    # --- frames out
    def submit(self, frame_bgr):
        if not self._alive:
            return
        h, w = frame_bgr.shape[:2]
        if w > PRESAGE_SEND_W:
            frame_bgr = cv2.resize(frame_bgr, (PRESAGE_SEND_W, int(h * PRESAGE_SEND_W / w)),
                                   interpolation=cv2.INTER_AREA)
        ts = time.time() * 1e6
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        self._slot = (np.ascontiguousarray(frame_bgr), ts)   # keep only the newest frame
        self._event.set()

    def _writer(self):
        while self._alive:
            self._event.wait(0.5)
            self._event.clear()
            item, self._slot = self._slot, None
            if item is None:
                continue
            frame, ts = item
            h, w = frame.shape[:2]
            try:
                self.proc.stdin.write(struct.pack("<IId", w, h, ts))
                self.proc.stdin.write(frame.tobytes())
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                self._dead("bridge exited")
                return

    # --- results in
    def _reader(self):
        for raw in self.proc.stdout:
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            self.messages.append(msg)
            t = msg.get("type")
            if t == "talking":
                self.detected, self.stable = msg["detected"], msg["stable"]
                self.last_msg = time.time()
                if self.detected:
                    self.last_true = self.last_msg
                if self.status != "running":
                    self.status = "running"
            elif t == "status":
                s = msg["status"]
                if s == 3:
                    self.status = "running"
                elif s == 5:
                    self.status = self._explain("error")
            elif t == "validation":
                self.validation = msg.get("hint", "")
            elif t == "error":
                self.status = self._explain("error: " + msg.get("message", ""))
        time.sleep(0.2)  # let the stderr thread catch the SDK's last log lines
        self._dead(self._explain(self.status if self.status.startswith(("error", "auth")) else "bridge exited"))

    def _explain(self, fallback):
        """Turn SDK output into a readable reason (a rejected key is only logged to stderr)."""
        text = " ".join(self.stderr_tail).lower() + " " + fallback.lower()
        if "401" in text or "authentication" in text or "usage verification failed" in text:
            return "auth failed: check API key"
        return fallback

    def _stderr(self):
        for line in self.proc.stderr:   # drain so the pipe never blocks; SDK logs land here
            line = line.decode(errors="ignore").strip()
            if line:
                self.stderr_tail.append(line)

    def _dead(self, why):
        self._alive = False
        if why.startswith("auth") or not self.status.startswith(("error", "auth")):
            self.status = why

    @property
    def live(self):
        return self._alive and self.status == "running" and time.time() - self.last_msg < PRESAGE_STALE_S

    def talking(self, now):
        return self.detected or (now - self.last_true < HOLD_S)

    def close(self):
        self._alive = False
        if self.proc:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()


# ============================================================================ local fallback
class LocalTalk:
    def __init__(self):
        self.hist = deque()
        self.last_true = float("-inf")
        self.level = 0.0       # lip-gap range in the window (for the HUD)
        self.reversals = 0

    def update(self, pts, now):
        if pts is None:
            self.hist.clear()
            return
        scale = np.linalg.norm(pts[EYE_OUTER_R] - pts[EYE_OUTER_L]) + 1e-6
        gap = float(np.linalg.norm(pts[LOWER_LIP] - pts[UPPER_LIP])) / scale
        self.hist.append((now, gap))
        while self.hist and now - self.hist[0][0] > LOCAL_WINDOW_S:
            self.hist.popleft()
        vals = [v for _, v in self.hist]
        if len(vals) < 5:
            return
        self.level = max(vals) - min(vals)
        # count open/close reversals, ignoring wiggles that are small in absolute terms or
        # relative to the overall movement (so jitter during a yawn doesn't count)
        step = max(LOCAL_STEP, LOCAL_STEP_FRAC * self.level)
        rev, direction, extreme = 0, 0, vals[0]
        for v in vals[1:]:
            if direction >= 0 and v < extreme - step:
                rev += direction == 1
                direction, extreme = -1, v
            elif direction <= 0 and v > extreme + step:
                rev += direction == -1
                direction, extreme = 1, v
            elif (direction == 1 and v > extreme) or (direction == -1 and v < extreme):
                extreme = v
        self.reversals = rev
        if self.level > LOCAL_RANGE and rev >= LOCAL_REVERSALS:
            self.last_true = now

    def talking(self, now):
        return now - self.last_true < HOLD_S


# ============================================================================ gate
class TalkGate:
    def __init__(self, mode=TALK_MODE):
        self.mode = mode
        self.enabled = True
        self.presage = PresageTalk() if mode != "local" else None
        self.local = LocalTalk()
        self.source = "local"

    def feed(self, frame, info, now):
        if self.presage:
            self.presage.submit(frame)
        self.local.update(info["pts"] if info else None, now)

    def talking(self, now):
        if not self.enabled:
            self.source = "off"
            return False
        p_live = self.presage is not None and self.presage.live
        if self.mode == "presage" or (self.mode == "auto" and p_live):
            self.source = "presage" if p_live else "presage (not live)"
            return p_live and self.presage.talking(now)
        if self.mode == "both" and p_live:
            self.source = "presage+local"
            return self.presage.talking(now) or self.local.talking(now)
        self.source = "local"
        return self.local.talking(now)

    def presage_status(self):
        return self.presage.status if self.presage else "disabled"

    def close(self):
        if self.presage:
            self.presage.close()
