"""Raw calibration-session storage: one .npz per session in eye-track/sessions/.

Arrays (one row per frame): F features, S screen target (px), G CV group id, V visit id (head phase),
K head-phase flag, T time (s, perf_counter), PH phase code (index into meta['phases']), R round number,
C group is a contiguous time block. Metadata (JSON): screen size, feature names, phases, timestamps,
library versions. Lets you refit offline (refit.py) and continue a session with F11 after a restart.
"""

import glob
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SESSION_DIR = os.path.join(HERE, "sessions")
PHASES = ["dots", "pursuit", "head", "valid", "mixed", "posture"]
KEYS = ("F", "S", "G", "V", "K", "T", "PH", "R", "C")


def new_path():
    os.makedirs(SESSION_DIR, exist_ok=True)
    return os.path.join(SESSION_DIR, time.strftime("%Y%m%d-%H%M%S") + ".npz")


def save(path, arrays, meta):
    """arrays: dict with KEYS. Written atomically (temp file then rename)."""
    meta = dict(meta, phases=PHASES, saved=time.ctime(), n=int(len(arrays["F"])))
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **{k: np.asarray(arrays[k]) for k in KEYS})
    os.replace(tmp, path)


def load(path):
    """-> (arrays dict, meta dict)"""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in KEYS}, json.loads(str(z["meta"]))


def latest():
    files = sorted(glob.glob(os.path.join(SESSION_DIR, "*.npz")))
    return files[-1] if files else None
