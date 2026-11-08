"""Replay a saved session through the runtime filters to tune/verify the fixation hold (and RLS refiner).

    python eval_runtime.py [session.npz]

The gaze model is fitted on the session, then every phase is replayed in time order through the same
One Euro filter and FixationHold that the app uses:
  dots / valid  (the user is fixating a target): jitter = RMS spread of the cursor around its own mean,
                error = distance of the mean cursor from the target
  pursuit       (the cursor must follow a moving target): mean tracking error vs the true dot position,
                which shows how much lag/error a wrong hold would add
Smaller jitter with an unchanged pursuit error means the hold helps.
"""

import os

# Many small matrix solves: BLAS multithreading only adds overhead (3x slower fits measured), so use 1 thread.
# Must run before numpy is imported.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


import sys

import numpy as np

import samples as sessions
from gaze_model import fit_pool
from runtime_filters import FixationHold, OneEuro, OnlineRefiner

ONE_EURO = (0.6, 0.004, 1.0)       # same defaults as eye_cursor.py


def replay(P, T, fix):
    """Run predictions P (n,2) at times T through One Euro (+ fixation hold) -> output (n,2), fixating flags."""
    f = OneEuro(*ONE_EURO)
    out, fl = np.zeros_like(P), np.zeros(len(P), bool)
    for i in range(len(P)):
        s = f(P[i], T[i])
        if fix is not None:
            s, fl[i] = fix.update(s, T[i])
        out[i] = s
    return out, fl


def evaluate(A, screen, v_enter, v_exit, radius, min_s):
    hc, m = fit_pool(A, screen, nested=False)
    P = np.array([m.predict(f) for f in hc.apply_many(A["F"])])
    k = screen[0] / 1920.0
    res = {}
    for name, use in (("one-euro only", False), ("with fixation hold", True)):
        jit, err, pur, hold = [], [], [], []
        for ph, kind in ((0, "dot"), (3, "dot"), (1, "pursuit")):
            sel = A["PH"] == ph
            for g in np.unique(A["G"][sel]):
                ix = np.flatnonzero(sel & (A["G"] == g))
                if len(ix) < 8:
                    continue
                fix = FixationHold(v_enter * k, v_exit * k, radius * k, min_s) if use else None
                out, fl = replay(P[ix], A["T"][ix], fix)
                tail = slice(len(ix) // 3, None)                 # skip the first third (eyes still landing)
                if kind == "dot":
                    jit.append(np.sqrt(np.mean(np.sum((out[tail] - out[tail].mean(0)) ** 2, 1))))
                    err.append(np.linalg.norm(out[tail].mean(0) - A["S"][ix][0]))
                    hold.append(fl[tail].mean())
                else:
                    pur.append(np.mean(np.linalg.norm(out - A["S"][ix], axis=1)))
                    hold.append(fl.mean() * 0)                   # not counted
        res[name] = dict(jitter=np.mean(jit), dot_err=np.mean(err), pursuit_err=np.mean(pur) if pur else float("nan"),
                         held=np.mean([h for h in hold if h is not None]))
    return res


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else sessions.latest()
    A, meta = sessions.load(path)
    A = dict(A)
    screen = meta["screen"]
    print(f"{path}: {len(A['F'])} samples, screen {screen}")
    grid = [(120, 350, 70, 0.15), (80, 300, 50, 0.15), (200, 500, 90, 0.15), (120, 350, 70, 0.3), (60, 250, 40, 0.1)]
    print(f"{'v_enter v_exit radius min_s':32s}{'variant':22s}{'jitter':>8s}{'dot err':>9s}{'pursuit err':>13s}")
    for g in grid:
        r = evaluate(A, screen, *g)
        for name, v in r.items():
            print(f"{str(g):32s}{name:22s}{v['jitter']:8.1f}{v['dot_err']:9.1f}{v['pursuit_err']:13.1f}")
            if name == "one-euro only" and g != grid[0]:
                pass


if __name__ == "__main__":
    main()
