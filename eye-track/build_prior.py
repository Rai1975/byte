"""Build a population prior (prior.json) from several users' saved sessions.

    python build_prior.py sessions_userA/ sessions_userB/ other_user.npz ...

Each argument is ONE user: a .npz session file, or a folder whose sessions are pooled. For every user a
plain linear model on the full feature set is fitted (after head compensation); its coefficients are
converted to screen-fraction per raw-feature unit, so they transfer across screen sizes, and the median
over users becomes the prior. The head-compensation matrix B is averaged the same way.

eye_cursor.py / refit.py load prior.json (if it exists) and the ridge then shrinks the linear and
head-compensation coefficients toward it instead of toward zero. The ridge strength is still chosen by
grouped CV, so a poor prior is simply ignored (CV picks a small alpha).
"""

import os

# Many small matrix solves: BLAS multithreading only adds overhead (3x slower fits measured), so use 1 thread.
# Must run before numpy is imported.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


import glob
import json
import os
import sys
import time

import numpy as np

import samples as sessions
from gaze_model import GazeModel, fit_pool
from refit import pool_sessions

PRIOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prior.json")


def user_files(arg):
    return sorted(glob.glob(os.path.join(arg, "*.npz"))) if os.path.isdir(arg) else [arg]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    coefs, Bs = [], []
    nfeat = len(GazeModel.FEATURES)
    for arg in sys.argv[1:]:
        files = [f for f in user_files(arg)]
        if not files:
            print("no sessions in", arg)
            continue
        A, screen = pool_sessions(files)
        if A["F"].shape[1] != nfeat:
            print(f"skipping {arg}: {A['F'].shape[1]} features, need {nfeat}")
            continue
        hc, m = fit_pool(A, screen, nested=False, kinds=("lin",), masks=[list(range(nfeat))])
        per_axis = []
        for ax in (0, 1):
            W = m.axes[ax]["W"]
            per_axis.append([W[1 + i] / m.sd[c] / m.scale[ax] for i, c in enumerate(m.axes[ax]["cols"])])
        coefs.append(np.array(per_axis))             # (2, nfeat) screen fraction per raw unit
        Bs.append(hc.B)
        print(f"{arg}: cv {m.cv_err:.0f}px, head explains {hc.r2 if hc.r2 is not None else float('nan'):.2f}")
    if not coefs:
        sys.exit("nothing to build from")
    C = np.median(coefs, axis=0)
    B = np.median(Bs, axis=0)
    names = GazeModel.FEATURES
    out = dict(version=1, features=names, n_users=len(coefs), built=time.ctime(),
               axes=[dict(coef={n: float(C[ax][i]) for i, n in enumerate(names)}) for ax in (0, 1)],
               head_B=B.tolist())
    with open(PRIOR_PATH, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {PRIOR_PATH} from {len(coefs)} user(s)")


if __name__ == "__main__":
    main()
