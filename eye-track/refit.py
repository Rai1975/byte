"""Refit the gaze model offline from saved calibration sessions and print grouped-CV error.

    python refit.py                       # newest session in sessions/
    python refit.py sessions/A.npz B.npz  # pool several sessions (e.g. several days)
    python refit.py --all                 # pool every session
    python refit.py --compare             # also run the old validation (no neighbour gap) side by side

Reports:
  selection CV  grouped CV used to choose the model (optimistic: it also picked the winner)
  nested CV     honest error of the whole procedure (choice of order/alpha/head features/stage 2 inside)
  valid-dot err the accuracy-check dots ('valid' phase) scored by a model trained WITHOUT them
No camera or UI is needed, so feature changes can be tried on saved data without recalibrating.
"""

import os

# Many small matrix solves: BLAS multithreading only adds overhead (3x slower fits measured), so use 1 thread.
# Must run before numpy is imported.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


import argparse
import glob
import os
import sys
import time

import numpy as np

import samples as sessions
from gaze_model import GazeModel, fit_pool, load_prior


def pool_sessions(paths):
    parts, screen, goff = [], None, 0
    for p in paths:
        A, meta = sessions.load(p)
        if goff == 0 and not parts:
            nfeat = A["F"].shape[1]
        if A["F"].shape[1] != nfeat:
            print(f"skipping {os.path.basename(p)}: {A['F'].shape[1]} features != {nfeat}")
            continue
        if screen is None:
            screen = meta["screen"]
        elif meta["screen"] != screen:
            print(f"skipping {os.path.basename(p)}: screen {meta['screen']} != {screen}")
            continue
        A = dict(A)
        A["G"] = A["G"] + goff
        A["V"] = A["V"] + goff
        goff = int(max(A["G"].max(), A["V"].max())) + 2
        A["SRC"] = np.full(len(A["F"]), len(parts), np.int16)
        parts.append(A)
    cat = {k: np.concatenate([a[k] for a in parts]) for k in parts[0]}
    return cat, screen


def valid_error(A, screen, model_kwargs):
    """Train without the 'valid' phase, score on its dots (mean per-dot distance of the averaged prediction)."""
    vi = sessions.PHASES.index("valid")
    test = A["PH"] == vi
    if not test.any() or (~test).sum() < 100:
        return None
    tr = ~test
    hc, m = fit_pool({k: v[tr] for k, v in A.items()}, screen, **model_kwargs)
    P = np.array([m.predict(f) for f in hc.apply_many(A["F"][test])])
    S, G = A["S"][test], A["G"][test]
    errs = [np.linalg.norm(P[G == g].mean(0) - S[G == g][0]) for g in np.unique(G)]
    return float(np.mean(errs)), len(errs), m


def posture_error(A, hc, m):
    """Mean per-dot error on the posture-check dots (phase 'posture'; never used for fitting).
    Dots arrive in blocks of 5 per posture, in order."""
    pi = sessions.PHASES.index("posture")
    sel = A["PH"] == pi
    if not sel.any():
        return None
    F, S, G = A["F"][sel], A["S"][sel], A["G"][sel]
    P = np.array([m.predict(f) for f in hc.apply_many(F)])
    errs = [np.linalg.norm(P[G == g].mean(0) - S[G == g][0]) for g in np.unique(G)]
    out = {}
    for i in range(0, len(errs), 5):
        out[f"posture {i // 5 + 1}"] = float(np.mean(errs[i:i + 5]))
    out["mean"] = float(np.mean(errs))
    return out


def run(A, screen, label, adj, nested, **opts):
    t0 = time.time()
    hc, m = fit_pool(A, screen, nested=nested, adj=adj, **opts)
    r2 = "n/a" if hc.r2 is None else f"{100 * hc.r2:.0f}%"
    print(f"\n[{label}]  fit {time.time() - t0:.1f}s")
    print(f"  model        : {m.summary}")
    print(f"  head explains: {r2} of iris shift")
    print(f"  selection CV : {m.cv_err:.1f} px   (stage 1 only {m.cv_base:.1f})" if m.cv_err else "  selection CV : n/a")
    if m.cv_nested is not None:
        print(f"  nested CV    : {m.cv_nested:.1f} px   <- honest estimate")
    ve = valid_error(A, screen, dict(nested=False, adj=adj, **opts))
    pe = posture_error(A, hc, m)
    if pe:
        print("  posture check: " + "   ".join(f"{k} {v:.0f} px" for k, v in pe.items())
              + "   (never fitted; compare with valid-dot err above)")
    if ve:
        print(f"  valid-dot err: {ve[0]:.1f} px on {ve[1]} dots (model trained without them)")
    return m


def sweep(A, screen, variants=None):
    """One line per variant: nested-CV error, selection-CV error, held-out valid-dot error."""
    variants = variants or [("baseline", {}), ("no clean", dict(clean=False)),
                            ("weights=openness", dict(weights="openness")), ("huber", dict(robust=True)),
                            ("huber+openness", dict(robust=True, weights="openness"))]
    print(f"{'variant':22s} {'nested':>8s} {'select':>8s} {'valid':>8s}   model")
    for name, o in variants:
        hc, m = fit_pool(A, screen, nested=True, **o)
        ve = valid_error(A, screen, dict(nested=False, **o))
        nv = "n/a" if m.cv_nested is None else f"{m.cv_nested:.1f}"
        print(f"{name:22s} {nv:>8s} {m.cv_err:8.1f} {ve[0] if ve else float('nan'):8.1f}   {m.summary}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="session .npz files (default: newest)")
    ap.add_argument("--all", action="store_true", help="pool every file in sessions/")
    ap.add_argument("--compare", action="store_true", help="also run the old validation without neighbour gaps")
    ap.add_argument("--no-nested", action="store_true", help="skip nested CV (faster)")
    ap.add_argument("--weights", choices=["none", "openness"], default=None, help="per-sample weights")
    ap.add_argument("--robust", action="store_true", help="Huber re-weighting")
    ap.add_argument("--no-clean", action="store_true", help="skip fit-time cleaning of static dots")
    ap.add_argument("--prior", default=None, help="prior.json to shrink toward (default: none)")
    ap.add_argument("--sweep", action="store_true", help="table of robustness variants (weights/Huber/clean)")
    args = ap.parse_args()

    paths = args.files
    if args.all:
        paths = sorted(glob.glob(os.path.join(sessions.SESSION_DIR, "*.npz")))
    elif not paths:
        p = sessions.latest()
        paths = [p] if p else []
    if not paths:
        sys.exit("No sessions found. Calibrate with eye_cursor.py first (F8).")

    A, screen = pool_sessions(paths)
    print(f"{len(paths)} session(s), {len(A['F'])} samples, {len(np.unique(A['G']))} CV groups, screen {screen}")
    for i, name in enumerate(sessions.PHASES):
        n = int((A["PH"] == i).sum())
        if n:
            print(f"  {name:8s}: {n} samples")
    if args.sweep:
        sweep(A, screen)
        return
    opts = dict(prior=load_prior(args.prior), weights=args.weights, robust=True if args.robust else None, clean=False if args.no_clean else None)
    run(A, screen, "grouped CV (neighbour gap)", "auto", not args.no_nested, **opts)
    if args.compare:
        run(A, screen, "old CV (plain leave-one-group-out)", None, False)


if __name__ == "__main__":
    main()
