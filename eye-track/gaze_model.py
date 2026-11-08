"""Gaze regression model (no camera/UI deps, so offline scripts can import it)."""

import os

# Many small matrix solves: BLAS multithreading only adds overhead (3x slower fits measured), so use 1 thread.
# Must run before numpy is imported.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


import numpy as np

RIDGE = 0.01            # fallback regularization strength (normally chosen by cross-validation)


class HeadComp:
    """Removes the effect of head pose from the iris offset.

    Learned ONLY from the head-movement phase, where the gaze target is fixed: any change of the
    iris offset there is caused by head movement, so within each static dot we regress the iris
    offset change on the head-pose change (linear, ridge). The eye model then sees
        iris' = iris - B . (head_pose - neutral_pose)
    which is what the iris would read with the head at its neutral pose. neutral_pose = mean pose
    of the head-still phases."""

    COLS = [3, 4, 5, 6, 7, 8]       # head yaw, pitch, x, y, z, roll in the feature vector
    MIN_R2 = 0.05                   # below this the compensation is switched off
    # gaze features that head pose contaminates: iris x/y, openness, eye-look blendshapes, head-frame iris.
    # With the gaze fixed (head phase) head pitch alone moves the eyelid/blendshape signals by many SDs,
    # in different directions from the iris, so ALL of them are compensated, not just the iris.
    # Compensating all of them (TARGETS_ALL) did not improve held-out error on the first clean session, so the
    # default stays iris-only; set HeadComp.TARGETS = HeadComp.TARGETS_ALL to try it.
    TARGETS_ALL = [0, 1, 2, 9, 10, 11, 12, 13, 14]
    TARGETS = [0, 1]

    def __init__(self):
        self.targets = list(self.TARGETS)
        self.B = np.zeros((len(self.COLS), len(self.targets)))
        self.ref = np.zeros(len(self.COLS))
        self.r2 = None

    def fit(self, Fh, Vh, Fe, alpha=1e-3, B0=None):
        """B0: optional prior for B (6x2, raw units). The ridge penalty then pulls B toward B0 instead of 0."""
        Fh, Vh = np.asarray(Fh, float), np.asarray(Vh)
        self.targets = [t for t in self.TARGETS if t < Fh.shape[1]]
        H, I = Fh[:, self.COLS].copy(), Fh[:, self.targets].copy()
        for v in np.unique(Vh):
            m = Vh == v
            H[m] -= H[m].mean(0)
            I[m] -= I[m].mean(0)
        sd = H.std(0) + 1e-9
        X = H / sd
        reg = alpha * len(X) * np.eye(X.shape[1])
        rhs = X.T @ I
        if B0 is not None:
            B0 = np.asarray(B0, float)
            B0 = np.pad(B0, ((0, 0), (0, len(self.targets) - B0.shape[1])))[:, :len(self.targets)]
            rhs = rhs + reg @ (B0 * sd[:, None])
        B = np.linalg.solve(X.T @ X + reg, rhs)
        Ii, Bi = I[:, :2], (X @ B)[:, :2]
        self.r2 = float(1 - ((Ii - Bi) ** 2).sum() / (((Ii ** 2).sum()) + 1e-12))     # reported for the iris
        self.B = B / sd[:, None] if self.r2 >= self.MIN_R2 else np.zeros_like(B)
        Fe = np.asarray(Fe, float)
        self.ref = Fe[:, self.COLS].mean(0) if len(Fe) else Fh[:, self.COLS].mean(0)

    def apply(self, f):
        f = np.array(f, float)
        f[self.targets] -= (f[self.COLS] - self.ref) @ self.B
        return f

    def apply_many(self, F):
        F = np.array(F, float)
        F[:, self.targets] -= (F[:, self.COLS] - self.ref) @ self.B
        return F

    def to_dict(self):
        return dict(B=self.B.tolist(), ref=self.ref.tolist(), r2=self.r2, targets=self.targets)

    def from_dict(self, d):
        self.B, self.ref, self.r2 = np.array(d["B"]), np.array(d["ref"]), d.get("r2")
        self.targets = d.get("targets", list(range(self.B.shape[1])))      # old files: iris only


class GazeModel:
    """Two-stage gaze regression (features -> screen px).

    Stage 1: one ridge regression per screen axis on standardized features. Each axis picks its own
    polynomial order (linear / quadratic / cubic in the iris offset) and ridge strength by
    leave-one-group-out cross-validation (a group = one calibration dot or one pursuit segment).

    Stage 2 ("residual field"): the stage-1 error still depends on WHERE on the screen you look
    (lens/eye geometry bends the mapping). The held-out stage-1 errors are smoothed into a field over
    predicted screen position (Gaussian kernel, Nadaraya-Watson with shrinkage), and predictions are
    nudged by that field. Kept only if it improves held-out error."""

    FEATURES = ["iris_x", "iris_y", "openness", "head_yaw", "head_pitch",
                "head_x", "head_y", "head_z", "head_roll",
                "bs_h", "bs_up", "bs_down",            # MediaPipe eyeLook* blendshapes (horizontal, up, down)
                "ih_x", "ih_y", "ih_z"]                # iris offset expressed in the head frame
    HEAD = [3, 4, 5, 6, 7, 8]
    BS = [9, 10, 11]
    IH = [12, 13, 14]
    # candidate feature sets; EACH SCREEN AXIS picks its own by grouped CV, so a feature group is only
    # used where it lowers held-out error (e.g. blendshapes can serve the vertical axis only)
    MASKS = ([0, 1, 2], [0, 1, 2] + HEAD, [0, 1, 2] + BS, [0, 1, 2] + HEAD + BS,
             [0, 1, 2] + IH, [0, 1, 2] + HEAD + IH, [0, 1, 2] + HEAD + BS + IH)
    KINDS = ("lin", "quad", "cubic", "lin_int", "quad_int")   # *_int: iris x head/extra-feature terms
    ALPHAS = (1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)
    SIGMAS = (0.05, 0.08, 0.12, 0.2)    # field bandwidth, fraction of the screen size
    LAMBDAS = (5.0, 25.0, 100.0)        # field shrinkage toward zero correction, in samples
    HUBER_C = 1.345                     # Huber threshold in units of the robust residual scale
    COMPLEXITY_TOL = 0.1                # prefer the simplest model whose CV MSE is within this fraction of the best
    CLAMP_MARGIN = 0.0                  # None = off; else standardized inputs are clipped to the training range +/- this fraction
    ROBUST = False                      # Huber re-weighting (default off unless shown to help)
    CELL = 0.02                         # field binning cell, fraction of the screen size

    def __init__(self):
        self.mu = self.sd = None
        self._kinds = None
        self._prior = None
        self.lo = self.hi = None   # per-feature range (standardized) seen in training; predictions clamp to it
        self.axes = None        # per screen axis: dict(kind, alpha, W)
        self.cols = None        # feature columns in use
        self.scale = np.array([1920.0, 1080.0])
        self.field = None       # dict(cen, res, cnt, sigma, lam) or None
        self.cv_base = None     # held-out error (px) of stage 1 alone
        self.cv_err = None      # held-out error (px) of the final model (selection CV: optimistic)
        self.cv_nested = None   # nested-CV error (px): honest estimate of the whole procedure

    @staticmethod
    def _expand(Z, kind):
        """Design matrix. Z columns: iris x, iris y, openness, then the extra features of the mask.
        '*_int' kinds add iris x extra-feature products (head pose / blendshapes shifting the mapping)."""
        ix, iy = Z[:, 0], Z[:, 1]
        base = kind.split("_")[0]
        cols = [np.ones(len(Z)), Z]
        if base in ("quad", "cubic"):
            cols += [ix * ix, iy * iy, ix * iy]
        if base == "cubic":
            cols += [ix ** 3, iy ** 3, ix * ix * iy, ix * iy * iy]
        if kind.endswith("_int") and Z.shape[1] > 3:
            ex = Z[:, 3:]
            cols += [ix[:, None] * ex, iy[:, None] * ex]
        return np.column_stack(cols)

    @staticmethod
    def _solve(X, y, alpha, w=None, w0=None):
        """Ridge toward the prior w0 (zeros when None): argmin |y - Xw|^2_W + alpha n |w - w0|^2 (no penalty on the intercept)."""
        reg = np.eye(X.shape[1]) * alpha * len(X)
        reg[0, 0] = 0.0
        Xw = X if w is None else X * w[:, None]
        rhs = Xw.T @ y
        if w0 is not None:
            rhs = rhs + reg @ w0
        return np.linalg.solve(Xw.T @ X + reg, rhs)

    def _w0(self, ax, cols, dim):
        """Prior weight vector for one screen axis: prior coefficients (screen fractions per RAW feature
        unit) are converted to this user's standardized-feature units; non-linear terms have prior 0."""
        if self._prior is None:
            return None
        coef = self._prior["axes"][ax]["coef"]
        w0 = np.zeros(dim)
        for i, c in enumerate(cols):
            w0[1 + i] = coef.get(self.FEATURES[c], 0.0) * self.scale[ax] * self.sd[c]
        return w0

    # ---- grouped cross-validation machinery -------------------------------------------------
    # A "group" is one calibration dot or one contiguous time block. `adj` maps a group to its
    # temporal neighbours: when a group is held out, its neighbours are left out of training too
    # (neighbouring blocks share slow drift and eye lag, so they would leak). adj=None -> plain
    # leave-one-group-out (the previous behaviour).
    def _stats(self, X, y, idx, ids, adj):
        """Per-fold training normal equations (fold = one held-out group plus its temporal neighbours),
        obtained by subtracting the held-out groups' statistics from the totals. Independent of alpha."""
        w = self._w
        A = {g: (X[idx[g]] * w[idx[g], None]).T @ X[idx[g]] for g in ids}
        b = {g: (X[idx[g]] * w[idx[g], None]).T @ y[idx[g]] for g in ids}
        n = {g: len(idx[g]) for g in ids}
        TA, Tb, Tn = sum(A.values()), sum(b.values()), sum(n.values())
        idset = set(ids)
        Atr, btr, ntr = [], [], []
        for g in ids:
            ex = [g] + [h for h in (adj or {}).get(g, ()) if h in idset]
            Atr.append(TA - sum(A[e] for e in ex))
            btr.append(Tb - sum(b[e] for e in ex))
            ntr.append(Tn - sum(n[e] for e in ex))
        return np.array(Atr), np.array(btr), np.array(ntr, float)

    def _oof_stats(self, X, st, idx, ids, alpha, out, w0=None):
        """Fill `out` with out-of-fold predictions for the samples of `ids`: one batched ridge solve over
        all folds (exact, the same numbers as fitting each fold separately)."""
        Atr, btr, ntr = st
        p = X.shape[1]
        reg = np.zeros_like(Atr)
        d = np.arange(1, p)                                  # no penalty on the intercept
        reg[:, d, d] = (alpha * ntr)[:, None]
        rhs = btr if w0 is None else btr + reg @ w0
        Wf = np.linalg.solve(Atr + reg, rhs[..., None])[..., 0]
        for i, g in enumerate(ids):
            out[idx[g]] = X[idx[g]] @ Wf[i] if ntr[i] >= p + 2 else np.nan

    @staticmethod
    def _group_err(P, S, idx, ids):
        return float(np.mean([np.sqrt(np.mean(np.sum((P[idx[g]] - S[idx[g]]) ** 2, 1))) for g in ids]))

    @staticmethod
    def _bin(P, R, scale, cell):
        key = np.floor(P / (cell * scale)).astype(int)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        cnt = np.bincount(inv).astype(float)
        cen = np.stack([np.bincount(inv, P[:, k]) / cnt for k in (0, 1)], 1)
        res = np.stack([np.bincount(inv, R[:, k]) / cnt for k in (0, 1)], 1)
        return dict(cen=cen, res=res, cnt=cnt)

    @staticmethod
    def _field(P, f, scale):
        d = (P[:, None, :] - f["cen"][None]) / (f["sigma"] * scale)
        k = np.exp(-0.5 * (d ** 2).sum(-1)) * f["cnt"][None]
        return (k @ f["res"]) / (k.sum(1, keepdims=True) + f["lam"])

    def _stage1(self, Z, S, idx, ids, adj, masks, can_cv):
        """Per screen axis, pick feature set, polynomial kind and alpha by grouped CV over `ids`.
        Returns (total_mse, used_columns, axes, oof)."""
        rows = np.concatenate([idx[g] for g in ids])
        axes_, oof_, total, used = [], np.full(S.shape, np.nan), 0.0, set()
        kinds = self._kinds or self.KINDS
        for ax in (0, 1):
            y = S[:, ax]
            if can_cv:
                best, cands = None, []
                for cols in masks:
                    Zm = Z[:, cols]
                    for k_ in kinds:
                        if k_.endswith("_int") and len(cols) <= 3:
                            continue                     # nothing to interact with
                        X = self._expand(Zm, k_)
                        st = self._stats(X, y, idx, ids, adj)
                        w0 = self._w0(ax, cols, X.shape[1])
                        for a in self.ALPHAS:
                            p = np.full(len(y), np.nan)
                            self._oof_stats(X, st, idx, ids, a, p, w0)
                            sc = float(np.nanmean((p[rows] - y[rows]) ** 2))
                            cands.append((sc, X.shape[1], cols, k_, a, p))
                            if best is None or sc < best[0]:
                                best = (sc, cols, k_, a, p)
                if self.COMPLEXITY_TOL > 0:        # simplest model within tolerance of the best: extrapolates less
                    ok = [c for c in cands if c[0] <= best[0] * (1 + self.COMPLEXITY_TOL)]
                    c = min(ok, key=lambda c: (c[1], c[0]))
                    best = (c[0], c[2], c[3], c[4], c[5])
                total += best[0]
                _, cols, kind, alpha, oof_[:, ax] = best
            else:
                cols, kind, alpha = masks[-1], "quad", RIDGE
            Xr = self._expand(Z[rows][:, cols], kind)
            axes_.append(dict(cols=list(cols), kind=kind, alpha=alpha,
                              W=self._solve(Xr, y[rows], alpha, self._w[rows], self._w0(ax, cols, Xr.shape[1]))))
            used |= set(cols)
        return total, sorted(used), axes_, oof_

    def _pick_field(self, oof, S, idx, ids, adj):
        """Grid-search the stage-2 field by grouped CV. Returns (err, sigma, lambda); the residuals
        are only defined for rows of `ids`."""
        R = S - oof
        idset = set(ids)
        allrows = np.concatenate([idx[h] for h in ids])
        best = None
        for sg in self.SIGMAS:
            for lam in self.LAMBDAS:
                P2 = oof.copy()
                for g in ids:
                    ex = np.concatenate([idx[g]] + [idx[h] for h in (adj or {}).get(g, ()) if h in idset])
                    tr = np.setdiff1d(allrows, ex)
                    f = self._bin(oof[tr], R[tr], self.scale, self.CELL)
                    f.update(sigma=sg, lam=lam)
                    P2[idx[g]] = oof[idx[g]] + self._field(oof[idx[g]], f, self.scale)
                e = self._group_err(P2, S, idx, ids)
                if best is None or e < best[0]:
                    best = (e, sg, lam)
        return best

    def _pipeline(self, Z, S, idx, ids, adj, masks):
        """Full two-stage fit on `ids`. Returns dict(cols, axes, field, cv_base, cv_err, oof)."""
        rows = np.concatenate([idx[g] for g in ids])
        w0 = self._w
        _, cols, axes, oof = self._stage1(Z, S, idx, ids, adj, masks, True)
        if self.robust:
            # Huber via held-out residuals: samples that the other groups cannot explain (glitches,
            # catch-up saccades, blink aftermath) get weight c*s/|r|; then stage 1 is repeated
            r = np.linalg.norm(S[rows] - oof[rows], axis=1)
            s_ = 1.4826 * np.median(r) + 1e-9
            hub = np.minimum(1.0, self.HUBER_C * s_ / np.maximum(r, 1e-9))
            self._w = w0.copy()
            self._w[rows] = w0[rows] * hub
            self._w[rows] *= len(rows) / self._w[rows].sum()
            _, cols, axes, oof = self._stage1(Z, S, idx, ids, adj, masks, True)
        base = self._group_err(oof, S, idx, ids)
        out = dict(cols=cols, axes=axes, field=None, cv_base=base, cv_err=base, oof=oof)
        self._w = w0
        e, sg, lam = self._pick_field(oof, S, idx, ids, adj)
        if e < 0.97 * base:
            fld = self._bin(oof[rows], (S - oof)[rows], self.scale, self.CELL)
            fld.update(sigma=sg, lam=lam)
            out.update(field=fld, cv_err=e)
        return out

    def _predict_with(self, Z, cols, axes, field):
        P = np.column_stack([self._expand(Z[:, a["cols"]], a["kind"]) @ a["W"] for a in axes])
        if field is not None:
            P = P + self._field(P, field, self.scale)
        return P

    def fit(self, F, S, groups=None, scale=None, adj=None, nested=True, weights=None, robust=None,
            masks=None, kinds=None, prior=None):
        F, S = np.asarray(F, float), np.asarray(S, float)
        w = np.ones(len(F)) if weights is None else np.asarray(weights, float)
        self._w = w * (len(w) / w.sum())        # mean weight 1, so alpha keeps its meaning
        self.robust = self.ROBUST if robust is None else robust
        if scale is not None:
            self.scale = np.asarray(scale, float)
        self.mu = F.mean(0)
        self.sd = F.std(0) + 1e-9
        Z = (F - self.mu) / self.sd
        lo, hi = np.percentile(Z, 0.5, axis=0), np.percentile(Z, 99.5, axis=0)
        if self.CLAMP_MARGIN is None:
            self.lo = self.hi = None
        else:
            pad = self.CLAMP_MARGIN * (hi - lo)
            self.lo, self.hi = lo - pad, hi + pad
        groups = np.zeros(len(F), int) if groups is None else np.asarray(groups)
        ids = list(np.unique(groups))
        idx = {g: np.flatnonzero(groups == g) for g in ids}
        can_cv = len(ids) >= 6
        self.axes, self.field = None, None
        self.cv_base = self.cv_err = self.cv_nested = None
        n_feat = Z.shape[1]
        self._kinds = kinds
        self._prior = prior
        masks = [[c for c in m if c < n_feat] for m in (masks or self.MASKS)]
        masks = [m for i, m in enumerate(masks) if m not in masks[:i]]
        if not can_cv:
            _, self.cols, self.axes, _ = self._stage1(Z, S, idx, ids, adj, masks, False)
            return
        r = self._pipeline(Z, S, idx, ids, adj, masks)
        self.cols, self.axes, self.field = r["cols"], r["axes"], r["field"]
        self.cv_base, self.cv_err = r["cv_base"], r["cv_err"]
        if nested and len(ids) >= 12:
            self.cv_nested = self._nested_cv(Z, S, idx, ids, adj, masks)

    def _nested_cv(self, Z, S, idx, ids, adj, masks, K=5):
        """Honest error of the WHOLE procedure (hyper-parameter choice and stage 2 included): outer
        folds hold out ~1/K of the groups (plus their neighbours); everything is selected on the rest."""
        idset = set(ids)
        P = np.full(S.shape, np.nan)
        for k in range(K):
            te = [g for i, g in enumerate(ids) if i % K == k]
            ex = set(te)
            for g in te:
                ex |= {h for h in (adj or {}).get(g, ()) if h in idset}
            tr = [g for g in ids if g not in ex]
            if len(tr) < 6:
                continue
            r = self._pipeline(Z, S, idx, tr, adj, masks)
            rows = np.concatenate([idx[g] for g in te])
            P[rows] = self._predict_with(Z[rows], r["cols"], r["axes"], r["field"])
        ok = [g for g in ids if not np.isnan(P[idx[g]]).any()]
        return self._group_err(P, S, idx, ok) if ok else None

    def predict(self, f):
        z = (np.asarray(f, float)[None, :] - self.mu) / self.sd
        if self.lo is not None:
            z = np.clip(z, self.lo, self.hi)
        p = np.array([(self._expand(z[:, a["cols"]], a["kind"]) @ a["W"])[0] for a in self.axes])
        if self.field is not None:
            p = p + self._field(p[None, :], self.field, self.scale)[0]
        return p

    @property
    def ready(self):
        return self.axes is not None

    def mask_label(self, cols):
        extra = [n for n, grp in (("head", self.HEAD), ("bs", self.BS), ("ih", self.IH)) if grp[0] in cols]
        return "+".join(["eye"] + extra)

    @property
    def summary(self):
        if not self.axes:
            return ""
        s = " | ".join(f"{'xy'[i]}: {self.mask_label(a['cols'])} {a['kind']}@{a['alpha']:g}"
                       for i, a in enumerate(self.axes))
        if self.field is not None:
            s += f" +field(s={self.field['sigma']:g})"
        return s

    def to_dict(self):
        fld = None
        if self.field is not None:
            fld = dict(cen=self.field["cen"].tolist(), res=self.field["res"].tolist(),
                       cnt=self.field["cnt"].tolist(), sigma=self.field["sigma"], lam=self.field["lam"])
        return dict(mu=self.mu.tolist(), sd=self.sd.tolist(), cols=list(self.cols), scale=self.scale.tolist(),
                    cv_err=self.cv_err, cv_base=self.cv_base, cv_nested=self.cv_nested, field=fld,
                    lo=None if self.lo is None else self.lo.tolist(), hi=None if self.hi is None else self.hi.tolist(),
                    axes=[dict(cols=list(a["cols"]), kind=a["kind"], alpha=a["alpha"], W=a["W"].tolist()) for a in self.axes])

    def from_dict(self, d):
        gcols = d.get("cols", list(range(len(d["mu"]))))
        axes = [dict(cols=a.get("cols", gcols), kind=a["kind"], alpha=a["alpha"], W=np.array(a["W"]))
                for a in d["axes"]]
        fld = d.get("field")
        if fld is not None:
            fld = dict(cen=np.array(fld["cen"]), res=np.array(fld["res"]), cnt=np.array(fld["cnt"]),
                       sigma=fld["sigma"], lam=fld["lam"])
        self.mu, self.sd = np.array(d["mu"]), np.array(d["sd"])
        self.cols = d.get("cols", list(range(len(self.mu))))
        self.scale = np.array(d["scale"])
        self.axes, self.field = axes, fld
        self.cv_err, self.cv_base = d.get("cv_err"), d.get("cv_base")
        self.cv_nested = d.get("cv_nested")
        self.lo = np.array(d["lo"]) if d.get("lo") is not None else None
        self.hi = np.array(d["hi"]) if d.get("hi") is not None else None



def robust_mask(F):
    """Boolean mask of frames whose iris features are close to the per-dot median (rejects saccades, glitches)."""
    F = np.asarray(F)
    med = np.median(F, axis=0)
    mad = np.median(np.abs(F - med), axis=0) * 1.4826 + 1e-6
    return np.all(np.abs(F[:, :2] - med[:2]) <= 3 * mad[:2], axis=1)


def robust_filter(F):
    """Drop frames whose iris features are far from the per-dot median (saccades, glitches)."""
    F = np.asarray(F)
    return F[robust_mask(F)]


def adjacency(G, contig_groups):
    """Temporal neighbours of each contiguous-time-block group (used to leave neighbours out of CV)."""
    have = set(int(x) for x in np.unique(G))
    return {g: [h for h in (g - 1, g + 1) if h in have and h in contig_groups]
            for g in have if g in contig_groups}


MIXED_PHASE, POSTURE_PHASE = 4, 5   # see samples.PHASES
DOT_PHASES = (0, 3)        # phase codes of static-dot data ('dots', 'valid'), see samples.PHASES
WEIGHTS = "none"           # per-sample weights: 'none' or 'openness'
CLEAN = True               # clean static dots at fit time (saccade-latency drop + outlier rejection)


def clean_mask(A, adaptive_settle=True):
    """Keep-mask for static-dot frames. Per dot: the frames before the iris first settles near its final
    position (saccade latency after the dot jumped) are dropped, and so are outliers (>3 robust SD from
    the dot's settled position). Pursuit and head-phase frames are always kept."""
    F, G, PH = A["F"], A["G"], A["PH"]
    keep = np.ones(len(F), bool)
    for g in np.unique(G[np.isin(PH, DOT_PHASES)]):
        ix = np.flatnonzero(G == g)                      # arrays are in time order
        if len(ix) < 6:
            continue
        f = F[ix, :2]
        half = f[len(ix) // 2:]
        fin = np.median(half, 0)
        mad = np.maximum(1.4826 * np.median(np.abs(half - fin), 0), 0.003)
        ok = np.all(np.abs(f - fin) <= 3 * mad, axis=1)
        k = ok.copy()
        if adaptive_settle:
            for i in range(len(ix) - 2):
                if ok[i:i + 3].all():
                    k[:i] = False
                    break
        keep[ix] = k
    return keep


def feature_weights(F, mode):
    """Per-sample weights. 'openness': half-closed eyes hide the iris and give noisy offsets."""
    if mode == "openness":
        op = F[:, 2]
        return np.clip(op / (0.85 * np.median(op)), 0.15, 1.0)
    return None


def fit_pool(A, scale, nested=False, adj="auto", headcomp=None, model=None, weights=None, robust=None,
             clean=None, masks=None, kinds=None, prior=None):
    """Fit head compensation + gaze model from pooled training data (shared by the app and offline scripts).
    A = dict of per-sample arrays as stored in a session (F, S, G, V, K, PH, C, ...); K marks head-phase
    samples, V is the visit id (head phase), C marks contiguous time-block groups.
    adj: 'auto' -> neighbours of contiguous groups; None -> plain leave-one-group-out (old behaviour).
    Returns (headcomp, model)."""
    keep = np.asarray(A["PH"]) != POSTURE_PHASE          # posture-shift check data is NEVER used for fitting
    if CLEAN if clean is None else clean:
        keep &= clean_mask(A)
    if not keep.all():
        A = {k: np.asarray(v)[keep] if len(v) == len(keep) else v for k, v in A.items()}
    F, S, G, V, K = (np.asarray(A[k]) for k in ("F", "S", "G", "V", "K"))
    hc = headcomp or HeadComp()
    still = ~K & (np.asarray(A["PH"]) != MIXED_PHASE)    # head-still frames define the neutral pose
    if K.any() and still.any():
        hc.fit(F[K], V[K], F[still], B0=(prior or {}).get("head_B"))
    elif still.any():
        hc.ref = F[still][:, HeadComp.COLS].mean(0)
    m = model or GazeModel()
    if isinstance(adj, str):
        adj = adjacency(G, set(G[np.asarray(A["C"])].tolist()))
    Fh = hc.apply_many(F)
    m.fit(Fh, S, G, scale, adj=adj, nested=nested,
          weights=feature_weights(F, WEIGHTS if weights is None else weights), robust=robust,
          masks=masks, kinds=kinds, prior=prior)
    return hc, m


def load_prior(path):
    """Population prior from build_prior.py (JSON), or None if the file doesn't exist. With no prior the
    ridge shrinks toward zero exactly as before."""
    import json
    import os
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    if d.get("features") != GazeModel.FEATURES[:len(d.get("features", []))]:
        return None
    return dict(axes=d["axes"], head_B=np.array(d["head_B"]) if d.get("head_B") else None)
