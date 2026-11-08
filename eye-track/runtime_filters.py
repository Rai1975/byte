"""Runtime signal processing for the cursor: One Euro smoothing, fixation hold, online refinement.

No camera/UI dependencies, so eval_runtime.py can replay saved sessions through the same code.
"""

import math
from collections import deque

import numpy as np


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


class FixationHold:
    """Velocity-threshold fixation detector (I-VT) that holds the cursor still during fixations.

    Input is the already smoothed gaze point. When the gaze speed (displacement over `window` seconds)
    stays below `v_enter` for `min_s`, a fixation starts and the output is pinned to the running mean of
    the points seen since it began (so small residual drift and jitter vanish). The fixation ends the
    moment the gaze speed exceeds `v_exit` or the point leaves `radius` px around the anchor; the output
    then follows the input again with no added lag. Thresholds are given in px / px per second and scaled
    by screen width / 1920 by the caller.
    """

    def __init__(self, v_enter=200.0, v_exit=500.0, radius=90.0, min_s=0.15, window=0.15):
        self.v_enter, self.v_exit, self.radius, self.min_s, self.window = v_enter, v_exit, radius, min_s, window
        self.reset()

    def reset(self):
        self.hist = deque()
        self.anchor = None
        self.n = 0
        self.slow_since = None
        self.last_t = None

    def _speed(self, p, t):
        while self.hist and t - self.hist[0][0] > self.window:
            self.hist.popleft()
        if not self.hist:
            return 0.0
        t0, p0 = self.hist[0]
        return float(np.linalg.norm(p - p0)) / max(t - t0, 1e-3)

    @property
    def fixating(self):
        return self.anchor is not None

    def update(self, p, t):
        """-> (output point, fixating)"""
        p = np.asarray(p, float)
        if self.last_t is not None and t - self.last_t > 0.5:      # tracking gap (blink, talking pause)
            self.reset()
        self.last_t = t
        v = self._speed(p, t)
        self.hist.append((t, p))
        if self.anchor is not None:
            if v > self.v_exit or np.linalg.norm(p - self.anchor) > self.radius:
                self.anchor, self.n, self.slow_since = None, 0, None
                return p, False
            self.n += 1
            self.anchor += (p - self.anchor) / min(self.n, 30)      # slow running mean
            return self.anchor.copy(), True
        if v < self.v_enter:
            if self.slow_since is None:
                self.slow_since = t
                self.sum = p.copy()
                self.cnt = 1
            else:
                self.sum += p
                self.cnt += 1
            if t - self.slow_since >= self.min_s:
                self.anchor = self.sum / self.cnt
                self.n = self.cnt
                return self.anchor.copy(), True
        else:
            self.slow_since = None
        return p, False


class OnlineRefiner:
    """Optional online refinement by recursive least squares (OFF by default).

    Learns a small correction  delta = theta^T [1, x/sw, y/sh]  (px) that is added to the model's gaze
    prediction, from (prediction, true location) pairs. The prior is delta = 0 (ridge strength `ridge`),
    `forget` < 1 lets it follow slow drift, and the correction is clamped to `max_px`.

    CAUTION: the true location must be independent of the cursor. If the cursor already follows the
    prediction, a click at the cursor says nothing. It is informative when the click lands somewhere else
    (a physical mouse correction, a snap-to-target, or a deliberate click on a known button).
    """

    def __init__(self, sw, sh, forget=0.995, ridge=5.0, max_px=150.0):
        self.sw, self.sh, self.forget, self.max_px = sw, sh, forget, max_px
        self.theta = np.zeros((3, 2))
        self.P = np.eye(3) / ridge                      # prior covariance: correction = 0, strength 1/ridge
        self.n = 0

    def _phi(self, p):
        return np.array([1.0, p[0] / self.sw, p[1] / self.sh])

    def correction(self, p):
        d = self._phi(p) @ self.theta
        n = float(np.linalg.norm(d))
        return d if n <= self.max_px else d * (self.max_px / n)

    def update(self, pred, true):
        """pred = model prediction at the time of the event (before correction), true = actual location."""
        phi = self._phi(np.asarray(pred, float))
        err = np.asarray(true, float) - np.asarray(pred, float) - phi @ self.theta
        Pphi = self.P @ phi
        k = Pphi / (self.forget + phi @ Pphi)
        self.theta += np.outer(k, err)
        self.P = (self.P - np.outer(k, Pphi)) / self.forget
        self.n += 1
