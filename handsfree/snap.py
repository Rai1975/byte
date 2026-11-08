"""
Snap the cursor onto nearby clickable things (buttons, links, checkboxes, tabs, icons...).

Snapper (background thread) asks Windows UI Automation what is on screen around the gaze point
and publishes the clickable elements it finds. SnapLogic (main thread) decides whether the
cursor should lock onto one of them:

  - snap IN when the gaze comes within SNAP_IN_PX of a target's edge for SNAP_DWELL_S
    (a brief flick out of the zone, up to PENDING_GRACE_S, doesn't restart the dwell)
  - stay snapped until the gaze has been more than SNAP_OUT_PX from it for RELEASE_S
    (hysteresis in space and time: jitter doesn't unsnap you, a deliberate look away does)
  - switch to another target only when the gaze is right on top of it for SWITCH_DWELL_S
  - when the gaze is on plain text (documents, text boxes, page text) the capture zone
    shrinks to the target itself, so you can still point at text freely
  - very large elements (full-width list rows, big panels) never snap

Snapping is skipped entirely while dragging/scrolling/zooming or in the action menu (actions.py).
"""

import math
import threading
import time

# ----------------------------------------------------------------------------- tuning
SNAP_IN_PX = 60          # capture distance from a target's edge
SNAP_OUT_PX = 130        # release distance (must be > SNAP_IN_PX)
SNAP_DWELL_S = 0.12      # gaze must stay near a candidate this long before snapping
SWITCH_DWELL_S = 0.30    # ...and right on top of a different one this long to switch to it
RELEASE_S = 0.30         # gaze must stay beyond SNAP_OUT_PX this long before unsnapping
PENDING_GRACE_S = 0.15   # while not snapped, a candidate survives this long out of the zone while dwelling
MAX_TARGET_W = 420       # bigger elements are not snap targets
MAX_TARGET_H = 160
PROBE_R = 50             # UIA is probed at the gaze point and 8 points this far around it
SCAN_PERIOD_S = 0.05     # pause between scans (one scan takes ~9 x 15 ms)

CLICKABLE = {
    "ButtonControl", "HyperlinkControl", "CheckBoxControl", "RadioButtonControl",
    "TabItemControl", "MenuItemControl", "SplitButtonControl", "ComboBoxControl",
    "ListItemControl", "TreeItemControl",
}
TEXTY = {"EditControl", "DocumentControl", "TextControl"}


def rect_dist(p, r):
    """Distance from point p to rectangle r=(l, t, r, b); 0 inside."""
    dx = max(r[0] - p[0], 0, p[0] - r[2])
    dy = max(r[1] - p[1], 0, p[1] - r[3])
    return math.hypot(dx, dy)


def rect_center(r):
    return ((r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0)


class Snapper(threading.Thread):
    """Background UIA scanner. Set `query` to the gaze point (or None to idle)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.query = None
        self.alive = True
        self.status = "starting"
        self.result = ([], False, 0.0)    # (candidate rects, gaze is on text, scan time)

    def run(self):
        try:
            import uiautomation as auto
        except ImportError:
            self.status = "uiautomation not installed (conda env update -f environment.yml)"
            return
        with auto.UIAutomationInitializerInThread():
            self.status = "ready"
            while self.alive:
                q = self.query
                if q is None:
                    time.sleep(0.05)
                    continue
                try:
                    self.result = self._scan(auto, q)
                except Exception as e:     # COM hiccups (window closed mid-query etc.)
                    self.status = f"scan error: {e.__class__.__name__}"
                    time.sleep(0.2)
                    continue
                self.status = "ready"
                time.sleep(SCAN_PERIOD_S)

    def _scan(self, auto, q):
        x0, y0 = int(q[0]), int(q[1])
        probes = [(x0, y0)] + [(int(x0 + PROBE_R * math.cos(a)), int(y0 + PROBE_R * math.sin(a)))
                               for a in [i * math.pi / 4 for i in range(8)]]
        found, on_text = {}, False
        for i, (x, y) in enumerate(probes):
            el = auto.ControlFromPoint(x, y)
            first_type = el.ControlTypeName if el is not None else ""
            hit = None
            for _ in range(3):              # the element itself, then up to 2 parents
                if el is None:
                    break
                if el.ControlTypeName in CLICKABLE:
                    hit = el
                    break
                el = el.GetParentControl()
            if i == 0 and hit is None:
                on_text = first_type in TEXTY
            if hit is not None:
                r = hit.BoundingRectangle
                rect = (r.left, r.top, r.right, r.bottom)
                w, h = rect[2] - rect[0], rect[3] - rect[1]
                if 0 < w <= MAX_TARGET_W and 0 < h <= MAX_TARGET_H:
                    found[rect] = hit.ControlTypeName
        return list(found.items()), on_text, time.time()

    def close(self):
        self.alive = False


class SnapLogic:
    """Main-thread snap decision with in/out hysteresis and dwell."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.current = None        # (rect, type) we're snapped to
        self.pending = None        # [rect, type, since, last_seen]
        self.out_since = None      # when the gaze left the current target's SNAP_OUT_PX zone

    @property
    def center(self):
        return rect_center(self.current[0]) if self.current else None

    def update(self, gaze, result, now):
        cands, on_text, _ = result
        cur = self.current
        if cur:
            if rect_dist(gaze, cur[0]) > SNAP_OUT_PX:
                if self.out_since is None:
                    self.out_since = now
                elif now - self.out_since >= RELEASE_S:
                    self.current = cur = None
                    self.out_since = None
            else:
                self.out_since = None
        margin = 0.0 if on_text else SNAP_IN_PX
        best, best_d = None, 1e9
        for rect, typ in cands:
            d = rect_dist(gaze, rect)
            if cur is not None:
                if rect == cur[0] or d > 0:    # while snapped, only a target under the gaze can steal it
                    continue
            elif d > margin:
                continue
            area = (rect[2] - rect[0]) * (rect[3] - rect[1])
            score = d + area * 1e-6            # tie-break: smaller target wins
            if score < best_d:
                best, best_d = (rect, typ), score
        pend = self.pending
        if best is None:
            if pend and (cur is not None or now - pend[3] > PENDING_GRACE_S):
                self.pending = None        # switching needs a continuous look; grace only for the first snap
        elif pend and pend[0] == best[0]:
            pend[3] = now
            if now - pend[2] >= (SWITCH_DWELL_S if cur else SNAP_DWELL_S):
                self.current, self.pending, self.out_since = best, None, None
        else:
            self.pending = [best[0], best[1], now, now]
        return self.center
