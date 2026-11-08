"""
Snap the cursor onto nearby clickable things (buttons, links, checkboxes, tabs, icons, text
boxes, and every window's minimize / maximize / close buttons).

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
  - very large elements (full-width list rows, big panels, multi-line text areas) never snap
  - buttons: the cursor sits on the centre. Single-line text boxes: the cursor follows your gaze
    left/right inside the box (on its middle line), so you can still put the caret anywhere
  - caption buttons: apps that draw their own title bar (VS Code, Chrome, Discord, Electron)
    don't expose minimize/maximize/close to UI Automation, so for the window under the gaze
    they're added from the window's styles, bounds and DPI (standard 46x32 px at 100% scale)

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
MAX_FIELD_W = 2600       # text boxes may be wide (address bars, search fields)...
MAX_FIELD_H = 90         # ...but only single-line ones snap; text areas stay free for selecting
FIELD_INSET_PX = 10      # keep the cursor this far inside a snapped text box's ends
CAPTION_W, CAPTION_H = 46, 32   # Windows caption button size at 100% scale
PROBE_R = 50             # UIA is probed at the gaze point and 8 points this far around it
SCAN_PERIOD_S = 0.05     # pause between scans (one scan takes ~9 x 15 ms)

CLICKABLE = {
    "ButtonControl", "HyperlinkControl", "CheckBoxControl", "RadioButtonControl",
    "TabItemControl", "MenuItemControl", "SplitButtonControl", "ComboBoxControl",
    "ListItemControl", "TreeItemControl",
}
FIELDS = {"EditControl"}  # snap targets where the cursor moves freely along the box
CAPTION = "CaptionButton"  # synthetic minimize/maximize/close targets
TEXTY = {"EditControl", "DocumentControl", "TextControl"}


def _target_ok(typ, w, h):
    if typ in FIELDS:
        return 0 < w <= MAX_FIELD_W and 0 < h <= MAX_FIELD_H
    return 0 < w <= MAX_TARGET_W and 0 < h <= MAX_TARGET_H


_W = None


def _win32():
    """ctypes handles with proper signatures (set up once)."""
    global _W
    if _W is None:
        import ctypes
        import ctypes.wintypes as wt
        u32, dwm, shc = ctypes.windll.user32, ctypes.windll.dwmapi, ctypes.windll.shcore
        u32.WindowFromPoint.restype, u32.WindowFromPoint.argtypes = wt.HWND, [wt.POINT]
        u32.GetAncestor.restype, u32.GetAncestor.argtypes = wt.HWND, [wt.HWND, ctypes.c_uint]
        u32.MonitorFromWindow.restype, u32.MonitorFromWindow.argtypes = wt.HMONITOR, [wt.HWND, wt.DWORD]
        u32.GetWindowLongW.restype, u32.GetWindowLongW.argtypes = ctypes.c_long, [wt.HWND, ctypes.c_int]
        dwm.DwmGetWindowAttribute.argtypes = [wt.HWND, wt.DWORD, ctypes.c_void_p, wt.DWORD]
        _W = (ctypes, wt, u32, dwm, shc)
    return _W


_SKIP_CLASSES = {"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd", "tooltips_class32",
                 "TaskListThumbnailWnd", "Windows.UI.Core.CoreWindow"}


def caption_targets(x, y, near_px, max_windows=6):
    """Minimize/maximize/close rects of every visible window whose caption buttons are within
    near_px of the gaze (x, y). Computed from the window itself, because many apps (VS Code,
    Chrome, Discord, Electron) don't expose them to UI Automation. Only buttons that are really
    on top at that spot are returned (a window behind another one doesn't count)."""
    ctypes, wt, u32, dwm, shc = _win32()
    cands = []

    def visit(h, _):
        if len(cands) >= max_windows * 4:
            return False
        if not u32.IsWindowVisible(h) or u32.IsIconic(h):
            return True
        style = u32.GetWindowLongW(h, -16)                    # GWL_STYLE
        if style & 0x40000000 or not style & 0x000B0000:      # WS_CHILD, or no SYSMENU/MIN/MAX box
            return True                                       # (custom-frame apps like VS Code drop WS_SYSMENU)
        cloaked = ctypes.c_int(0)                             # hidden by DWM (other virtual desktop, suspended app)
        if dwm.DwmGetWindowAttribute(h, 14, ctypes.byref(cloaked), 4) == 0 and cloaked.value:
            return True
        cands.append((h, style))
        return True

    u32.EnumWindows(ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)(visit), 0)

    out, used = [], 0
    for h, style in cands:                                     # top-most first
        if used >= max_windows:
            break
        cls = ctypes.create_unicode_buffer(64)
        u32.GetClassNameW(h, cls, 64)
        if cls.value in _SKIP_CLASSES:
            continue
        r = wt.RECT()
        if dwm.DwmGetWindowAttribute(h, 9, ctypes.byref(r), ctypes.sizeof(r)) != 0:
            u32.GetWindowRect(h, ctypes.byref(r))              # 9 = DWMWA_EXTENDED_FRAME_BOUNDS (visible frame)
        if r.right - r.left < 200 or r.bottom - r.top < 60:
            continue
        # Scale from the MONITOR, not the window: a DPI-unaware app reports 96 dpi, but Windows
        # stretches it to the monitor's scale, so its buttons are drawn at that size.
        dx, dy = wt.UINT(96), wt.UINT(96)
        try:
            shc.GetDpiForMonitor(u32.MonitorFromWindow(h, 2), 0, ctypes.byref(dx), ctypes.byref(dy))
        except Exception:
            pass
        sc = (dx.value or 96) / 96.0
        bw, bh = round(CAPTION_W * sc), round(CAPTION_H * sc)
        right, top = r.right, r.top
        if rect_dist((x, y), (right - 3 * bw, top, right, top + bh)) > near_px:
            continue
        used += 1
        rects = [(right - bw, top, right, top + bh)]           # close
        if style & 0x00030000:                                 # WS_MINIMIZEBOX | WS_MAXIMIZEBOX
            rects.append((right - 2 * bw, top, right - bw, top + bh))
            rects.append((right - 3 * bw, top, right - 2 * bw, top + bh))
        for rc in rects:                                       # only if this window is on top there
            cx, cy = rect_center(rc)
            hit = u32.WindowFromPoint(wt.POINT(int(cx), int(cy)))
            if hit and u32.GetAncestor(hit, 2) == h:            # GA_ROOT
                out.append(rc)
    return out


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
                if el.ControlTypeName in CLICKABLE or el.ControlTypeName in FIELDS:
                    hit = el
                    break
                el = el.GetParentControl()
            accepted = False
            if hit is not None:
                r = hit.BoundingRectangle
                rect = (r.left, r.top, r.right, r.bottom)
                typ = hit.ControlTypeName
                if _target_ok(typ, rect[2] - rect[0], rect[3] - rect[1]):
                    found[rect] = typ
                    accepted = True
            if i == 0:                      # on page text / a big text area: no capture margin
                on_text = first_type in TEXTY and not accepted
        # Caption buttons the app didn't expose (a real UIA button at the same spot wins).
        for rect in caption_targets(x0, y0, SNAP_OUT_PX):
            c = rect_center(rect)
            if not any(rect_dist(c, fr) == 0 for fr in found):
                found[rect] = CAPTION
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

    def point(self, gaze):
        """Where the cursor goes while snapped: a button's centre, or for a text box the gaze
        slid along the box's middle line (so you can still aim at any spot in the text)."""
        if self.current is None:
            return None
        (l, t, r, b), typ = self.current
        if typ in FIELDS and gaze is not None:
            inset = min(FIELD_INSET_PX, (r - l) / 4)
            return (min(max(gaze[0], l + inset), r - inset), (t + b) / 2.0)
        return rect_center(self.current[0])

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
