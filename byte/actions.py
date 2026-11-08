"""
What jaw taps do. Gesture map:

  1 tap   left click
  2 taps  double click
  3 taps  action menu at the cursor, "Right click" highlighted. Taps only, no aiming:
            1 tap   move the highlight clockwise:
                    Right click -> Drag / select -> Scroll -> Zoom -> Cancel -> Right click ...
            2 taps  choose the highlighted option
            3 taps  cancel (it also closes by itself MENU_TIMEOUT_S after the last tap)
          Options:
            Right click     right click where the menu was opened
            Drag / select   press and hold there, move with your eyes, any tap drops
                            (drag-and-drop, text selection, window dragging)
            Scroll          look above / below the menu point to scroll, any tap stops
            Zoom            magnified view of the area; look at the exact spot, then
                            1 tap = click, 2 taps = double click, 3 taps = cancel

With MENU_ENABLED = False, 3 taps instead hold the left button directly (a tap drops it).

Accuracy helpers used for every action:
  - the click lands where you were looking just BEFORE the first tap (gaze averaged over
    AIM_WINDOW_S), not where the head jolt from the tap moved it
  - the cursor freezes from the first tap until the click is done (+ FREEZE_AFTER_S)

Mouse button and scroll events are sent from the app's main thread.
"""

import math

import numpy as np
from pynput.mouse import Button, Controller

# ----------------------------------------------------------------------------- tuning
AIM_WINDOW_S = (0.35, 0.03)  # average gaze from 350 ms to 30 ms before the first tap
FREEZE_AFTER_S = 0.2         # keep the cursor still this long after a click
FREEZE_MAX_S = 1.5           # safety: never freeze longer than this waiting for a gesture
MENU_ENABLED = True          # False: 3 taps hold the left button directly instead of the menu
MENU_RADIUS = 140            # centre -> action button distance (px)
MENU_ITEM_R = 62             # action button radius
MENU_DEAD = 50               # centre "Cancel" radius
MENU_TIMEOUT_S = 10.0        # menu closes this long after the last tap
MENU_ITEMS = [("Right click", -90), ("Drag / select", 0), ("Scroll", 90), ("Zoom", 180)]  # angle, 0 = right
SCROLL_DEAD_PX = 60          # gaze within this of the menu point = no scrolling
SCROLL_FULL_PX = 320         # gaze this far away = full speed
SCROLL_MAX_RATE = 14.0       # wheel notches per second at full speed
ZOOM_SRC = (240, 160)        # size of the magnified area (px)
ZOOM_SCALE = 3


class Actions:
    def __init__(self, sw, sh, cursor):
        self.sw, self.sh = sw, sh
        self.cursor = cursor            # CursorDriver: jumped directly when we click somewhere
        self.mouse = Controller()
        self.mode = "normal"            # normal | menu | drag | scroll | zoom
        self.point = None               # where the menu was opened (the real target)
        self.menu_center = None         # menu drawn here (moved so it fits on screen)
        self.menu_t = 0.0
        self.sel = None
        self.zoom = None
        self.scroll_dy = 0.0
        self._scroll_acc = 0.0
        self._last_frame = None
        self.hold = None                # cursor frozen here...
        self.hold_until = 0.0           # ...until this time
        self._now = 0.0

    # ------------------------------------------------------------------ helpers
    def _clamp(self, p):
        return (float(min(max(p[0], 0), self.sw - 1)), float(min(max(p[1], 0), self.sh - 1)))

    def _move(self, p):
        p = self._clamp(p)
        self.cursor._pos = np.array(p, dtype=float)     # so the glide doesn't drag it back
        try:
            self.mouse.position = (int(p[0]), int(p[1]))
        except Exception:
            pass
        return p

    def _freeze(self, p, seconds):
        self.hold, self.hold_until = p, self._now + seconds

    def _menu_geometry(self):
        cx, cy = self.menu_center
        return [(label, (cx + MENU_RADIUS * math.cos(math.radians(a)),
                         cy + MENU_RADIUS * math.sin(math.radians(a)))) for label, a in MENU_ITEMS]

    def _menu_pick(self, p):
        if p is None:
            return self.sel
        dx, dy = p[0] - self.menu_center[0], p[1] - self.menu_center[1]
        if math.hypot(dx, dy) < MENU_DEAD:
            return None
        ang = math.degrees(math.atan2(dy, dx))
        diffs = [abs((ang - a + 180) % 360 - 180) for _, a in MENU_ITEMS]
        return int(np.argmin(diffs))         # 4 wide 90-degree sectors: easy to hit

    def _zoom_setup(self):
        w, h = ZOOM_SRC
        x0 = int(min(max(self.point[0] - w / 2, 0), self.sw - w))
        y0 = int(min(max(self.point[1] - h / 2, 0), self.sh - h))
        W, H = w * ZOOM_SCALE, h * ZOOM_SCALE
        dx = int(min(max(self.point[0] - W / 2, 0), self.sw - W))
        dy = int(min(max(self.point[1] - H / 2, 0), self.sh - H))
        self.zoom = dict(region=(x0, y0, x0 + w, y0 + h), disp=(dx, dy, dx + W, dy + H))

    def _zoom_map(self, p):
        """Gaze point on the magnified view -> real screen point (None if off the view)."""
        (x0, y0, _, _), (dl, dt, dr, db) = self.zoom["region"], self.zoom["disp"]
        m = 40
        if p is None or not (dl - m <= p[0] <= dr + m and dt - m <= p[1] <= db + m):
            return None
        px = min(max(p[0], dl), dr - 1)
        py = min(max(p[1], dt), db - 1)
        return (x0 + (px - dl) / ZOOM_SCALE, y0 + (py - dt) / ZOOM_SCALE)

    def _to_normal(self):
        self.mode, self.sel, self.zoom = "normal", None, None

    # ------------------------------------------------------------------ events
    def tap_started(self, aim, now):
        """First tap of a gesture: freeze the cursor on the aim point (normal/drag modes)."""
        self._now = now
        if self.mode in ("normal", "drag") and aim is not None:
            self._freeze(aim, FREEZE_MAX_S)

    def gesture(self, g, aim, now):
        """Run gesture g (1/2/3) at aim. Returns a short label for the HUD."""
        self._now = now
        mode = self.mode
        if mode == "normal":
            if aim is None:
                return "no aim point"
            p = self._move(aim)
            if g == 1:
                self.mouse.click(Button.left, 1)
                label = "CLICK"
            elif g == 2:
                self.mouse.click(Button.left, 2)
                label = "DOUBLE CLICK"
            elif not MENU_ENABLED:          # 3 taps: hold the left button here; a tap drops it
                self.mouse.press(Button.left)
                self.mode, self.point = "drag", p
                self._freeze(p, FREEZE_AFTER_S)   # settle briefly, then the cursor follows your eyes
                return "HOLD (1 tap drops)"
            else:
                self.mode, self.point, self.menu_t, self.sel = "menu", p, now, 0   # "Right click" highlighted
                m = MENU_RADIUS + MENU_ITEM_R + 10
                self.menu_center = (min(max(p[0], m), self.sw - m), min(max(p[1], m), self.sh - m))
                return "MENU"
            self._freeze(p, FREEZE_AFTER_S)
            return label

        if mode == "menu":
            self.menu_t = now                       # each tap restarts the timeout
            if g == 1:                              # next option, clockwise; Cancel sits after Zoom
                self.sel = None if self.sel == len(MENU_ITEMS) - 1 else (
                    0 if self.sel is None else self.sel + 1)
                return "next: " + ("Cancel" if self.sel is None else MENU_ITEMS[self.sel][0])
            pick = self.sel if g == 2 else None     # 2 taps choose, 3 taps cancel
            if pick is None:
                self._to_normal()
                return "menu cancelled"
            name = MENU_ITEMS[pick][0]
            self._move(self.point)
            if name == "Right click":
                self._to_normal()
                self.mouse.click(Button.right, 1)
                self._freeze(self.point, FREEZE_AFTER_S)
                return "RIGHT CLICK"
            if name == "Drag / select":
                self.mouse.press(Button.left)
                self.mode = "drag"
                return "DRAG (any tap drops)"
            if name == "Scroll":
                self.mode, self._scroll_acc, self.scroll_dy = "scroll", 0.0, 0.0
                return "SCROLL (any tap stops)"
            self.mode = "zoom"
            self._zoom_setup()
            return "ZOOM"

        if mode == "drag":
            p = self._move(aim) if aim is not None else None
            self.mouse.release(Button.left)
            self._to_normal()
            if p:
                self._freeze(p, FREEZE_AFTER_S)
            return "DROP"

        if mode == "scroll":
            self._to_normal()
            return "scroll off"

        if mode == "zoom":
            target = self._zoom_map(aim) if g in (1, 2) else None
            self._to_normal()
            if target is None:
                return "zoom cancelled"
            p = self._move(target)
            self.mouse.click(Button.left, g)
            self._freeze(p, FREEZE_AFTER_S)
            return "ZOOM CLICK" if g == 1 else "ZOOM DOUBLE CLICK"
        return ""

    def cancel(self, why=""):
        """Leave any mode; drop a drag in progress (used when talking starts, on quit, etc.)."""
        if self.mode == "drag":
            try:
                self.mouse.release(Button.left)
            except Exception:
                pass
        was = self.mode
        self._to_normal()
        self.hold = None
        return was != "normal"

    # ------------------------------------------------------------------ per frame
    def hold_point(self, now):
        """Where the cursor must stay right now, or None to follow gaze / snapping."""
        if self.mode in ("menu", "scroll", "zoom"):
            return self.point
        if self.hold is not None and now < self.hold_until:
            return self.hold
        self.hold = None
        return None

    def frame(self, gaze, now, paused):
        dt = 0.0 if self._last_frame is None else min(now - self._last_frame, 0.1)
        self._last_frame = now
        if self.mode == "menu":                     # the highlight moves with taps, not gaze
            if now - self.menu_t > MENU_TIMEOUT_S:
                self._to_normal()
        elif self.mode == "scroll" and gaze is not None:
            self.scroll_dy = gaze[1] - self.point[1]
            if paused:
                return
            excess = abs(self.scroll_dy) - SCROLL_DEAD_PX
            if excess > 0:
                rate = SCROLL_MAX_RATE * min(1.0, excess / (SCROLL_FULL_PX - SCROLL_DEAD_PX))
                self._scroll_acc += rate * dt
                n = int(self._scroll_acc)
                if n:
                    self._scroll_acc -= n
                    self.mouse.scroll(0, -n if self.scroll_dy > 0 else n)   # look down = scroll down

    def overlay_state(self, gaze):
        st = {}
        if self.mode == "menu":
            st["menu"] = dict(center=self.menu_center, items=self._menu_geometry(), sel=self.sel,
                              radius=MENU_RADIUS, item_r=MENU_ITEM_R, dead=MENU_DEAD)
        elif self.mode == "scroll":
            st["scroll"] = dict(center=self.point, dy=self.scroll_dy, dead=SCROLL_DEAD_PX, full=SCROLL_FULL_PX)
            st["banner"] = "SCROLL: look up / down  -  any tap stops"
        elif self.mode == "zoom":
            st["zoom"] = dict(self.zoom, gaze=gaze)
        elif self.mode == "drag":
            st["banner"] = "HOLDING: move with your eyes  -  1 tap drops"
        return st
