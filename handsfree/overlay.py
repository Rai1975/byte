"""
Transparent, click-through, always-on-top window that draws on top of everything:
the snap highlight, the action menu, the scroll gauge, the zoom view and mode banners.

It runs its own Tk thread. The app only calls `overlay.set(state_dict)`; the Tk thread picks
the newest state up ~30 times a second and redraws. The window never takes focus and never
receives clicks (WS_EX_TRANSPARENT | WS_EX_NOACTIVATE), so it can't get in the way.
"""

import ctypes
import threading
import time

KEY = "#ff00fe"            # this exact colour is see-through
ACCENT = "#3fb8ff"
SNAP_COLOR = "#36e07a"
DARK = "#202428"
WHITE = "#f2f2f2"
SEL = "#2f7de1"
FONT = ("Segoe UI", 13, "bold")
FONT_SMALL = ("Segoe UI", 11)


class Overlay(threading.Thread):
    def __init__(self, sw, sh):
        super().__init__(daemon=True)
        self.sw, self.sh = sw, sh
        self.state = {}
        self.alive = True
        self._drawn = None
        self._zoom_key = None
        self._zoom_img = None

    def set(self, state):
        self.state = state          # replaced atomically; never mutated after hand-off

    def close(self):
        self.alive = False
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=1.5)      # let the Tk thread tear itself down (Tk objects must die there)

    # ------------------------------------------------------------------ Tk thread
    def run(self):
        import gc
        import tkinter as tk
        u32 = ctypes.windll.user32
        prev_fg = u32.GetForegroundWindow()
        self.tk = tk
        root = self.root = tk.Tk()
        root.overrideredirect(True)
        root.geometry(f"{self.sw}x{self.sh}+0+0")
        root.configure(bg=KEY)
        root.attributes("-topmost", True)
        root.attributes("-transparentcolor", KEY)
        self.canvas = tk.Canvas(root, width=self.sw, height=self.sh, bg=KEY, highlightthickness=0)
        self.canvas.pack()
        root.update_idletasks()
        self._click_through()
        root.update()
        if prev_fg:                     # Tk activates its window when it first appears: give focus back
            u32.SetForegroundWindow(prev_fg)
        self._last_top = time.time()
        root.after(30, self._tick)
        root.mainloop()
        self._zoom_img = self.canvas = self.root = self.tk = None
        root = None
        gc.collect()                    # free Tcl objects in this thread, not at interpreter exit

    def _click_through(self):
        try:
            u32 = ctypes.windll.user32
            hwnd = u32.GetParent(self.root.winfo_id())
            GWL_EXSTYLE = -20
            style = u32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            style |= 0x80000 | 0x20 | 0x80 | 0x08000000 | 0x8   # LAYERED TRANSPARENT TOOLWINDOW NOACTIVATE TOPMOST
            u32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
        except Exception:
            pass

    def _tick(self):
        if not self.alive:
            self.root.destroy()
            return
        st = self.state
        if st is not self._drawn:
            self._draw(st)
            self._drawn = st
        if time.time() - self._last_top > 2:      # stay above windows that grab topmost later
            self.root.attributes("-topmost", True)
            self._last_top = time.time()
        self.root.after(30, self._tick)

    def _draw(self, st):
        c = self.canvas
        c.delete("all")
        if "zoom" in st:
            self._draw_zoom(st["zoom"])
        if "snap" in st:
            l, t, r, b = st["snap"]
            c.create_rectangle(l - 4, t - 4, r + 4, b + 4, outline=SNAP_COLOR, width=3)
        if "menu" in st:
            self._draw_menu(st["menu"])
        if "scroll" in st:
            self._draw_scroll(st["scroll"])
        if "dot" in st:
            x, y = st["dot"]
            c.create_oval(x - 9, y - 9, x + 9, y + 9, outline=WHITE, width=2, fill=ACCENT)
        if st.get("banner"):
            self._label(self.sw // 2, 28, st["banner"])

    def _label(self, x, y, text, fill=DARK, font=FONT):
        c = self.canvas
        t = c.create_text(x, y, text=text, fill=WHITE, font=font)
        l, tp, r, b = c.bbox(t)
        bg = c.create_rectangle(l - 12, tp - 6, r + 12, b + 6, fill=fill, outline=ACCENT, width=2)
        c.tag_lower(bg, t)

    def _draw_menu(self, m):
        c = self.canvas
        cx, cy = m["center"]
        R, r = m["radius"], m["item_r"]
        c.create_oval(cx - R - r - 6, cy - R - r - 6, cx + R + r + 6, cy + R + r + 6,
                      outline=ACCENT, width=2, dash=(4, 6))
        for i, (label, (ix, iy)) in enumerate(m["items"]):
            fill = SEL if m["sel"] == i else DARK
            c.create_oval(ix - r, iy - r, ix + r, iy + r, fill=fill, outline=ACCENT, width=2)
            c.create_text(ix, iy, text=label, fill=WHITE, font=FONT, width=2 * r - 10, justify="center")
        cr = m["dead"]
        fill = SEL if m["sel"] is None else DARK
        c.create_oval(cx - cr, cy - cr, cx + cr, cy + cr, fill=fill, outline=ACCENT, width=2)
        c.create_text(cx, cy, text="Cancel", fill=WHITE, font=FONT_SMALL)
        self._label(cx, cy + R + r + 30, "look at an action, then 1 tap", font=FONT_SMALL)

    def _draw_scroll(self, s):
        c = self.canvas
        x, y = s["center"]
        full, dead = s["full"], s["dead"]
        c.create_rectangle(x - 8, y - full, x + 8, y + full, fill=DARK, outline=ACCENT, width=2)
        c.create_rectangle(x - 8, y - dead, x + 8, y + dead, fill="#3a3f44", outline="")
        c.create_polygon(x, y - full - 22, x - 14, y - full - 4, x + 14, y - full - 4, fill=ACCENT)
        c.create_polygon(x, y + full + 22, x - 14, y + full + 4, x + 14, y + full + 4, fill=ACCENT)
        gy = max(-full, min(full, s["dy"]))
        c.create_oval(x - 11, y + gy - 11, x + 11, y + gy + 11, fill=SEL if abs(gy) > dead else WHITE,
                      outline=WHITE, width=2)

    def _draw_zoom(self, z):
        c = self.canvas
        key = z["region"]
        if key != self._zoom_key:
            self._zoom_img = self._grab(z)
            self._zoom_key = key
        dl, dt, dr, db = z["disp"]
        if self._zoom_img is not None:
            c.create_image(dl, dt, image=self._zoom_img, anchor="nw")
        c.create_rectangle(dl - 3, dt - 3, dr + 3, db + 3, outline=ACCENT, width=3)
        if z.get("gaze"):
            gx, gy = z["gaze"]
            c.create_line(gx - 18, gy, gx + 18, gy, fill="#ff3b3b", width=2)
            c.create_line(gx, gy - 18, gx, gy + 18, fill="#ff3b3b", width=2)
        self._label((dl + dr) // 2, db + 26, "ZOOM: look at the spot  -  1 tap click, 2 taps double click, 3 cancel",
                    font=FONT_SMALL)

    def _grab(self, z):
        """Screenshot the source region with the overlay hidden, scaled up for display."""
        try:
            from PIL import ImageGrab, ImageTk
            self.canvas.delete("all")
            self.root.update()
            time.sleep(0.06)            # let the compositor drop our menu before the screenshot
            img = ImageGrab.grab(bbox=z["region"])
            dl, dt, dr, db = z["disp"]
            img = img.resize((dr - dl, db - dt))
            return ImageTk.PhotoImage(img)
        except Exception:
            return None
