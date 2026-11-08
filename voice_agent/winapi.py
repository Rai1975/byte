"""Low-level Windows helpers: SendInput keyboard, window lookup/focus.

Deliberately NO mouse input anywhere: the eye tracker owns the cursor.
Deliberately NOT pyautogui: its fail-safe raises when the cursor sits in a screen corner,
which an eye-driven cursor does all the time.
"""

import ctypes
import ctypes.wintypes as wt
import time

import psutil
import win32con
import win32gui
import win32process

user32 = ctypes.WinDLL("user32", use_last_error=True)

INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x1
KEYEVENTF_KEYUP = 0x2
KEYEVENTF_UNICODE = 0x4

ULONG_PTR = ctypes.c_size_t


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD),
                ("time", wt.DWORD), ("dwExtraInfo", ULONG_PTR)]


class MOUSEINPUT(ctypes.Structure):          # only here so the union has the right size
    _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD), ("dwFlags", wt.DWORD),
                ("time", wt.DWORD), ("dwExtraInfo", ULONG_PTR)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wt.DWORD), ("u", _INPUTUNION)]


user32.SendInput.argtypes = [wt.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = wt.UINT

# Name -> virtual-key code. F4-F10 are listed so they can be recognised and refused
# (see BLOCKED_KEYS); the hands-free mouse app uses them as global hotkeys.
VK = {
    "backspace": 0x08, "tab": 0x09, "enter": 0x0D, "return": 0x0D, "shift": 0x10, "ctrl": 0x11,
    "control": 0x11, "alt": 0x12, "pause": 0x13, "capslock": 0x14, "esc": 0x1B, "escape": 0x1B,
    "space": 0x20, "pageup": 0x21, "pagedown": 0x22, "end": 0x23, "home": 0x24, "left": 0x25,
    "up": 0x26, "right": 0x27, "down": 0x28, "insert": 0x2D, "delete": 0x2E, "win": 0x5B,
    "windows": 0x5B, "menu": 0x5D, "printscreen": 0x2C,
    "volumemute": 0xAD, "volumedown": 0xAE, "volumeup": 0xAF,
    "nexttrack": 0xB0, "prevtrack": 0xB1, "stop": 0xB2, "playpause": 0xB3,
}
for _i in range(1, 13):
    VK[f"f{_i}"] = 0x6F + _i
for _c in "abcdefghijklmnopqrstuvwxyz":
    VK[_c] = ord(_c.upper())
for _d in "0123456789":
    VK[_d] = ord(_d)
VK.update({"-": 0xBD, "=": 0xBB, ",": 0xBC, ".": 0xBE, "/": 0xBF, ";": 0xBA, "[": 0xDB, "]": 0xDD,
           "\\": 0xDC, "'": 0xDE, "`": 0xC0, "+": 0xBB})

EXTENDED = {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E, 0x5B, 0x5D,
            0xAD, 0xAE, 0xAF, 0xB0, 0xB1, 0xB2, 0xB3}

# Never sent, whatever Claude asks for.
BLOCKED_KEYS = {"f4", "f5", "f6", "f7", "f8", "f9", "f10"}      # hands-free mouse hotkeys
BLOCKED_COMBOS = {frozenset(("alt", "f4"))}                # closes windows, may lose unsaved work


def _send(inputs):
    arr = (INPUT * len(inputs))(*inputs)
    sent = user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))
    if sent != len(inputs):
        raise OSError(f"SendInput sent {sent}/{len(inputs)} events (error {ctypes.get_last_error()})")


def _vk_event(vk, up):
    flags = (KEYEVENTF_EXTENDEDKEY if vk in EXTENDED else 0) | (KEYEVENTF_KEYUP if up else 0)
    return INPUT(INPUT_KEYBOARD, _INPUTUNION(ki=KEYBDINPUT(vk, 0, flags, 0, 0)))


def press_combo(names):
    """Press and release a combo such as ['ctrl', 'shift', 't']."""
    names = [n.strip().lower() for n in names if n.strip()]
    if not names:
        raise ValueError("no keys given")
    for n in names:
        if n not in VK:
            raise ValueError(f"unknown key '{n}'")
        if n in BLOCKED_KEYS:
            raise ValueError(f"key '{n}' is reserved for the hands-free mouse and is blocked")
    if frozenset(names) in BLOCKED_COMBOS:
        raise ValueError(f"combo '{'+'.join(names)}' is blocked (could discard unsaved work)")
    codes = [VK[n] for n in names]
    _send([_vk_event(c, False) for c in codes])
    time.sleep(0.03)
    _send([_vk_event(c, True) for c in reversed(codes)])


def type_unicode(text):
    """Type text into the focused window (any characters, no layout dependence)."""
    for ch in text:
        if ch == "\n":
            press_combo(["enter"])
            continue
        raw = ch.encode("utf-16-le")
        units = [int.from_bytes(raw[i:i + 2], "little") for i in range(0, len(raw), 2)]
        evs = []
        for u in units:
            evs.append(INPUT(INPUT_KEYBOARD, _INPUTUNION(ki=KEYBDINPUT(0, u, KEYEVENTF_UNICODE, 0, 0))))
            evs.append(INPUT(INPUT_KEYBOARD, _INPUTUNION(ki=KEYBDINPUT(0, u, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, 0))))
        _send(evs)
        time.sleep(0.004)


# --- windows -----------------------------------------------------------------------------

def _exe_of(hwnd):
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        return psutil.Process(pid).name().lower()
    except Exception:
        return ""


def _hosted_exe(frame_hwnd):
    """Store apps (Calculator, Photos, ...) show up as ApplicationFrameHost windows; the real app
    owns a child window. Return that child's process name."""
    found = []

    def cb(child, _):
        e = _exe_of(child)
        if e and e != "applicationframehost.exe":
            found.append(e)
            return False
        return True

    try:
        win32gui.EnumChildWindows(frame_hwnd, cb, None)
    except Exception:
        pass
    return found[0] if found else ""


def find_windows(exes, cls=None):
    """Visible, titled top-level windows of the given process name(s), topmost first.
    `cls` optionally restricts to one window class (e.g. 'CabinetWClass' = File Explorer windows,
    which keeps the desktop/taskbar, owned by the same explorer.exe, out of the list)."""
    names = {exes.lower()} if isinstance(exes, str) else {e.lower() for e in exes}
    out = []

    def cb(hwnd, _):
        if (win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd)
                and (cls is None or win32gui.GetClassName(hwnd) == cls)):
            exe = _exe_of(hwnd)
            if exe in names or (exe == "applicationframehost.exe" and _hosted_exe(hwnd) in names):
                out.append(hwnd)
        return True

    win32gui.EnumWindows(cb, None)
    return out


def foreground_exe():
    return _exe_of(win32gui.GetForegroundWindow())


def focus_window(hwnd):
    """Bring a window to the foreground without touching the mouse. Returns True on success."""
    if win32gui.IsIconic(hwnd):
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    if win32gui.GetForegroundWindow() == hwnd:
        return True
    fg = win32gui.GetForegroundWindow()
    cur = ctypes.windll.kernel32.GetCurrentThreadId()
    fg_thread = win32process.GetWindowThreadProcessId(fg)[0] if fg else 0
    attached = False
    try:
        if fg_thread and fg_thread != cur:
            attached = bool(user32.AttachThreadInput(cur, fg_thread, True))
        win32gui.BringWindowToTop(hwnd)
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass
    finally:
        if attached:
            user32.AttachThreadInput(cur, fg_thread, False)
    for _ in range(10):
        if win32gui.GetForegroundWindow() == hwnd:
            return True
        time.sleep(0.03)
    return False
