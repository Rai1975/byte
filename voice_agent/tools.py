"""Tools Claude can call. Each tool: a JSON schema + a function returning a short string.

To add a tool: write `def my_tool(...)`, add it to REGISTRY with a schema. That's all.
Rules: no mouse input, no raw shell, no destructive actions (shutdown, delete, close window).
"""

import ctypes
import datetime
import json
import os
import re
import subprocess
import time
import urllib.parse
import winreg
from pathlib import Path

import config
import winapi

# --- app allowlist -----------------------------------------------------------------------
# name -> (process image name, how to launch). Only these can be launched.
_PF = [os.environ.get("ProgramFiles", r"C:\Program Files"),
       os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
       os.path.join(os.environ.get("LOCALAPPDATA", ""), "")]


def _app_path(exe, *subdirs):
    """Registry App Paths first (HKLM then HKCU), then well-known install folders."""
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            with winreg.OpenKey(hive, rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{exe}") as k:
                p = winreg.QueryValueEx(k, "")[0].strip('"')
                if os.path.exists(p):
                    return p
        except OSError:
            pass
    for base in _PF:
        for sub in subdirs:
            p = os.path.join(base, sub)
            if os.path.exists(p):
                return p
    return None


def chrome_path():
    return _app_path("chrome.exe", r"Google\Chrome\Application\chrome.exe")


# exes: process names owning the app's windows. start: Start-menu name, so launching works for
# store apps and desktop apps alike without hunting for install paths. launch: explicit command.
APPS = {
    "chrome": {"exes": ("chrome.exe",), "launch": lambda: [chrome_path()]},
    "edge": {"exes": ("msedge.exe",), "launch": lambda: ["msedge.exe"]},
    "firefox": {"exes": ("firefox.exe",), "start": "Firefox"},
    "notepad": {"exes": ("notepad.exe",), "launch": lambda: ["notepad.exe"]},
    "file explorer": {"exes": ("explorer.exe",), "cls": "CabinetWClass", "launch": lambda: ["explorer.exe"], "always_new": True},
    "calculator": {"exes": ("calculatorapp.exe", "calculator.exe"), "start": "Calculator"},
    "paint": {"exes": ("mspaint.exe", "paintapp.exe"), "start": "Paint"},
    "snipping tool": {"exes": ("snippingtool.exe",), "start": "Snipping Tool"},
    "task manager": {"exes": ("taskmgr.exe",), "launch": lambda: ["taskmgr.exe"]},
    "settings": {"exes": ("systemsettings.exe",), "uri": "ms-settings:"},
    "outlook": {"exes": ("olk.exe",), "start": "Outlook"},                       # the new Outlook
    "outlook classic": {"exes": ("outlook.exe",), "start": "Outlook (classic)"},
    "word": {"exes": ("winword.exe",), "start": "Word"},
    "excel": {"exes": ("excel.exe",), "start": "Excel"},
    "powerpoint": {"exes": ("powerpnt.exe",), "start": "PowerPoint"},
    "onenote": {"exes": ("onenote.exe",), "start": "OneNote"},
    "teams": {"exes": ("ms-teams.exe", "teams.exe"), "start": "Microsoft Teams"},
    "zoom": {"exes": ("zoom.exe",), "start": "Zoom Workplace"},
    "spotify": {"exes": ("spotify.exe",), "start": "Spotify"},
    "discord": {"exes": ("discord.exe",), "start": "Discord"},
    "whatsapp": {"exes": ("whatsapp.exe", "whatsapp.root.exe"), "start": "WhatsApp"},
    "obsidian": {"exes": ("obsidian.exe",), "start": "Obsidian"},
}
ALIASES = {"google chrome": "chrome", "browser": "chrome", "explorer": "file explorer",
           "files": "file explorer", "windows explorer": "file explorer", "microsoft edge": "edge",
           "calc": "calculator", "windows settings": "settings", "task mgr": "task manager",
           "new outlook": "outlook", "microsoft outlook": "outlook", "mail": "outlook", "email": "outlook",
           "classic outlook": "outlook classic", "outlook (classic)": "outlook classic",
           "microsoft word": "word", "ms word": "word", "microsoft excel": "excel", "ms excel": "excel",
           "power point": "powerpoint", "microsoft powerpoint": "powerpoint", "ms powerpoint": "powerpoint",
           "microsoft onenote": "onenote", "microsoft teams": "teams", "ms teams": "teams",
           "zoom workplace": "zoom", "snipping": "snipping tool", "snip": "snipping tool"}


def _resolve_app(name):
    n = (name or "").strip().lower()
    n = ALIASES.get(n, n)
    if n not in APPS:
        raise ValueError(f"app '{name}' is not allowed. Allowed: {', '.join(sorted(APPS))}")
    return n, APPS[n]


def _spawn(argv):
    if not argv or not argv[0]:
        raise RuntimeError("application not found on this PC")
    subprocess.Popen(argv, close_fds=True, creationflags=0x00000008 | 0x00000200)  # DETACHED | NEW_GROUP


def _start_menu_launch(display_name):
    """Launch by Start-menu name (store apps, Office, Discord, ...) via the AppsFolder shell namespace."""
    import win32com.client
    items = win32com.client.Dispatch("Shell.Application").NameSpace("shell:AppsFolder").Items()
    for i in range(items.Count):
        it = items.Item(i)
        if it.Name.lower() == display_name.lower():
            if os.path.exists(it.Path):
                return _spawn([it.Path])
            return os.startfile("shell:AppsFolder\\" + it.Path)
    raise RuntimeError(f"'{display_name}' is not installed (not in the Start menu)")


def _windows(app):
    return winapi.find_windows(app["exes"], app.get("cls"))


def _focus_app(key):
    wins = _windows(APPS[key])
    return bool(wins) and winapi.focus_window(wins[0])


# --- tools --------------------------------------------------------------------------------

def launch_app(name):
    key, app = _resolve_app(name)
    if not app.get("always_new") and _windows(app):
        if _focus_app(key):
            return f"{key} was already open; focused it"
        raise RuntimeError(f"{key} is open but Windows would not let me focus it")
    if "uri" in app:
        os.startfile(app["uri"])
    elif "launch" in app:
        _spawn(app["launch"]())
    else:
        _start_menu_launch(app["start"])
    return f"launched {key}"


def focus_app(name):
    key, _ = _resolve_app(name)
    return f"focused {key}" if _focus_app(key) else f"{key} has no open window"


def close_app(name):
    """Politely close every window of an allowed app (WM_CLOSE, like clicking the X: the app itself
    asks to save unsaved work). Never kills processes."""
    import win32con, win32gui
    key, app = _resolve_app(name)
    wins = _windows(app)
    if not wins:
        return f"{key} is not open"
    for h in wins:
        win32gui.PostMessage(h, win32con.WM_CLOSE, 0, 0)
    time.sleep(1.5)
    left = [h for h in wins if win32gui.IsWindow(h) and win32gui.IsWindowVisible(h)]
    if left:
        return f"asked {key} to close; {len(left)} window(s) still open (probably asking whether to save)"
    return f"closed {key} ({len(wins)} window(s))"


def _normalize_url(url):
    url = (url or "").strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", url):
        url = "https://" + url
    p = urllib.parse.urlparse(url)
    if p.scheme not in ("http", "https"):
        raise ValueError("only http and https links are allowed")
    host = p.hostname or ""
    if "." not in host and host != "localhost":
        raise ValueError(f"'{url}' does not look like a web address")
    return url


def _chrome_open(url):
    exe = chrome_path()
    if not exe:
        raise RuntimeError("Chrome is not installed (checked registry App Paths and Program Files)")
    _spawn([exe, "--new-tab", url])      # Chrome reuses its running instance and opens a tab


def open_url(url):
    url = _normalize_url(url)
    _chrome_open(url)
    return f"opened {url} in a new Chrome tab"


def search_web(query):
    query = (query or "").strip()
    if not query:
        raise ValueError("empty search")
    url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(query)
    _chrome_open(url)
    return f"searched Google for '{query}' in a new Chrome tab"


TAB_KEYS = {"new": ["ctrl", "t"], "close": ["ctrl", "w"], "next": ["ctrl", "tab"],
            "prev": ["ctrl", "shift", "tab"], "reload": ["ctrl", "r"],       # ctrl+r, NOT F5
            "back": ["alt", "left"], "forward": ["alt", "right"]}


def _ensure_focus(app):
    if not app:
        return
    key, _ = _resolve_app(app)
    if winapi.foreground_exe() not in APPS[key]["exes"] and not _focus_app(key):
        raise RuntimeError(f"{key} has no window to focus")


def browser_tab_action(action):
    if action not in TAB_KEYS:
        raise ValueError(f"action must be one of {sorted(TAB_KEYS)}")
    _ensure_focus("chrome")
    time.sleep(0.1)
    winapi.press_combo(TAB_KEYS[action])
    return f"chrome tab action '{action}' sent"


def type_text(text, focus=None):
    if not text:
        raise ValueError("no text")
    _ensure_focus(focus)
    time.sleep(0.1)
    winapi.type_unicode(text)
    return f"typed {len(text)} characters"


def press_keys(keys, focus=None):
    """keys like 'ctrl+shift+t' or 'enter'."""
    names = [k for k in re.split(r"\s*\+\s*", keys.strip()) if k] if keys.strip() != "+" else ["+"]
    _ensure_focus(focus)
    time.sleep(0.1)
    winapi.press_combo(names)
    return f"pressed {'+'.join(names)}"


def _volume():
    from pycaw.pycaw import AudioUtilities
    return AudioUtilities.GetSpeakers().EndpointVolume


def set_volume(level=None, change=None):
    vol = _volume()
    if level is None and change is None:
        raise ValueError("give level (0-100) or change (+/- percent)")
    cur = vol.GetMasterVolumeLevelScalar() * 100
    target = level if level is not None else cur + change
    target = max(0, min(100, float(target)))
    vol.SetMasterVolumeLevelScalar(target / 100, None)
    if target > 0:
        vol.SetMute(0, None)
    return f"volume set to {round(vol.GetMasterVolumeLevelScalar() * 100)}%"


def set_mute(mute=True):
    vol = _volume()
    vol.SetMute(1 if mute else 0, None)
    return "muted" if mute else "unmuted"


def _wmi():
    import win32com.client
    return win32com.client.GetObject(r"winmgmts:root\wmi")


def set_brightness(level=None, change=None):
    """Built-in laptop display only (WMI). External monitors need DDC/CI and are not supported.
    Never goes below 10%: a near-black screen is unusable for someone who cannot use a keyboard."""
    if level is None and change is None:
        raise ValueError("give level (0-100) or change (+/- percent)")
    w = _wmi()
    cur = list(w.ExecQuery("SELECT * FROM WmiMonitorBrightness"))
    methods = list(w.ExecQuery("SELECT * FROM WmiMonitorBrightnessMethods"))
    if not cur or not methods:
        raise RuntimeError("no built-in display brightness control found (external monitors are not supported)")
    now = int(cur[0].CurrentBrightness)
    target = int(round(max(10.0, min(100.0, float(level if level is not None else now + change)))))
    for m in methods:        # ExecMethod with typed in-params: a direct m.WmiSetBrightness(...) call is rejected
        ip = m.Methods_("WmiSetBrightness").InParameters.SpawnInstance_()
        ip.Timeout = 1
        ip.Brightness = target
        w.ExecMethod(m.Path_.Path, "WmiSetBrightness", ip)
    time.sleep(0.3)
    after = int(list(w.ExecQuery("SELECT * FROM WmiMonitorBrightness"))[0].CurrentBrightness)
    return f"brightness {now}% -> {after}%"


def lock_screen():
    if not ctypes.windll.user32.LockWorkStation():
        raise OSError("LockWorkStation failed")
    return "locked"


def screenshot():
    from PIL import ImageGrab
    config.SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.SCREENSHOT_DIR / datetime.datetime.now().strftime("screenshot-%Y%m%d-%H%M%S.png")
    ImageGrab.grab(all_screens=True).save(path)
    return f"saved {path}"


def minimize_all():
    import win32com.client
    win32com.client.Dispatch("Shell.Application").MinimizeAll()
    return "minimized all windows"


SETTINGS_PAGES = {   # friendly name -> ms-settings URI
    "home": "ms-settings:", "display": "ms-settings:display", "sound": "ms-settings:sound",
    "bluetooth": "ms-settings:bluetooth", "wifi": "ms-settings:network-wifi",
    "network": "ms-settings:network", "apps": "ms-settings:appsfeatures",
    "default apps": "ms-settings:defaultapps", "update": "ms-settings:windowsupdate",
    "privacy": "ms-settings:privacy", "accessibility": "ms-settings:easeofaccess",
    "personalization": "ms-settings:personalization", "battery": "ms-settings:batterysaver",
    "power": "ms-settings:powersleep", "storage": "ms-settings:storagesense",
    "notifications": "ms-settings:notifications", "keyboard": "ms-settings:easeofaccess-keyboard",
    "mouse": "ms-settings:mousetouchpad", "microphone": "ms-settings:privacy-microphone",
    "camera": "ms-settings:privacy-webcam", "time": "ms-settings:dateandtime",
    "language": "ms-settings:regionlanguage", "about": "ms-settings:about",
}


def open_settings_page(page):
    p = (page or "home").strip().lower()
    uri = SETTINGS_PAGES.get(p) or (p if re.fullmatch(r"ms-settings:[a-z0-9\-]*", p) else None)
    if not uri:
        raise ValueError(f"unknown settings page '{page}'. Known: {', '.join(sorted(SETTINGS_PAGES))}")
    os.startfile(uri)
    return f"opened settings: {uri}"


MEDIA = {"play_pause": "playpause", "next": "nexttrack", "previous": "prevtrack", "stop": "stop"}


def media_keys(action):
    if action not in MEDIA:
        raise ValueError(f"action must be one of {sorted(MEDIA)}")
    winapi.press_combo([MEDIA[action]])
    return f"media {action}"


def wait(seconds):
    s = max(0.0, min(5.0, float(seconds)))
    time.sleep(s)
    return f"waited {s:g}s"


# --- registry (what Claude sees) ----------------------------------------------------------
def _schema(desc, props=None, required=()):
    return {"description": desc,
            "input_schema": {"type": "object", "properties": props or {}, "required": list(required)}}


_APP_ENUM = {"type": "string", "description": "One of: " + ", ".join(sorted(APPS))}
_FOCUS = {"type": "string", "description": "Optional app to focus before sending input, e.g. 'notepad'"}

REGISTRY = {
    "close_app": (close_app, _schema("Close all windows of an allowed app. Runs immediately, no confirmation.", {"name": _APP_ENUM}, ["name"])),
    "launch_app": (launch_app, _schema("Open an allowed app (focuses it if already open).", {"name": _APP_ENUM}, ["name"])),
    "focus_app": (focus_app, _schema("Bring an open app window to the front.", {"name": _APP_ENUM}, ["name"])),
    "open_url": (open_url, _schema("Open a web address in a NEW Chrome tab.", {"url": {"type": "string"}}, ["url"])),
    "search_web": (search_web, _schema("Google search in a NEW Chrome tab. Also opens Chrome if closed.", {"query": {"type": "string"}}, ["query"])),
    "browser_tab_action": (browser_tab_action, _schema("Chrome tab/navigation action.", {"action": {"type": "string", "enum": sorted(TAB_KEYS)}}, ["action"])),
    "type_text": (type_text, _schema("Type text into the focused (or given) app.", {"text": {"type": "string"}, "focus": _FOCUS}, ["text"])),
    "press_keys": (press_keys, _schema("Press a key or combo, e.g. 'ctrl+l', 'enter', 'win+d'. F5-F10 and alt+f4 are refused.", {"keys": {"type": "string"}, "focus": _FOCUS}, ["keys"])),
    "set_volume": (set_volume, _schema("Set system volume to a level (0-100), or change it relatively.", {"level": {"type": "number", "minimum": 0, "maximum": 100}, "change": {"type": "number", "description": "percent to add (negative to lower)"}})),
    "set_brightness": (set_brightness, _schema("Set screen brightness (0-100), or change it relatively.", {"level": {"type": "number", "minimum": 0, "maximum": 100}, "change": {"type": "number", "description": "percent to add (negative to lower)"}})),
    "set_mute": (set_mute, _schema("Mute or unmute system audio.", {"mute": {"type": "boolean"}}, ["mute"])),
    "lock_screen": (lock_screen, _schema("Lock the computer.")),
    "screenshot": (screenshot, _schema("Save a screenshot of all screens to the Pictures/voice-agent folder.")),
    "minimize_all": (minimize_all, _schema("Minimize all windows (show desktop).")),
    "open_settings_page": (open_settings_page, _schema("Open a Windows Settings page.", {"page": {"type": "string", "description": "One of: " + ", ".join(sorted(SETTINGS_PAGES))}}, ["page"])),
    "media_keys": (media_keys, _schema("Media control.", {"action": {"type": "string", "enum": sorted(MEDIA)}}, ["action"])),
    "wait": (wait, _schema("Pause up to 5 seconds between steps (e.g. let an app open).", {"seconds": {"type": "number"}}, ["seconds"])),
}


def tool_specs():
    return [{"name": n, **spec} for n, (_, spec) in REGISTRY.items()]


CONFIRM_REQUIRED = set()     # tools listed here need a spoken "confirm" within 5 s (none right now)


def run_tool(name, args, confirmer=None):
    """Run one tool. Never raises: returns (ok, message).
    `confirmer(description) -> bool` asks the user to say "confirm" (see agent.py)."""
    if name not in REGISTRY:
        return False, f"unknown tool '{name}'"
    try:
        if name in CONFIRM_REQUIRED:
            if confirmer is None:
                return False, "this action needs spoken confirmation, which is unavailable here"
            if not confirmer(f"{name} {args}"):
                return False, "cancelled: the user did not confirm. Do not retry."
        return True, REGISTRY[name][0](**(args or {}))
    except Exception as e:                                   # tool errors must never kill the loop
        return False, f"{type(e).__name__}: {e}"
