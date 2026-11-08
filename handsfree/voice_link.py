"""
Link to the voice agent (voice_agent/): start it, and know when it is busy with a command.

The voice agent runs in its own Python environment (voice_agent/.venv: Whisper on the GPU,
Claude, wake word), so it is a separate process. It answers GET http://127.0.0.1:8765/status
with {"state": "listening"|"sleeping", "busy": bool, "phase": ...}. This thread polls that
about 10 times a second. While `busy` is True (from the wake word until the command is done)
the app pauses the eye cursor AND jaw clicks, so nothing moves or clicks while it acts.

If the agent is already running (e.g. the autostart task), it is used as is; a second copy
would exit straight away anyway (single-instance mutex).
"""

import json
import os
import subprocess
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_DIR = os.path.join(ROOT, "voice_agent")
AGENT_PY = os.path.join(AGENT_DIR, ".venv", "Scripts", "python.exe")
PORT = int(os.environ.get("CONTROL_PORT", 8765))

POLL_S = 0.1
STALE_S = 1.0          # no answer for this long -> treat as not busy (a crashed agent never freezes you)
MAX_BUSY_S = 40.0      # busy longer than this -> assume it's stuck and stop pausing input


class VoiceLink(threading.Thread):
    def __init__(self, launch=True):
        super().__init__(daemon=True)
        self.launch = launch
        self.alive = True
        self.proc = None                  # only set if WE started the agent
        self.status = "off (--no-voice)" if not launch else "starting"
        self.state = ""                   # listening | sleeping
        self.phase = ""                   # recording | transcribing | acting while busy
        self._busy = False
        self._busy_since = None
        self._last_ok = 0.0
        self.commands = 0                 # commands handled since start
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never via a proxy

    # ------------------------------------------------------------------ public
    @property
    def busy(self):
        if not self._busy or time.time() - self._last_ok > STALE_S:
            return False
        return time.time() - self._busy_since < MAX_BUSY_S

    def label(self):
        if self.busy:
            return f"BUSY ({self.phase})"
        return self.state or self.status

    def close(self):
        self.alive = False
        p = self.proc
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()

    # ------------------------------------------------------------------ thread
    def run(self):
        if not self.launch:
            return
        if self._poll() is None:          # not running yet: start it
            self._start()
        while self.alive:
            st = self._poll()
            if st is None:
                self._busy = False
                if self.proc is not None and self.proc.poll() is not None:
                    self.status = f"exited (code {self.proc.returncode}) - see voice_agent/logs"
                elif self._last_ok:
                    self.status = "not responding"
            else:
                was = self._busy
                self._busy = bool(st.get("busy"))
                self.phase = st.get("phase", "")
                self.state = st.get("state", "")
                self.status = "running"
                if self._busy and not was:
                    self._busy_since = time.time()
                    self.commands += 1
            time.sleep(POLL_S)

    def _start(self):
        if not os.path.exists(AGENT_PY):
            self.status = "not installed (see voice_agent/README.md)"
            self.alive = False
            return
        try:
            self.proc = subprocess.Popen(
                [AGENT_PY, "agent.py"], cwd=AGENT_DIR,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=0x08000000)  # CREATE_NO_WINDOW; it logs to voice_agent/logs/
            self.status = "loading (Whisper + wake word)..."
        except OSError as e:
            self.status = f"could not start: {e}"
            self.alive = False

    def _poll(self):
        try:
            with self._opener.open(f"http://127.0.0.1:{PORT}/status", timeout=0.3) as r:
                st = json.loads(r.read())
            self._last_ok = time.time()
            return st
        except (OSError, ValueError):
            return None
