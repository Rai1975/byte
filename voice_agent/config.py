"""Settings. Every value can be overridden with an environment variable of the same name,
or a line `NAME=value` in voice_agent/.env (read first) or the repo-root .env (read second).
Real environment variables win over both files."""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_env_files():
    for path in (HERE / ".env", HERE.parent / ".env"):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        except OSError:
            pass


_load_env_files()


def _get(name, default, cast=str):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return cast(raw)
    except ValueError:
        return default


# --- wake word -------------------------------------------------------------------------
# Built-in openWakeWord models: alexa, hey_jarvis, hey_mycroft, hey_rhasspy.
# You can also give the path to your own .onnx model. See README "Changing the wake word".
WAKE_WORD = _get("WAKE_WORD", "hey_jarvis")
WAKE_THRESHOLD = _get("WAKE_THRESHOLD", 0.5, float)   # 0..1. Lower = more sensitive
WAKE_COOLDOWN_S = 1.5                                 # ignore re-triggers right after a wake

# --- microphone ------------------------------------------------------------------------
# Blank = system default. Otherwise a device index or a part of its name
# (see `python agent.py --list-devices`).
MIC_DEVICE = _get("MIC_DEVICE", "")
SAMPLE_RATE = 16000
FRAME = 1280                                          # 80 ms, what openWakeWord expects

# --- recording / silence ---------------------------------------------------------------
SILENCE_S = _get("SILENCE_S", 1.0, float)             # end of command after this much quiet
MAX_RECORD_S = _get("MAX_RECORD_S", 15.0, float)
NO_SPEECH_TIMEOUT_S = _get("NO_SPEECH_TIMEOUT_S", 5.0, float)
SPEECH_FACTOR = _get("SPEECH_FACTOR", 2.5, float)     # speech = louder than noise floor * this
MIN_SPEECH_RMS = _get("MIN_SPEECH_RMS", 250, float)   # absolute floor for the threshold (int16 units)

# --- speech to text (local) ------------------------------------------------------------
WHISPER_MODEL = _get("WHISPER_MODEL", "base.en")      # tiny.en, base.en, small.en (slower, better)
WHISPER_HINT = "Chrome, YouTube, Google, Notepad, File Explorer, Settings, new tab, screenshot, volume."

# --- Claude ----------------------------------------------------------------------------
CLAUDE_MODEL = _get("CLAUDE_MODEL", "claude-haiku-5-5")
FAST_MODEL = "claude-haiku-5-5"
MAX_TOKENS = _get("MAX_TOKENS", 512, int)
MAX_TOOL_ROUNDS = 6

# --- control seam (localhost) ----------------------------------------------------------
CONTROL_PORT = _get("CONTROL_PORT", 8765, int)

# --- misc ------------------------------------------------------------------------------
LOG_DIR = HERE / "logs"
SCREENSHOT_DIR = Path.home() / "Pictures" / "voice-agent"
