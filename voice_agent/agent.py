"""Voice agent: wake word -> record -> local Whisper -> Claude tool use -> actions. Never speaks.

    python agent.py                    run (what autostart runs)
    python agent.py --list-devices     show microphones
    python agent.py --test-wake        live wake-word scores + mic level, for tuning
    python agent.py --text "mute"      run the Claude+tools path without audio
    python agent.py --call-tool NAME key=value ...   run one tool directly (no Claude), e.g. set_volume level=30
    python agent.py --fast             use the fast model (Haiku)
"""

import argparse
import difflib
import json
import logging
import logging.handlers
import re
import sys
import threading
import time

import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import config

log = logging.getLogger("voice_agent")

SLEEP_PHRASES = ["stop listening", "sleep", "go to sleep", "sleep mode", "pause listening"]
EXIT_PHRASES = ["exit voice control", "quit voice control", "close voice control"]
RESUME_PHRASES = ["resume", "resume listening", "wake up", "start listening", "resume voice control"]


def setup_logging(console):
    config.LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    log.setLevel(logging.INFO)
    fh = logging.handlers.RotatingFileHandler(config.LOG_DIR / "voice_agent.log", maxBytes=1_000_000,
                                              backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    if console and sys.stdout is not None:       # pythonw has no stdout
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        log.addHandler(sh)


def normalize(text):
    t = re.sub(r"[^a-z ]", " ", text.lower())
    t = re.sub(r"\b(hey|ok|okay|jarvis|alexa|mycroft|please)\b", " ", t)
    return " ".join(t.split())


def matches(text, phrases):
    t = normalize(text)
    return bool(t) and bool(difflib.get_close_matches(t, phrases, n=1, cutoff=0.85))


def load_wake_model():
    """Create the openWakeWord model, downloading the model files first if they are missing."""
    from openwakeword.model import Model
    import openwakeword.utils as u
    try:
        return Model(wakeword_models=[config.WAKE_WORD], inference_framework="onnx")
    except Exception as e:
        if "NO_SUCHFILE" not in str(e) and "doesn't exist" not in str(e):
            raise
        log.info("wake-word models missing; downloading...")
        u.download_models()
        return Model(wakeword_models=[config.WAKE_WORD], inference_framework="onnx")


class Agent:
    def __init__(self, model):
        self.model = model
        self.sleeping = False
        self.quit = False
        self.trigger = threading.Event()
        self.pending = []                         # commands from the control endpoint

    # --- control seam -------------------------------------------------------------------
    def on_control(self, cmd):
        log.info("control: %s", cmd)
        if cmd == "pause":
            self.sleeping = True
        elif cmd == "resume":
            self.sleeping = False
        elif cmd == "trigger":
            self.trigger.set()

    def state(self):
        return "sleeping" if self.sleeping else "listening"

    # --- one spoken interaction ---------------------------------------------------------
    def handle_utterance(self, mic, floor, transcribe, brain):
        from audio import beep, record_command
        beep("wake")
        mic.drain()
        audio = record_command(mic, floor)
        if audio is None:
            log.info("no speech after wake")
            beep("fail")
            return
        text = transcribe(audio)
        log.info("heard: %r%s", text, " [sleeping]" if self.sleeping else "")
        if not text:
            beep("fail")
            return
        if self.sleeping:
            if matches(text, RESUME_PHRASES):
                self.sleeping = False
                beep("resume")
            else:
                beep("fail")
            return
        if matches(text, EXIT_PHRASES):
            beep("sleep")
            self.quit = True
        elif matches(text, SLEEP_PHRASES):
            self.sleeping = True
            beep("sleep")
        elif matches(text, RESUME_PHRASES):
            beep("resume")
        else:
            try:
                ok, summary = brain.run(text)
            except Exception:
                log.exception("claude/tool failure")
                ok, summary = False, "error"
            log.info("result: %s (%s)", "success" if ok else "failure", summary)
            beep("ok" if ok else "fail")

    # --- main loop ----------------------------------------------------------------------
    def run(self):
        import audio
        from brain import Brain
        from control import start_server

        brain = Brain(self.model)
        transcribe = audio.Transcriber()
        oww = load_wake_model()
        start_server(config.CONTROL_PORT, self.on_control, self.state)
        mic, floor = audio.Mic(), audio.NoiseFloor()
        last_wake = 0.0
        failures = 0
        log.info("ready. wake word: %s (threshold %.2f)", config.WAKE_WORD, config.WAKE_THRESHOLD)
        while not self.quit:
            try:
                if mic.stream is None:
                    mic.open()
                    floor.calibrate(mic)
                    oww.reset()
                frame = mic.read()
                failures = 0
                woke = False
                if self.trigger.is_set():
                    self.trigger.clear()
                    woke = True
                else:
                    floor.track(audio.rms(frame))
                    scores = oww.predict(frame)
                    if max(scores.values(), default=0) >= config.WAKE_THRESHOLD and time.time() - last_wake > config.WAKE_COOLDOWN_S:
                        woke = True
                if woke:
                    self.handle_utterance(mic, floor, transcribe, brain)
                    mic.drain()                    # drop audio from beeps / processing time
                    oww.reset()
                    last_wake = time.time()
            except Exception:
                failures += 1
                log.exception("loop error (%d); reopening mic", failures)
                mic.close()
                time.sleep(min(30, 1.0 * failures))   # back off, then recover on its own
        mic.close()
        log.info("exited by voice command")


# --- diagnostics ------------------------------------------------------------------------
def test_wake():
    import numpy as np
    import audio
    from openwakeword.model import Model
    oww = load_wake_model()
    mic, floor = audio.Mic(), audio.NoiseFloor()
    mic.open()
    floor.calibrate(mic)
    print(f"Say the wake word ({config.WAKE_WORD}). Threshold {config.WAKE_THRESHOLD}. Ctrl+C to stop.")
    peak = 0.0
    try:
        while True:
            f = mic.read()
            s = max(oww.predict(f).values(), default=0)
            peak = max(peak * 0.98, s)
            lvl = audio.rms(f)
            hit = "  <-- WAKE" if s >= config.WAKE_THRESHOLD else ""
            print(f"\rscore {s:4.2f} {'#' * int(s * 30):<30} peak {peak:4.2f}  mic {lvl:6.0f} (speech>{floor.threshold:.0f}){hit}   ", end="", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        mic.close()


def acquire_single_instance():
    import win32api, win32event, winerror
    h = win32event.CreateMutex(None, False, "Local\\VoiceAgentSingleton")
    if win32api.GetLastError() == winerror.ERROR_ALREADY_EXISTS:
        return None
    return h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--test-wake", action="store_true")
    ap.add_argument("--text", metavar="COMMAND")
    ap.add_argument("--call-tool", nargs="+", metavar="NAME [key=value ...]")
    ap.add_argument("--fast", action="store_true", help=f"use {config.FAST_MODEL}")
    ap.add_argument("--model", help="override the Claude model")
    a = ap.parse_args()
    model = a.model or (config.FAST_MODEL if a.fast else config.CLAUDE_MODEL)
    interactive = bool(a.list_devices or a.test_wake or a.text or a.call_tool)
    setup_logging(console=interactive or sys.stdout is not None and sys.stdout.isatty())

    if a.list_devices:
        import audio
        return audio.list_devices()
    if a.test_wake:
        return test_wake()
    if a.call_tool:
        import tools
        args = {}
        for part in a.call_tool[1:]:                 # key=value (value is JSON if it parses), or a JSON object
            if part.lstrip().startswith("{"):
                args.update(json.loads(part))
            else:
                k, _, v = part.partition("=")
                try:
                    args[k] = json.loads(v)
                except ValueError:
                    args[k] = v
        print(tools.run_tool(a.call_tool[0], args))
        return
    if a.text:
        from brain import Brain
        try:
            ok, summary = Brain(model).run(a.text)
        except RuntimeError as e:
            sys.exit(f"error: {e}")
        print("success" if ok else "failure", "-", summary)
        return

    lock = acquire_single_instance()      # keep this reference alive: the mutex dies with the handle
    if lock is None:
        log.info("already running; exiting")
        return
    try:
        Agent(model).run()
    except Exception:
        log.exception("fatal")
        sys.exit(1)               # non-zero => the supervisor / Task Scheduler restarts us


if __name__ == "__main__":
    main()
