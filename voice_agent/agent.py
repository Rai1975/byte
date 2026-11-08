"""Voice agent: wake word -> record -> local Whisper -> Claude tool use -> actions. Never speaks.

    python agent.py                    run (what autostart runs)
    python agent.py --list-devices     show microphones
    python agent.py --test-wake        live wake-word scores + mic level, for tuning
    python agent.py --text "mute"      run the Claude+tools path without audio
    python agent.py --text "write this down um ..." [--paste]   test dictation cleanup (and paste)
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
import dictate

log = logging.getLogger("voice_agent")

SLEEP_PHRASES = ["stop listening", "sleep", "go to sleep", "sleep mode", "pause listening"]
EXIT_PHRASES = ["exit voice control", "quit voice control", "close voice control"]
CONFIRM_PHRASES = ["confirm", "confirmed", "yes confirm", "i confirm"]
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
        """Called the instant the wake word is heard. No sound is played first: the mic queue
        already holds whatever was said right after the wake word, and recording starts from it."""
        from audio import beep, record_command
        import win32gui
        t0 = time.time()
        target = win32gui.GetForegroundWindow()      # where dictated text will be pasted
        audio = record_command(mic, floor)
        t1 = time.time()
        if audio is None:
            log.info("no speech after wake")
            beep("fail")
            return
        text = transcribe(audio)
        t2 = time.time()
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
        elif dictate.split_trigger(text) is not None:
            try:
                ok = self.dictate(mic, floor, transcribe, brain, audio, text, target)
            except Exception:
                log.exception("dictation failure")
                ok = False
            beep("ok" if ok else "fail")
            log.info("timing: listen %.1fs | first speech-to-text %.1fs | dictation total %.1fs",
                     t1 - t0, t2 - t1, time.time() - t2)
        else:
            try:
                ok, summary = brain.run(text, lambda what: self.ask_confirm(mic, floor, transcribe, what))
            except Exception:
                log.exception("claude/tool failure")
                ok, summary = False, "error"
            log.info("result: %s (%s)", "success" if ok else "failure", summary)
            beep("ok" if ok else "fail")
            log.info("timing: listen %.1fs | speech-to-text %.1fs | claude+tools %.1fs",
                     t1 - t0, t2 - t1, time.time() - t2)

    def dictate(self, mic, floor, transcribe, brain, first_audio, first_text, target):
        """"write this down [text]": keep recording until a long pause, clean up with Claude, paste.
        Text said in the same breath as the trigger counts. The mic queue kept filling while the
        first part was transcribed, so nothing said in between is lost."""
        import numpy as np
        from audio import record_command
        inline = dictate.split_trigger(first_text)
        log.info("dictation started%s", f" (inline: {inline!r})" if inline else "")
        more = record_command(mic, floor, max_s=config.DICTATION_MAX_S, silence_s=config.DICTATION_SILENCE_S,
                              no_speech_s=config.DICTATION_SILENCE_S if inline else config.DICTATION_START_S)
        if more is None:
            raw = inline
        else:
            # Transcribe both parts together so sentences that span the first pause stay coherent.
            whole = transcribe(np.concatenate([first_audio, more]), hint=config.DICTATION_HINT)
            raw = dictate.split_trigger(whole)
            if raw is None:                       # trigger heard differently the second time
                raw = f"{inline} {transcribe(more, hint=config.DICTATION_HINT)}".strip()
        log.info("dictation heard: %r", raw)
        if not raw:
            log.info("nothing dictated")
            return False
        text = dictate.clean(brain.client, config.DICTATION_MODEL, raw)
        log.info("dictation cleaned: %r", text)
        if not text:
            return False
        log.info("dictation: %s", dictate.paste(text, target))
        return True

    def ask_confirm(self, mic, floor, transcribe, what):
        """Triple beep, then 5 s to say "confirm". Anything else (or silence) cancels."""
        from audio import beep, record_command
        log.info("asking for spoken confirmation: %s", what)
        beep("confirm")
        mic.drain()
        audio = record_command(mic, floor, max_s=5, no_speech_s=5)
        text = transcribe(audio) if audio is not None else ""
        ok = matches(text, CONFIRM_PHRASES)
        log.info("confirmation heard: %r -> %s", text, "confirmed" if ok else "cancelled")
        return ok

    # --- main loop ----------------------------------------------------------------------
    def run(self):
        import audio
        from brain import Brain
        from control import start_server
        from wake import load_wake

        brain = Brain(self.model)
        transcribe = audio.Transcriber()
        wake = load_wake()
        start_server(config.CONTROL_PORT, self.on_control, self.state)
        mic, floor = audio.Mic(), audio.NoiseFloor()
        last_wake = 0.0
        failures = 0
        log.info("ready. wake backend: %s (%s)", config.WAKE_BACKEND, wake.name)
        while not self.quit:
            try:
                if mic.stream is None:
                    mic.open()
                    floor.calibrate(mic)
                    wake.reset()
                frame = mic.read()
                failures = 0
                woke = False
                if self.trigger.is_set():
                    self.trigger.clear()
                    woke = True
                else:
                    floor.track(audio.rms(frame))
                    fired, _ = wake.process(frame)
                    if fired and time.time() - last_wake > config.WAKE_COOLDOWN_S:
                        woke = True
                if woke:
                    self.handle_utterance(mic, floor, transcribe, brain)
                    mic.drain()                    # drop audio from beeps / processing time
                    wake.reset()
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
    import audio
    from wake import load_wake
    wake = load_wake()
    mic, floor = audio.Mic(), audio.NoiseFloor()
    mic.open()
    floor.calibrate(mic)
    print(f"Backend: {config.WAKE_BACKEND} ({wake.name}). Say the wake word; hits are listed below. Ctrl+C to stop.")
    hits, last = 0, ""
    try:
        while True:
            f = mic.read()
            fired, info = wake.process(f)
            lvl = audio.rms(f)
            if fired:
                hits += 1
                last = f"{time.strftime('%H:%M:%S')} {info}"
                print(f"\n>>> WAKE #{hits}: {last}")
            bar = "#" * min(40, int(lvl / max(floor.threshold, 1) * 10))
            print(f"\rmic {lvl:6.0f} (speech>{floor.threshold:.0f}) {bar:<40} hits {hits}  {info if not fired else ''}   ", end="", flush=True)
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
    ap.add_argument("--yes", action="store_true", help="with --text: pre-confirm actions that need a spoken confirm (testing)")
    ap.add_argument("--paste", action="store_true", help='with --text "write this down ...": paste the result into the focused window after 3 s')
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
    if a.text and dictate.split_trigger(a.text) is not None:
        from brain import Brain
        raw = dictate.split_trigger(a.text)
        try:
            text = dictate.clean(Brain(model).client, a.model or config.DICTATION_MODEL, raw) if raw else ""
        except RuntimeError as e:
            sys.exit(f"error: {e}")
        print(text or "(nothing to write)")
        if a.paste and text:
            print("pasting into the focused window in 3 s...")
            time.sleep(3)
            print(dictate.paste(text))
            time.sleep(config.CLIPBOARD_RESTORE_S + 0.2)   # let the clipboard restore run
        return
    if a.text:
        from brain import Brain
        try:
            ok, summary = Brain(model).run(a.text, (lambda what: True) if a.yes else None)
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
