"""Microphone, beeps, silence-based recording, local speech-to-text."""

import logging
import queue
import threading
import time
import winsound

import numpy as np
import sounddevice as sd

import config

log = logging.getLogger("voice_agent")


# --- beeps (short, never block for long) --------------------------------------------------
BEEPS = {
    "wake": [(1000, 70)],
    "ok": [(800, 70), (1200, 90)],
    "confirm": [(1000, 60), (1000, 60), (1000, 60)],
    "fail": [(300, 120), (200, 220)],
    "sleep": [(900, 80), (600, 80), (400, 120)],
    "resume": [(500, 70), (800, 70), (1100, 90)],
}


def beep(kind):
    if not config.BEEPS:
        return
    try:
        for freq, ms in BEEPS[kind]:
            winsound.Beep(freq, ms)
    except Exception:
        log.exception("beep failed")


# --- devices ------------------------------------------------------------------------------
def list_devices():
    default_in = sd.default.device[0]
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            api = sd.query_hostapis(d["hostapi"])["name"]
            mark = "*" if i == default_in else " "
            print(f"{mark} {i:3d}  {d['name']}  [{api}, {d['max_input_channels']} ch]")
    print("\n* = system default. Set MIC_DEVICE to an index or part of a name (env var or voice_agent/.env).")


def resolve_device(spec):
    spec = (spec or "").strip()
    if not spec:
        return None
    if spec.isdigit():
        return int(spec)
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and spec.lower() in d["name"].lower():
            return i
    raise RuntimeError(f"MIC_DEVICE '{spec}' not found. Run: python agent.py --list-devices")


class MicStalled(Exception):
    pass


class Mic:
    """16 kHz mono int16 stream delivered in 80 ms frames via a queue."""

    def __init__(self):
        self.q = queue.Queue(maxsize=400)
        self.stream = None

    def open(self):
        self.close()
        dev = resolve_device(config.MIC_DEVICE)
        self.stream = sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1, dtype="int16",
                                     blocksize=config.FRAME, device=dev, callback=self._cb)
        self.stream.start()
        info = sd.query_devices(dev if dev is not None else sd.default.device[0])
        log.info("mic open: %s", info["name"])

    def _cb(self, indata, frames, t, status):
        if status:
            log.debug("audio status: %s", status)
        try:
            self.q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            try:
                self.q.get_nowait()          # drop oldest
                self.q.put_nowait(indata[:, 0].copy())
            except queue.Empty:
                pass

    def read(self, timeout=3.0):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            raise MicStalled("no audio from the microphone")

    def qsize(self):
        return self.q.qsize()

    def skip(self, n):
        """Discard the next n frames (waits briefly for ones not captured yet)."""
        for _ in range(max(0, n)):
            try:
                self.q.get(timeout=0.2)
            except queue.Empty:
                return

    def drain(self):
        n = 0
        while True:
            try:
                self.q.get_nowait()
                n += 1
            except queue.Empty:
                return n

    def close(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None


def rms(frame):
    return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))


class NoiseFloor:
    """Room-noise estimate. Calibrated once at start, then tracked while idle."""

    def __init__(self):
        self.value = config.MIN_SPEECH_RMS / config.SPEECH_FACTOR

    def calibrate(self, mic, seconds=1.5):
        mic.drain()
        levels = []
        end = time.time() + seconds
        while time.time() < end:
            levels.append(rms(mic.read()))
        self.value = max(float(np.percentile(levels, 20)), 1.0)   # quiet part, so talking during start-up doesn't inflate it
        log.info("noise floor calibrated: %.0f (speech threshold %.0f)", self.value, self.threshold)

    def track(self, frame_rms):
        # Slowly follow quiet frames; ignore loud ones (speech) so talking never raises the floor.
        if frame_rms < self.value:                       # quieter than we thought: follow quickly
            self.value = 0.8 * self.value + 0.2 * max(frame_rms, 1.0)
        elif frame_rms < self.threshold:                 # room got a bit louder: follow slowly
            self.value = 0.98 * self.value + 0.02 * frame_rms

    @property
    def threshold(self):
        return max(self.value * config.SPEECH_FACTOR, config.MIN_SPEECH_RMS)


def record_command(mic, floor, max_s=None, no_speech_s=None, preroll=(), silence_s=None):
    """Record until silence_s (default SILENCE_S) of quiet after speech. Returns float32 audio or None if no speech."""
    max_s = max_s or config.MAX_RECORD_S
    no_speech_s = no_speech_s or config.NO_SPEECH_TIMEOUT_S
    silence_s = silence_s or config.SILENCE_S
    thr = floor.threshold
    frames, spoke, quiet = [], False, 0.0
    t0 = time.time()
    frame_s = config.FRAME / config.SAMPLE_RATE
    pending = list(preroll)               # audio captured between the wake word and now
    while time.time() - t0 < max_s:
        f = pending.pop(0) if pending else mic.read()
        frames.append(f)
        if rms(f) > thr:
            spoke, quiet = True, 0.0
        else:
            quiet += frame_s
            if spoke and quiet >= silence_s:
                break
            if not spoke and time.time() - t0 > no_speech_s:
                return None
    if not spoke:
        return None
    return np.concatenate(frames).astype(np.float32) / 32768.0


# --- speech to text -----------------------------------------------------------------------
def _add_nvidia_dlls():
    """pip's nvidia-* wheels keep their DLLs under site-packages/nvidia/*/bin; make Windows find them."""
    import os
    import site
    for sp in site.getsitepackages():
        root = os.path.join(sp, "nvidia")
        if os.path.isdir(root):
            for sub in os.listdir(root):
                b = os.path.join(root, sub, "bin")
                if os.path.isdir(b):
                    os.add_dll_directory(b)
                    os.environ["PATH"] = b + os.pathsep + os.environ.get("PATH", "")


class Transcriber:
    """faster-whisper on the GPU when possible (about 5x faster), else CPU int8."""

    def __init__(self):
        _add_nvidia_dlls()
        from faster_whisper import WhisperModel
        log.info("loading whisper model %s...", config.WHISPER_MODEL)
        try:
            self.model = WhisperModel(config.WHISPER_MODEL, device="cuda", compute_type="float16")
            self.device = "cuda"
            noise = (np.random.default_rng(0).standard_normal(16000) * 0.05).astype(np.float32)
            list(self.model.transcribe(noise, language="en", beam_size=1, vad_filter=False)[0])   # warm-up: CUDA init now, not on your first command
        except Exception as e:
            log.warning("GPU speech-to-text unavailable (%s); using CPU", str(e)[:120])
            self.model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
            self.device = "cpu"
        log.info("whisper ready on %s", self.device)

    def __call__(self, audio, hint=config.WHISPER_HINT):
        segs, _ = self.model.transcribe(audio, language="en", beam_size=config.WHISPER_BEAM, vad_filter=True,
                                        initial_prompt=hint, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segs).strip()
