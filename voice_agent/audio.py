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
    "fail": [(300, 120), (200, 220)],
    "sleep": [(900, 80), (600, 80), (400, 120)],
    "resume": [(500, 70), (800, 70), (1100, 90)],
}


def beep(kind):
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
        self.value = max(float(np.percentile(levels, 50)), 1.0)
        log.info("noise floor calibrated: %.0f (speech threshold %.0f)", self.value, self.threshold)

    def track(self, frame_rms):
        # Slowly follow quiet frames; ignore loud ones (speech) so talking never raises the floor.
        if frame_rms < self.threshold:
            self.value = 0.98 * self.value + 0.02 * max(frame_rms, 1.0)

    @property
    def threshold(self):
        return max(self.value * config.SPEECH_FACTOR, config.MIN_SPEECH_RMS)


def record_command(mic, floor, max_s=None, no_speech_s=None):
    """Record until SILENCE_S of quiet after speech. Returns float32 audio or None if no speech."""
    max_s = max_s or config.MAX_RECORD_S
    no_speech_s = no_speech_s or config.NO_SPEECH_TIMEOUT_S
    thr = floor.threshold
    frames, spoke, quiet = [], False, 0.0
    t0 = time.time()
    frame_s = config.FRAME / config.SAMPLE_RATE
    while time.time() - t0 < max_s:
        f = mic.read()
        frames.append(f)
        if rms(f) > thr:
            spoke, quiet = True, 0.0
        else:
            quiet += frame_s
            if spoke and quiet >= config.SILENCE_S:
                break
            if not spoke and time.time() - t0 > no_speech_s:
                return None
    if not spoke:
        return None
    return np.concatenate(frames).astype(np.float32) / 32768.0


# --- speech to text -----------------------------------------------------------------------
class Transcriber:
    def __init__(self):
        from faster_whisper import WhisperModel
        log.info("loading whisper model %s (first run downloads it)...", config.WHISPER_MODEL)
        self.model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")

    def __call__(self, audio):
        segs, _ = self.model.transcribe(audio, language="en", beam_size=1, vad_filter=True,
                                        initial_prompt=config.WHISPER_HINT, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segs).strip()
