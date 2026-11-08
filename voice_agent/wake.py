"""Wake-word detection. Two backends behind one interface:

  sherpa       (default) open-vocabulary keyword spotting: no training, any phrase you can spell
               with the model's word pieces. Phrases live in wake_keywords.txt.
  openwakeword pretrained models: hey_jarvis, alexa, hey_mycroft, hey_rhasspy, or your own .onnx.

Both: `process(frame_int16) -> (fired, info)` and `reset()`.
"""

import logging
import tarfile
import urllib.request

import numpy as np

import config

log = logging.getLogger("voice_agent")

SHERPA_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/"
              "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2")
SHERPA_DIR = config.HERE / "models" / "sherpa_kws" / "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
KEYWORDS_FILE = config.HERE / "wake_keywords.txt"


def _ensure_sherpa_model():
    if (SHERPA_DIR / "tokens.txt").exists():
        return
    log.info("downloading keyword-spotting model (17 MB)...")
    dest = SHERPA_DIR.parent
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "kws.tar.bz2"
    urllib.request.urlretrieve(SHERPA_URL, archive)
    with tarfile.open(archive, "r:bz2") as t:
        t.extractall(dest, filter="data")       # "data" filter refuses paths outside dest
    archive.unlink()


class SherpaWake:
    def __init__(self):
        import sherpa_onnx
        _ensure_sherpa_model()
        # wake_keywords.txt lines: "<word pieces> @NAME". Sensitivity is appended here.
        lines = []
        for raw in KEYWORDS_FILE.read_text(encoding="utf-8").splitlines():
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            lines.append(f"{raw} :{config.SHERPA_SCORE} #{config.SHERPA_THRESHOLD}")
        config.LOG_DIR.mkdir(exist_ok=True)
        self._kw = config.LOG_DIR / "keywords_runtime.txt"
        self._kw.write_text("\n".join(lines) + "\n", encoding="utf-8")
        d = str(SHERPA_DIR)
        self.sp = sherpa_onnx.KeywordSpotter(
            tokens=f"{d}/tokens.txt",
            encoder=f"{d}/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
            decoder=f"{d}/decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
            joiner=f"{d}/joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
            keywords_file=str(self._kw), num_threads=1,
            keywords_score=config.SHERPA_SCORE, keywords_threshold=config.SHERPA_THRESHOLD)
        self.stream = self.sp.create_stream()
        self.name = "keywords from wake_keywords.txt"

    def process(self, frame):
        self.stream.accept_waveform(config.SAMPLE_RATE, frame.astype(np.float32) / 32768.0)
        hit = ""
        while self.sp.is_ready(self.stream):
            self.sp.decode_stream(self.stream)
            r = self.sp.get_result(self.stream)
            if r:
                hit = r
                self.sp.reset_stream(self.stream)
        return bool(hit), (f"heard {hit}" if hit else "")

    def reset(self):
        self.stream = self.sp.create_stream()


class OpenWakeWordWake:
    def __init__(self):
        from openwakeword.model import Model
        import openwakeword.utils as u
        try:
            self.m = Model(wakeword_models=[config.WAKE_WORD], inference_framework="onnx")
        except Exception as e:
            if "NO_SUCHFILE" not in str(e) and "doesn't exist" not in str(e):
                raise
            log.info("wake-word models missing; downloading...")
            u.download_models()
            self.m = Model(wakeword_models=[config.WAKE_WORD], inference_framework="onnx")
        self.name = config.WAKE_WORD

    def process(self, frame):
        s = max(self.m.predict(frame).values(), default=0)
        return s >= config.WAKE_THRESHOLD, f"score {s:.2f}"

    def reset(self):
        self.m.reset()


def load_wake():
    backend = config.WAKE_BACKEND.lower()
    if backend == "sherpa":
        return SherpaWake()
    if backend == "openwakeword":
        return OpenWakeWordWake()
    raise RuntimeError(f"WAKE_BACKEND must be 'sherpa' or 'openwakeword', not '{backend}'")
