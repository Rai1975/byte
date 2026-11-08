# Voice agent

Say the wake word, then a command. It runs the command and never talks back. Short beeps tell you what happened:

| Beep | Meaning |
|------|---------|
| one short high | wake word heard, speak now |
| rising two-note | command worked (also: resumed) |
| low falling | failed, not understood, or no speech heard |
| falling three-note | going to sleep / exiting |

Pipeline: wake word (sherpa-onnx keyword spotting, or openWakeWord) → record until silence → faster-whisper (local) → Claude tool use → actions. Only the Claude call leaves your PC.

It is separate from the eye/jaw app: its own `.venv`, nothing imported from `handsfree/`, no mouse input, never presses F5–F10 (those are that app's hotkeys).

## Setup

```powershell
cd voice_agent
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

`ANTHROPIC_API_KEY` is read from the environment, then `voice_agent/.env`, then the repo-root `.env`. Without it the agent says so and exits.

Run: `.\.venv\Scripts\python agent.py`. The first run downloads the Whisper model and the 17 MB keyword-spotting model.

Default wake word: **"byte"** (or "hey byte"). "Bite" sounds identical, so it wakes the agent too.

## Voice controls

- **"stop listening"** / **"sleep"**: pauses. While asleep, say the wake word then **"resume"** (also "wake up", "start listening").
- **"exit voice control"**: quits. It stays off until the next login or manual start.
- No shutdown/restart/delete tools exist, and `alt+f4` and F5–F10 are refused. Claude answers "unclear" for them.

## Autostart at login (no console window)

```powershell
powershell -ExecutionPolicy Bypass -File install_autostart.ps1          # install
powershell -ExecutionPolicy Bypass -File install_autostart.ps1 -Remove  # remove
```

Task Scheduler runs `run_hidden.pyw` (pythonw), which restarts the agent after a crash. The task has restart-on-failure too. Logs: `logs/voice_agent.log` (rotating).

## Config (env vars or `.env`; defaults in `config.py`)

| Setting | Default | Notes |
|---|---|---|
| `MIC_DEVICE` | system default | index or part of name; list with `--list-devices` |
| `WAKE_BACKEND` | `sherpa` | or `openwakeword` (then `WAKE_WORD` = `hey_jarvis`, `alexa`, `hey_mycroft`, `hey_rhasspy`, or a path to your own `.onnx`; `WAKE_THRESHOLD` 0.5) |
| `SHERPA_THRESHOLD` | 0.1 | lower = more sensitive (more false wakes) |
| `SHERPA_SCORE` | 2.0 | higher = more sensitive |
| `WHISPER_MODEL` | `base.en` | `small.en` is more accurate, slower |
| `CLAUDE_MODEL` | `claude-haiku-5-5` | set `claude-sonnet-5-5` for harder commands |
| `SILENCE_S` | 1.0 | quiet time that ends a command |
| `CONTROL_PORT` | 8765 | |

## Changing the wake word

Edit `wake_keywords.txt`: one line per way the model might "spell" the sound, as word pieces from `models/sherpa_kws/*/tokens.txt`, ending in `@NAME`. Example for "computer": `▁COM PU TER @COMPUTER`. Generate pieces with `sherpa_onnx.text2token(["COMPUTER"], tokens=..., tokens_type="bpe", bpe_model=...)` (needs `pip install pypinyin`), and add alternative splits if it misses. Longer, distinctive phrases are far more reliable than one short word. Or switch to `WAKE_BACKEND=openwakeword` for the built-in models.

## Tuning

- `python agent.py --test-wake`: mic level and a list of wake hits. Say the wake word 10 times, then talk normally. Too many false wakes → **raise** `SHERPA_THRESHOLD` (e.g. 0.2) or lower `SHERPA_SCORE`. Missing you → lower the threshold (0.05) or raise the score (3.0).
- Cuts you off early → raise `SILENCE_S`. Never stops recording in a noisy room → raise `SPEECH_FACTOR`.
- The noise floor is calibrated at start (stay quiet for 1.5 s) and tracked while idle.

## Testing without a mic

```powershell
python agent.py --text "go to youtube.com in a new tab"      # Claude + tools
python agent.py --call-tool set_volume level=30              # one tool, no Claude
```

## Control seam (for the eye/jaw app)

`POST http://127.0.0.1:8765/pause | /resume | /trigger`, `GET /status`. POSTs need the header `X-Voice-Agent: 1`. `/trigger` is push-to-talk: it starts recording immediately with no wake word, so a jaw clench can replace the wake word. Not wired into your app.

```python
import urllib.request
urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8765/trigger", method="POST", headers={"X-Voice-Agent": "1"}))
```

## Adding a tool

In `tools.py`, write a function that returns a short string (raise on error), then add it to `REGISTRY` with a description and JSON schema. Keep it to direct APIs; no mouse, no raw shell, nothing destructive.

## Files

`agent.py` main loop and CLI · `audio.py` mic, beeps, recording, Whisper · `brain.py` Claude loop and prompt · `tools.py` tools · `winapi.py` SendInput and window focus · `control.py` localhost endpoint · `run_hidden.pyw` + `install_autostart.ps1` autostart.
