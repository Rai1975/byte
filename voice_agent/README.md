# Voice agent

Say the wake word, then a command. It runs the command and never talks back. Short beeps tell you what happened:

| Beep | Meaning |
|------|---------|
| one short high | wake word heard, speak now |
| rising two-note | command worked (also: resumed) |
| low falling | failed, not understood, or no speech heard |
| falling three-note | going to sleep / exiting |

Pipeline: openWakeWord → record until silence → faster-whisper (local) → Claude tool use → actions. Only the Claude call leaves your PC.

It is separate from the eye/jaw app: its own `.venv`, nothing imported from `handsfree/`, no mouse input, never presses F5–F10 (those are that app's hotkeys).

## Setup

```powershell
cd voice_agent
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -c "import openwakeword.utils as u; u.download_models()"   # wake-word models
```

`ANTHROPIC_API_KEY` is read from the environment, then `voice_agent/.env`, then the repo-root `.env`. Without it the agent says so and exits.

Run: `.\.venv\Scripts\python agent.py` (first run downloads the Whisper model).

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
| `WAKE_WORD` | `hey_jarvis` | `alexa`, `hey_mycroft`, `hey_rhasspy`, or path to your own `.onnx` |
| `WAKE_THRESHOLD` | 0.5 | |
| `WHISPER_MODEL` | `base.en` | `small.en` is more accurate, slower |
| `CLAUDE_MODEL` | `claude-haiku-5-5` | set `claude-sonnet-5-5` for harder commands |
| `SILENCE_S` | 1.0 | quiet time that ends a command |
| `CONTROL_PORT` | 8765 | |

## Tuning

- `python agent.py --test-wake`: live wake score, peak, and mic level. Say the wake word a few times. Raise `WAKE_THRESHOLD` if it fires on other speech; lower it if it misses you.
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
