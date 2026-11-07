# Mouth Click

Assistive input system combining head-tracking cursor movement, speech-to-text, and bite-based clicking.

## Components

### `bite_click.py` — Bite-to-Click (MediaPipe)
Detects jaw clenching via webcam using MediaPipe Face Mesh. Tracks chin-to-nose distance — when you bite, that distance shrinks, triggering a mouse click.

- Gentle bite = left click
- Hard bite = right click

### `mouth_click.mjs` — Presage SmartSpectra Integration
Uses Presage SDK for blink detection and facial expressions. Use this for the broader pipeline (cursor tracking, speech-to-text).

## Quick Start — Bite Click

```bash
pip install opencv-python mediapipe pynput
python bite_click.py
```

1. Keep jaw relaxed during calibration (~2 seconds)
2. Bite gently to left-click, bite hard to right-click
3. Press `q` to quit

### Tuning

Watch the `Drop` percentage on screen while biting, then edit thresholds in `bite_click.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GENTLE_BITE_PERCENT` | `0.06` | % drop from baseline for left click |
| `HARD_BITE_PERCENT` | `0.12` | % drop from baseline for right click |
| `COOLDOWN_SECONDS` | `0.4` | Minimum time between clicks |

## Quick Start — Presage Integration

```bash
npm install
npm start
```

Requires a Presage API key — set `PRESAGE_API_KEY` env var or edit `mouth_click.mjs`.
