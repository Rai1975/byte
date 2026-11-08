"""Hands-free mouse. Run from the repo root:

    python main.py                 # eye cursor + jaw-tap clicks + voice agent
    python main.py --no-clicks     # eye cursor only (no Arduino)
    python main.py --no-voice      # don't start the voice agent
    python main.py --port COM10    # pick the Arduino port instead of auto-detecting
    python main.py --camera 1      # another webcam
"""

from handsfree.app import main

if __name__ == "__main__":
    main()
