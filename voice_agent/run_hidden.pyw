"""Console-less supervisor: runs agent.py and restarts it if it crashes.
Exit code 0 (e.g. the spoken "exit voice control") stops the supervisor too."""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYW = HERE / ".venv" / "Scripts" / "pythonw.exe"

delay = 2
while True:
    started = time.time()
    code = subprocess.call([str(PYW), str(HERE / "agent.py")], cwd=str(HERE), creationflags=0x08000000)
    if code == 0:
        break
    delay = 2 if time.time() - started > 60 else min(delay * 2, 60)   # back off on rapid crashes
    time.sleep(delay)
