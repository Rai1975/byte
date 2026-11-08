"""Read instructions aloud (text-to-speech) so you don't have to read text while staring at a dot.

Uses the Windows built-in speech engine (System.Speech) through one long-lived hidden PowerShell
process, so there is nothing to install. If it can't start, say() silently does nothing.
"""

import base64
import subprocess
import sys

_PS = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.Rate = %RATE%
$s.Volume = 100
while (($line = [Console]::In.ReadLine()) -ne $null) {
    if ($line -eq 'STOP') { $s.SpeakAsyncCancelAll(); continue }
    $text = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($line))
    $s.SpeakAsyncCancelAll()
    [void]$s.SpeakAsync($text)
}
"""


class Speaker:
    def __init__(self, enabled=True, rate=0):
        """rate: -10 (slow) .. 10 (fast)."""
        self.proc = None
        self.last = None
        if not enabled or sys.platform != "win32":
            return
        try:
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self.proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", _PS.replace("%RATE%", str(int(rate)))],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=flags)
        except Exception:
            self.proc = None

    @property
    def available(self):
        return self.proc is not None and self.proc.poll() is None

    def _send(self, line):
        if not self.available:
            return
        try:
            self.proc.stdin.write((line + "\n").encode("ascii"))
            self.proc.stdin.flush()
        except Exception:
            self.proc = None

    def say(self, text):
        """Speak `text`, interrupting anything still being spoken."""
        self.last = text
        self._send(base64.b64encode(text.encode("utf-8")).decode("ascii"))

    def stop(self):
        self._send("STOP")

    def close(self):
        if self.proc is not None:
            try:
                self.proc.stdin.close()
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None
