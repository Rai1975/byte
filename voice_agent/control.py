"""Integration seam: a tiny localhost HTTP endpoint so another program (e.g. the eye/jaw app)
can control the agent.

    POST http://127.0.0.1:8765/pause     -> same as "stop listening"
    POST http://127.0.0.1:8765/resume    -> wake up again
    POST http://127.0.0.1:8765/trigger   -> push-to-talk: start listening now, no wake word
    GET  http://127.0.0.1:8765/status    -> {"state": "listening" | "sleeping",
                                             "busy": true while handling a command,
                                             "phase": "" | "recording" | "transcribing" | "acting"}

Every POST needs the header  X-Voice-Agent: 1 . Browsers cannot send that header cross-site
without a CORS preflight (which we never allow), so a web page cannot poke the agent.
"""

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("voice_agent")


def start_server(port, on_command, get_status):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _host_ok(self):         # blocks DNS-rebinding
            return self.headers.get("Host", "").split(":")[0] in ("127.0.0.1", "localhost")

        def do_GET(self):
            if not self._host_ok():
                return self._reply(403, {"error": "bad host"})
            if self.path == "/status":
                return self._reply(200, get_status())
            self._reply(404, {"error": "not found"})

        def do_POST(self):
            if not self._host_ok() or self.headers.get("X-Voice-Agent") != "1":
                return self._reply(403, {"error": "missing X-Voice-Agent header"})
            cmd = self.path.strip("/")
            if cmd not in ("pause", "resume", "trigger"):
                return self._reply(404, {"error": "unknown command"})
            on_command(cmd)
            self._reply(200, {"ok": True, "command": cmd})

        def log_message(self, fmt, *args):
            log.debug("control: " + fmt, *args)

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, name="control", daemon=True).start()
    log.info("control endpoint on http://127.0.0.1:%d", port)
    return srv
