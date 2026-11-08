"""Claude tool-use loop: one spoken command in, tool calls out. No speech back."""

import logging
import os

import config
import fastpath
import tools

log = logging.getLogger("voice_agent")

SYSTEM = """You turn one spoken command into actions on a Windows PC, using tools. The user cannot use a keyboard or mouse. You never speak; you only act.
- The text is speech-to-text: expect mistakes ("choose my brightness" = "turn my brightness"). If exactly one harmless command is a plausible reading, do it. Never guess at numbers or at closing apps; if unsure, answer: unclear. Ignore a leading wake word.
- Prefer open_url / search_web over typing into the browser. They open a NEW Chrome tab and start Chrome if needed. "Open Chrome and search X" = just search_web(X).
- Use launch_app / close_app for apps (call close_app straight away, no confirmation), browser_tab_action for tab commands, set_volume/set_mute for sound, set_brightness for the screen.
- Do exactly what was asked, in order, nothing extra. Make ALL needed tool calls together in your first reply, in order (there is no second turn). Never invent a command.
- If a command is unclear or impossible with your tools (e.g. shutdown, restart, delete files), call no tools and reply with the single word: unclear.
- After the tools succeed, reply with at most three words."""


class Brain:
    def __init__(self, model):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Set it as a user environment variable "
                               "(or put ANTHROPIC_API_KEY=... in voice_agent/.env), then restart.")
        import anthropic
        self.client = anthropic.Anthropic(api_key=key, max_retries=2, timeout=30.0)
        self.model = model
        self.specs = tools.tool_specs()

    def run(self, command, confirmer=None):
        """Returns (success, summary). Common commands run instantly via fastpath; the rest go to Claude."""
        fast = fastpath.match(command)
        if fast:
            ok_all = True
            for name, args in fast:
                ok, msg = tools.run_tool(name, args, confirmer)
                log.info("fast %s(%s) -> %s %s", name, args, "OK" if ok else "FAIL", msg)
                ok_all = ok_all and ok
            return ok_all, "fast path"
        messages = [{"role": "user", "content": command}]
        ran = 0
        failed = False
        for _ in range(config.MAX_TOOL_ROUNDS):
            resp = self.client.messages.create(model=self.model, max_tokens=config.MAX_TOKENS,
                                               system=SYSTEM, tools=self.specs, messages=messages)
            calls = [b for b in resp.content if b.type == "tool_use"]
            if not calls:
                text = " ".join(b.text for b in resp.content if b.type == "text").strip()
                log.info("claude: %s", text)
                if ran == 0:
                    return False, text or "no action"
                return not failed, text
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for c in calls:
                ok, msg = tools.run_tool(c.name, c.input, confirmer)
                ran += 1
                log.info("tool %s(%s) -> %s %s", c.name, c.input, "OK" if ok else "FAIL", msg)
                if not ok:
                    failed = True
                results.append({"type": "tool_result", "tool_use_id": c.id, "content": msg, "is_error": not ok})
            messages.append({"role": "user", "content": results})
            if not failed:
                return True, "done"      # nothing is spoken back, so skip the extra Claude call that would just say "Done"
            return False, "tool failed"
