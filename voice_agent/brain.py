"""Claude tool-use loop: one spoken command in, tool calls out. No speech back."""

import logging
import os

import config
import tools

log = logging.getLogger("voice_agent")

SYSTEM = """You turn one spoken command into actions on a Windows PC, using tools. The user cannot use a keyboard or mouse. You never speak; you only act.
- The text is speech-to-text: expect small mistakes, and ignore a leading wake word ("Jarvis").
- Prefer open_url / search_web over typing into the browser. They open a NEW Chrome tab and start Chrome if needed. "Open Chrome and search X" = just search_web(X).
- Use launch_app for apps, browser_tab_action for tab commands, set_volume/set_mute for sound.
- Do exactly what was asked, in order, nothing extra. Never invent a command.
- If a command is unclear or impossible with your tools (e.g. shutdown, delete, close a window), call no tools and reply with the single word: unclear.
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

    def run(self, command):
        """Returns (success, summary). Executes tool calls as Claude requests them."""
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
                ok, msg = tools.run_tool(c.name, c.input)
                ran += 1
                log.info("tool %s(%s) -> %s %s", c.name, c.input, "OK" if ok else "FAIL", msg)
                if not ok:
                    failed = True
                results.append({"type": "tool_result", "tool_use_id": c.id, "content": msg, "is_error": not ok})
            messages.append({"role": "user", "content": results})
        return False, "too many steps"
