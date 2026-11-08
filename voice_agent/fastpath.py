"""Instant path: common commands are matched locally and run without calling Claude.
Anything that doesn't match exactly falls through to Claude, so nothing is lost, it is just faster.

match(text) -> list of (tool_name, args) or None.
"""

import re

import tools

FILLER = re.compile(r"^(?:(?:hey|ok|okay|please|can you|could you|would you|i want you to|i want to|go ahead and)\s+)+")


def _norm(text):
    t = text.lower().replace("%", " percent")
    t = re.sub(r"[^a-z0-9. ]", " ", t)
    t = re.sub(r"\.(?!\d)|(?<!\d)\.", " ", t)           # drop sentence dots, keep decimals
    t = " ".join(t.split())
    t = FILLER.sub("", t)
    t = re.sub(r"\s+(please|now|for me)$", "", t)
    return t.strip()


NUM = r"(\d{1,3})(?: percent)?"
STEP = 10   # "louder", "brighter", "turn it down": change by this many percent

# (regex, builder(match) -> list of (tool, args)); full-match against the normalized text
RULES = [
    (r"unmute(?: the)?(?: sound| volume| audio| speakers?)?", lambda m: [("set_mute", {"mute": False})]),
    (r"mute(?: the)?(?: sound| volume| audio| speakers?)?", lambda m: [("set_mute", {"mute": True})]),
    (rf"(?:set |change |turn )?(?:the )?volume (?:to |at )?{NUM}", lambda m: [("set_volume", {"level": int(m[1])})]),
    (r"(?:turn )?(?:the )?volume up|louder|turn (?:it )?up(?: the volume)?", lambda m: [("set_volume", {"change": STEP})]),
    (r"(?:turn )?(?:the )?volume down|quieter|turn (?:it )?down(?: the volume)?", lambda m: [("set_volume", {"change": -STEP})]),
    (rf"(?:set |change |turn )?(?:the )?(?:screen )?brightness (?:to |at )?{NUM}", lambda m: [("set_brightness", {"level": int(m[1])})]),
    (r"(?:turn )?(?:the )?(?:screen )?brightness up|brighter|turn up (?:the )?brightness", lambda m: [("set_brightness", {"change": STEP})]),
    (r"(?:turn )?(?:the )?(?:screen )?brightness down|dimmer|turn down (?:the )?brightness", lambda m: [("set_brightness", {"change": -STEP})]),
    (r"(?:go to |switch to )?(?:the )?next tab", lambda m: [("browser_tab_action", {"action": "next"})]),
    (r"(?:go to |switch to )?(?:the )?(?:previous|last|prior) tab", lambda m: [("browser_tab_action", {"action": "prev"})]),
    (r"(?:open |make )?(?:a )?new tab", lambda m: [("browser_tab_action", {"action": "new"})]),
    (r"close (?:this |the |current )?(?:tab|page)", lambda m: [("browser_tab_action", {"action": "close"})]),
    (r"(?:reload|refresh)(?: the| this)?(?: page| tab)?", lambda m: [("browser_tab_action", {"action": "reload"})]),
    (r"go back|back", lambda m: [("browser_tab_action", {"action": "back"})]),
    (r"go forward|forward", lambda m: [("browser_tab_action", {"action": "forward"})]),
    (r"(?:take )?(?:a )?screen ?shot|capture (?:the )?screen", lambda m: [("screenshot", {})]),
    (r"minimi[sz]e (?:everything|all|all windows|all the windows|windows)|show (?:the )?desktop", lambda m: [("minimize_all", {})]),
    (r"lock(?: the)?(?: computer| screen| pc| laptop)?", lambda m: [("lock_screen", {})]),
    (r"play|pause|play pause|play or pause|pause music|play music", lambda m: [("media_keys", {"action": "play_pause"})]),
    (r"(?:next|skip)(?: song| track)?", lambda m: [("media_keys", {"action": "next"})]),
    (r"(?:previous|last) (?:song|track)", lambda m: [("media_keys", {"action": "previous"})]),
]
COMPILED = [(re.compile(p), f) for p, f in RULES]
APP_RULE = re.compile(r"(open|launch|start|close|quit)(?: the| my)? (.+?)(?: app| application)?")


def match(text):
    t = _norm(text)
    if not t:
        return None
    for rx, build in COMPILED:
        m = rx.fullmatch(t)
        if m:
            return build(m)
    m = APP_RULE.fullmatch(t)
    if m:
        verb, name = m[1], m[2]
        try:
            key, _ = tools._resolve_app(name)       # only allowlisted apps; anything else goes to Claude
        except ValueError:
            return None
        return [("launch_app" if verb in ("open", "launch", "start") else "close_app", {"name": key})]
    return None
