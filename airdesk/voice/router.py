import re
import subprocess
import urllib.parse
import webbrowser


def _normalize(text):
    """lowercase, strip punctuation whisper adds, collapse whitespace.
    '+' becomes a space so 'Ctrl+S' matches like 'control s'."""
    text = "".join(c if c.isalnum() or c == " " else
                   (" " if c == "+" else "")
                   for c in text.lower())
    return " ".join(text.split())


_APPS = {
    "notepad": "notepad", "calculator": "calc", "paint": "mspaint",
    "explorer": "explorer", "file explorer": "explorer", "files": "explorer",
    "chrome": "chrome", "edge": "msedge", "browser": "msedge",
    "terminal": "wt", "settings": "ms-settings:", "task manager": "taskmgr",
    "word": "winword", "excel": "excel", "spotify": "spotify",
    "vs code": "code", "code": "code", "obsidian": "obsidian",
}

_MODS = {"control": "ctrl", "ctrl": "ctrl", "alt": "alt", "shift": "shift",
         "windows": "win", "win": "win"}

_COMBO_RE = re.compile(
    r"^(?:press |hit |do )?((?:control|ctrl|alt|shift|windows|win)\s+)+"
    r"([a-z0-9]|f\d{1,2}|enter|tab|escape|space|delete|backspace|home|end"
    r"|left|right|up|down|plus|minus)$")


class Router:
    """Routes final transcripts to typing (dictation) or actions (commands)."""

    def __init__(self, cfg, kb, mouse, engine):
        self.kb = kb
        self.mouse = mouse
        self.engine = engine
        self.mode = cfg.get("mode", "dictation")
        self.smart_spacing = cfg.get("typing", {}).get("smart_spacing", True)
        self._last_typed = 0
        self._last_char = "\n"
        self._status_sent = False
        self._commands = self._build_commands()

    # ------------------------------------------------------------ public

    def handle(self, text):
        norm = _normalize(text)
        if not norm:
            return
        if self.mode == "command":
            if norm.startswith("type "):
                self._type(text.split(" ", 1)[1] if " " in text else "")
            elif not self._try_command(norm):
                self._status(f"? {text}")
        else:  # dictation: short utterances may be commands, rest is typed
            if len(norm.split()) <= 4 and self._try_command(norm, inline=True):
                return
            self._type(text)

    # ----------------------------------------------------------- typing

    def _type(self, text):
        text = text.strip()
        if not text:
            return
        if (self.smart_spacing and self._last_char not in " \n\t([{'\"-"
                and text[0] not in ".,!?;:)]}'\""):
            text = " " + text
        self.kb.type_text(text)
        self._last_typed = len(text)
        self._last_char = text[-1]
        self._status(f"typed: {text.strip()[:48]}")

    def _delete_last(self):
        if self._last_typed:
            self.kb.backspace(self._last_typed)
            self._status(f"deleted {self._last_typed} chars")
            self._last_typed = 0
            self._last_char = "\n"

    def _newline(self, n=1):
        self.kb.type_text("\n" * n)
        self._last_typed = n
        self._last_char = "\n"

    # ---------------------------------------------------------- commands

    def _try_command(self, norm, inline=False):
        for pattern, fn, inline_ok in self._commands:
            if inline and not inline_ok:
                continue
            m = pattern.fullmatch(norm)
            if m:
                self._status_sent = False
                fn(m)
                if not self._status_sent:  # echo plain kb/mouse commands
                    self._status(norm)
                return True
        m = _COMBO_RE.fullmatch(norm)
        if m:
            mods = [_MODS[w] for w in norm.split()[:-1] if w in _MODS]
            # dedupe, keep order
            mods = list(dict.fromkeys(mods))
            self.kb.tap(*mods, m.group(2))
            self._status(f"pressed {'+'.join(mods)}+{m.group(2)}")
            return True
        return False

    def _build_commands(self):
        kb, mouse = self.kb, self.mouse
        table = [
            # dictation editing
            (r"(?:insert )?new line|next line", lambda m: self._newline()),
            (r"(?:insert )?new paragraph", lambda m: self._newline(2)),
            (r"delete that|scratch that", lambda m: self._delete_last()),
            (r"undo(?: that)?", lambda m: kb.tap("ctrl", "z")),
            (r"redo(?: that)?", lambda m: kb.tap("ctrl", "y")),
            # bare keys
            (r"(?:press )?enter", lambda m: kb.tap("enter")),
            (r"(?:press )?backspace", lambda m: kb.tap("backspace")),
            (r"(?:press )?(?:escape|cancel)", lambda m: kb.tap("escape")),
            (r"(?:press )?tab(?: key)?", lambda m: kb.tap("tab")),
            (r"(?:press )?space(?: bar)?", lambda m: kb.tap("space")),
            (r"(?:press )?delete(?: key)?", lambda m: kb.tap("delete")),
            # editing
            (r"select all", lambda m: kb.tap("ctrl", "a")),
            (r"copy(?: that| this)?", lambda m: kb.tap("ctrl", "c")),
            (r"paste(?: that| this| it)?", lambda m: kb.tap("ctrl", "v")),
            (r"cut(?: that| this)?", lambda m: kb.tap("ctrl", "x")),
            (r"save(?: file| that)?", lambda m: kb.tap("ctrl", "s")),
            (r"find|search this page", lambda m: kb.tap("ctrl", "f")),
            # windows & desktop
            (r"close (?:window|this window|it)", lambda m: kb.tap("alt", "f4")),
            (r"close tab", lambda m: kb.tap("ctrl", "w")),
            (r"new tab", lambda m: kb.tap("ctrl", "t")),
            (r"reopen tab", lambda m: kb.tap("ctrl", "shift", "t")),
            (r"next tab", lambda m: kb.tap("ctrl", "tab")),
            (r"previous tab|last tab", lambda m: kb.tap("ctrl", "shift", "tab")),
            (r"switch window[s]?", lambda m: kb.alt_tab()),
            (r"(?:show )?task view", lambda m: kb.win_tab()),
            (r"show desktop", lambda m: kb.win_d()),
            (r"minimize(?: window)?", lambda m: kb.tap("win", "down")),
            (r"maximize(?: window)?", lambda m: kb.tap("win", "up")),
            (r"snap left", lambda m: kb.tap("win", "left")),
            (r"snap right", lambda m: kb.tap("win", "right")),
            # navigation
            (r"go back", lambda m: kb.tap("alt", "left")),
            (r"go forward", lambda m: kb.tap("alt", "right")),
            (r"refresh|reload(?: page)?", lambda m: kb.tap("f5")),
            (r"scroll up", lambda m: mouse.wheel(360)),
            (r"scroll down", lambda m: mouse.wheel(-360)),
            (r"page up", lambda m: kb.tap("pageup")),
            (r"page down", lambda m: kb.tap("pagedown")),
            (r"go to top", lambda m: kb.tap("ctrl", "home")),
            (r"go to bottom", lambda m: kb.tap("ctrl", "end")),
            (r"zoom in", lambda m: kb.tap("ctrl", "plus")),
            (r"zoom out", lambda m: kb.tap("ctrl", "minus")),
            (r"(?:reset|normal) zoom", lambda m: kb.tap("ctrl", "0")),
            # mouse
            (r"(?:left )?click", lambda m: mouse.click()),
            (r"double click", lambda m: mouse.double_click()),
            (r"right click", lambda m: mouse.right_click()),
            (r"middle click", lambda m: mouse.middle_click()),
            # media & volume
            (r"volume up", lambda m: [kb.volume(True) for _ in range(3)]),
            (r"volume down", lambda m: [kb.volume(False) for _ in range(3)]),
            (r"mute|unmute", lambda m: kb.tap("mute")),
            (r"(?:play|pause)(?: music| media)?", lambda m: kb.tap("play_pause")),
            (r"next (?:track|song)", lambda m: kb.tap("next_track")),
            (r"previous (?:track|song)|last song", lambda m: kb.tap("prev_track")),
            # modes
            (r"dictation mode|start dictation", lambda m: self._set_mode("dictation")),
            (r"command mode|start commands?", lambda m: self._set_mode("command")),
            (r"stop listening|go to sleep|mute mic(?:rophone)?",
             lambda m: self._mic_off()),
            # known apps only, so dictating "open the door" never launches
            # anything ("open <whatever>" works in command mode below)
            (r"(?:open|launch|start) (%s)" % "|".join(
                sorted(map(re.escape, _APPS), key=len, reverse=True)),
             lambda m: self._open_app(m.group(1))),
        ]
        # command-mode only: too easy to trigger while dictating a sentence
        # like "open the door" or "search for the file"
        cmd_only = [
            (r"(?:open|launch|start) (.+)", lambda m: self._open_app(m.group(1))),
            (r"(?:search for|google) (.+)", lambda m: self._web_search(m.group(1))),
        ]
        return ([(re.compile(p), fn, True) for p, fn in table]
                + [(re.compile(p), fn, False) for p, fn in cmd_only])

    def _open_app(self, name):
        name = name.strip()
        target = _APPS.get(name, name)
        try:
            subprocess.Popen(["cmd", "/c", "start", "", target],
                             creationflags=subprocess.CREATE_NO_WINDOW)
            self._status(f"opening {name}")
        except OSError:
            self._status(f"could not open {name}")

    def _web_search(self, query):
        webbrowser.open("https://www.google.com/search?q="
                        + urllib.parse.quote(query))
        self._status(f"searching: {query}")

    def _set_mode(self, mode):
        self.mode = mode
        self._status(f"{mode} mode")

    def _mic_off(self):
        self.engine.mic_on = False
        self._status("mic off")

    def _status(self, msg):
        self._status_sent = True
        self.engine.voice_status = msg
        self.engine.notify(msg)
        print(f"[voice] {msg}")
