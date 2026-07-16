"""Voice system tests.

  python scripts/test_voice.py            # router + segmenter unit tests
  python scripts/test_voice.py --stt DIR  # also transcribe every .wav in DIR
                                          # (filename stem, _ for spaces, is
                                          # the expected text) and score it
"""
import argparse
import difflib
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airdesk.voice.router import Router, _normalize
from airdesk.voice.audio import Segmenter, RATE, BLOCK

FAILURES = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# ----------------------------------------------------------------- fakes

class FakeKb:
    def __init__(self):
        self.events = []

    def tap(self, *names):
        self.events.append(("tap",) + names)

    def type_text(self, text):
        self.events.append(("type", text))

    def backspace(self, n=1):
        self.events.append(("backspace", n))

    def alt_tab(self):
        self.events.append("alt_tab")

    def win_tab(self):
        self.events.append("win_tab")

    def win_d(self):
        self.events.append("win_d")

    def volume(self, up):
        self.events.append(("volume", up))


class FakeMouse:
    def __init__(self):
        self.events = []

    def click(self):
        self.events.append("click")

    def double_click(self):
        self.events.append("double_click")

    def right_click(self):
        self.events.append("right_click")

    def middle_click(self):
        self.events.append("middle_click")

    def wheel(self, d):
        self.events.append(("wheel", d))


class FakeEngine:
    def __init__(self):
        self.mic_on = True
        self.voice_status = ""
        self.notifications = []

    def notify(self, msg):
        self.notifications.append(msg)


def make_router(mode="dictation"):
    kb, mouse, eng = FakeKb(), FakeMouse(), FakeEngine()
    r = Router({"mode": mode, "typing": {"smart_spacing": True}}, kb, mouse, eng)
    return r, kb, mouse, eng


# ------------------------------------------------------------ router tests

def test_dictation_types():
    r, kb, _, _ = make_router()
    r.handle("Hello world, this is a longer dictated sentence.")
    check("dictation types text",
          kb.events and kb.events[0][0] == "type"
          and "Hello world" in kb.events[0][1], str(kb.events))


def test_smart_spacing():
    r, kb, _, _ = make_router()
    r.handle("First sentence here we go.")
    r.handle("Second sentence arrives now.")
    joined = "".join(e[1] for e in kb.events if e[0] == "type")
    check("smart spacing between utterances", ". Second" in joined, joined)


def test_delete_that():
    r, kb, _, _ = make_router()
    r.handle("Some text that will be removed entirely.")
    typed = len(kb.events[0][1])
    r.handle("delete that")
    check("delete that backspaces exactly what was typed",
          ("backspace", typed) in kb.events, str(kb.events))


def test_commands():
    cases = [
        ("press enter", ("tap", "enter")),
        ("Enter.", ("tap", "enter")),          # whisper punctuation survives
        ("select all", ("tap", "ctrl", "a")),
        ("Copy that", ("tap", "ctrl", "c")),
        ("paste", ("tap", "ctrl", "v")),
        ("close window", ("tap", "alt", "f4")),
        ("new tab", ("tap", "ctrl", "t")),
        ("go back", ("tap", "alt", "left")),
        ("zoom in", ("tap", "ctrl", "plus")),
        ("page down", ("tap", "pagedown")),
        ("press control shift t", ("tap", "ctrl", "shift", "t")),
        ("snap left", ("tap", "win", "left")),
        ("maximize window", ("tap", "win", "up")),
        ("mute", ("tap", "mute")),
        ("next track", ("tap", "next_track")),
    ]
    for spoken, expected in cases:
        r, kb, _, _ = make_router(mode="command")
        r.handle(spoken)
        check(f"command: '{spoken}'", expected in kb.events, str(kb.events))


def test_mouse_commands():
    for spoken, expected in [("click", "click"), ("double click", "double_click"),
                             ("right click", "right_click"),
                             ("scroll down", ("wheel", -360))]:
        r, _, mouse, _ = make_router(mode="command")
        r.handle(spoken)
        check(f"command: '{spoken}'", expected in mouse.events, str(mouse.events))


def test_dictation_inline_commands():
    r, kb, _, _ = make_router()
    r.handle("new line")
    check("dictation: 'new line' presses enter, not typed",
          ("type", "\n") in kb.events, str(kb.events))
    r2, kb2, _, _ = make_router()
    r2.handle("We should select all the options in the menu carefully.")
    check("dictation: long sentence containing command words is typed",
          kb2.events[0][0] == "type", str(kb2.events))


def test_mode_switching():
    r, kb, _, _ = make_router()
    r.handle("command mode")
    check("switch to command mode", r.mode == "command", r.mode)
    r.handle("this matches nothing at all whatsoever")
    check("command mode never types unmatched text",
          not any(e[0] == "type" for e in kb.events if isinstance(e, tuple)),
          str(kb.events))
    r.handle("type hello there")
    check("command mode: explicit 'type X' types",
          any(e[0] == "type" and "hello there" in e[1] for e in kb.events
              if isinstance(e, tuple)), str(kb.events))
    r.handle("dictation mode")
    check("switch back to dictation", r.mode == "dictation", r.mode)


def test_stop_listening():
    r, _, _, eng = make_router()
    r.handle("stop listening")
    check("'stop listening' mutes mic", eng.mic_on is False, str(eng.mic_on))


def test_normalize():
    check("normalize strips whisper punctuation",
          _normalize(" Press Enter. ") == "press enter",
          _normalize(" Press Enter. "))
    check("normalize turns 'Ctrl+Shift+T' into words",
          _normalize("Ctrl+Shift+T.") == "ctrl shift t",
          _normalize("Ctrl+Shift+T."))


def test_plus_combo():
    r, kb, _, _ = make_router(mode="command")
    r.handle("Ctrl+S")
    check("'Ctrl+S' presses control s", ("tap", "ctrl", "s") in kb.events,
          str(kb.events))
    r2, kb2, _, _ = make_router(mode="command")
    r2.handle("press windows e")
    check("'press windows e' works", ("tap", "win", "e") in kb2.events,
          str(kb2.events))


def test_open_search_not_inline():
    r, kb, _, _ = make_router()  # dictation
    opened = []
    r._open_app = lambda name: opened.append(name)
    r.handle("open the door")
    check("dictation types 'open the door' instead of launching",
          not opened and any(e[0] == "type" for e in kb.events), str(kb.events))
    r2, _, _, _ = make_router(mode="command")
    opened2, searched2 = [], []
    r2._open_app = lambda name: opened2.append(name)
    r2._web_search = lambda q: searched2.append(q)
    r2.handle("open notepad")
    r2.handle("search for hand tracking papers")
    check("command mode launches apps", opened2 == ["notepad"], str(opened2))
    check("command mode searches the web",
          searched2 == ["hand tracking papers"], str(searched2))


def test_open_known_app_inline():
    r, kb, _, _ = make_router()  # dictation
    opened = []
    r._open_app = lambda name: opened.append(name)
    r.handle("open chrome")
    check("dictation launches known apps ('open chrome')",
          opened == ["chrome"] and not kb.events, f"{opened} {kb.events}")
    r.handle("open task manager")
    check("dictation launches multi-word known apps",
          opened == ["chrome", "task manager"], str(opened))


def test_status_reaches_hud():
    r, _, _, eng = make_router(mode="command")
    r.handle("new tab")
    check("plain commands echo to the HUD",
          "new tab" in eng.notifications, str(eng.notifications))
    r.handle("press control shift t")
    check("key combos notify the HUD",
          "pressed ctrl+shift+t" in eng.notifications, str(eng.notifications))
    r.handle("command mode")
    check("commands with their own status don't double-notify",
          eng.notifications.count("command mode") == 1
          and "command mode" not in eng.notifications[:-1],
          str(eng.notifications))


def test_all_command_keys_exist():
    """Every key name the grammar can tap must exist in the VK map."""
    from airdesk.actions.keyboard import Keyboard, VK

    class ValidatingKb(Keyboard):
        def tap(self, *names):
            for n in names:
                VK[n]  # KeyError -> unmapped key name

        def type_text(self, text):
            pass

        def backspace(self, n=1):
            pass

    phrases = [
        "enter", "backspace", "escape", "tab", "space", "delete",
        "select all", "copy", "paste", "cut", "save", "find", "undo", "redo",
        "close window", "close tab", "new tab", "reopen tab", "next tab",
        "previous tab", "switch window", "task view", "show desktop",
        "minimize", "maximize window", "snap left", "snap right",
        "go back", "go forward", "refresh", "page up", "page down",
        "go to top", "go to bottom", "zoom in", "zoom out", "reset zoom",
        "volume up", "volume down", "mute", "play", "next track",
        "previous song", "new line", "new paragraph",
        "press control shift escape", "press windows e", "press alt f4",
    ]
    kb, mouse, eng = ValidatingKb(), FakeMouse(), FakeEngine()
    r = Router({"mode": "command", "typing": {}}, kb, mouse, eng)
    bad = []
    for p in phrases:
        try:
            if not r._try_command(_normalize(p)):
                bad.append(f"unmatched: {p}")
        except KeyError as e:
            bad.append(f"unmapped key {e} for: {p}")
    check("every grammar phrase matches and maps to real keys", not bad,
          "; ".join(bad))


# --------------------------------------------------------- segmenter tests

def _blocks(seconds, level):
    n = int(seconds * RATE / BLOCK)
    rng = np.random.default_rng(0)
    return [rng.normal(0, level, BLOCK).astype(np.float32) for _ in range(n)]


def test_segmenter():
    cfg = {"rms_start": 0.010, "rms_end": 0.006, "silence_end_s": 0.6,
           "pre_roll_s": 0.4, "min_utterance_s": 0.25, "max_utterance_s": 25}
    seg = Segmenter(cfg)
    got = []
    for b in (_blocks(1.0, 0.001) + _blocks(1.5, 0.05) + _blocks(1.0, 0.001)):
        u = seg.push(b)
        if u is not None:
            got.append(u)
    check("segmenter emits one utterance for speech burst", len(got) == 1,
          f"{len(got)} utterances")
    if got:
        dur = len(got[0]) / RATE
        check("utterance includes pre-roll and speech (1.9-2.6s)",
              1.9 <= dur <= 2.6, f"{dur:.2f}s")

    seg2 = Segmenter(cfg)
    emitted = [seg2.push(b) for b in _blocks(3.0, 0.001)]
    check("segmenter ignores pure silence/noise floor",
          not any(u is not None for u in emitted), "")

    seg3 = Segmenter(cfg)
    tick = _blocks(0.09, 0.05) + _blocks(1.0, 0.001)  # 90ms keyboard click
    emitted = [seg3.push(b) for b in tick]
    check("segmenter drops sub-minimum blips (keyboard clicks)",
          not any(u is not None for u in emitted), "")

    seg4 = Segmenter(cfg)  # 30s monologue must split, not stall
    long_run = [seg4.push(b) for b in _blocks(30.0, 0.05)]
    utts = [u for u in long_run if u is not None]
    check("30s monologue is flushed at max length", len(utts) >= 1
          and abs(len(utts[0]) / RATE - cfg["max_utterance_s"]) < 1.0,
          f"{len(utts)} flushes, first {len(utts[0]) / RATE if utts else 0:.1f}s")
    check("segmenter stays hot after a max-length flush", seg4.active, "")


# ---------------------------------------------------------------- stt test

_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
          "six": 6, "seven": 7, "eight": 8, "nine": 9}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
          "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
          "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_ORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
         "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9}
_ALIASES = {"doctor": "dr", "mister": "mr", "okay": "ok"}


def _norm_words(s):
    """Number-format-agnostic normalization so 'twenty first' == '21st':
    whisper's inverse text normalization must not be scored as an error."""
    s = "".join(c if c.isalnum() else " " for c in s.lower())
    toks, out, i = s.split(), [], 0
    while i < len(toks):
        t = _ALIASES.get(toks[i], toks[i])
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if t in _TENS and (nxt in _UNITS or nxt in _ORDS):
            out.append(str(_TENS[t] + _UNITS.get(nxt, 0) + _ORDS.get(nxt, 0)))
            i += 2
            continue
        for table in (_TENS, _TEENS, _UNITS, _ORDS):
            if t in table:
                out.append(str(table[t]))
                break
        else:
            # strip ordinal suffix from digits: 21st -> 21
            if t[:-2].isdigit() and t[-2:] in ("st", "nd", "rd", "th"):
                out.append(t[:-2])
            else:
                out.append(t)
        i += 1
    return " ".join(out)


def test_stt(wav_dir):
    from airdesk.voice.transcriber import Transcriber
    tr = Transcriber({"model": "turbo", "device": "auto", "language": "en"})

    silence = np.zeros(RATE * 2, dtype=np.float32)
    check("silence transcribes to nothing", tr.transcribe(silence) == "",
          tr.transcribe(silence))
    noise = np.random.default_rng(1).normal(0, 0.002, RATE * 2).astype(np.float32)
    check("faint noise transcribes to nothing", tr.transcribe(noise) == "",
          tr.transcribe(noise))

    scores = []
    for wav_path in sorted(Path(wav_dir).glob("*.wav")):
        with wave.open(str(wav_path)) as w:
            assert w.getframerate() == RATE, f"{wav_path} must be 16 kHz"
            audio = (np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                     .astype(np.float32) / 32768.0)
        expected = wav_path.stem.split("--")[0].replace("_", " ")
        got = tr.transcribe(audio)
        ratio = difflib.SequenceMatcher(
            None, _norm_words(expected), _norm_words(got)).ratio()
        scores.append(ratio)
        check(f"stt {wav_path.name}: {ratio:.0%}", ratio >= 0.80,
              f"expected '{expected}' got '{got}'")
        print(f"        -> {got}")
    if scores:
        print(f"  mean accuracy: {sum(scores) / len(scores):.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stt", metavar="WAV_DIR",
                    help="also run transcription accuracy tests on this dir")
    args = ap.parse_args()

    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and name != "test_stt":
            print(name)
            fn()
    if args.stt:
        print("test_stt")
        test_stt(args.stt)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("all voice tests passed")
