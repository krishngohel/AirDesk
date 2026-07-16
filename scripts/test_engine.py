"""Offline test of the gesture engine: feeds synthetic hand landmarks through
the real state machine with fake input devices (nothing touches the real
mouse/keyboard/windows). Run: python scripts/test_engine.py"""
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airdesk.vision.landmarks import HandData
from airdesk.gestures import engine as engine_mod
from airdesk.gestures.engine import Engine


# ----------------------------------------------------------------- fakes

class FakeVS:
    left, top, width, height = 0, 0, 1920, 1080


class FakeMouse:
    def __init__(self):
        self.vs = FakeVS()
        self.pos = (960, 540)
        self.events = []
        self._left_down = False
        self.down_positions = []

    def move_to(self, x, y):
        self.pos = (int(x), int(y))

    def left_down(self):
        self._left_down = True
        self.events.append("left_down")
        self.down_positions.append(self.pos)

    def left_up(self):
        if self._left_down:
            self._left_down = False
            self.events.append("left_up")

    def right_down(self):
        self.events.append("right_down")

    def right_up(self):
        self.events.append("right_up")

    def click(self):
        self.events.append("click")

    def double_click(self):
        self.events.append("double_click")

    def middle_click(self):
        self.events.append("middle_click")

    def wheel(self, d):
        self.events.append(("wheel", int(d)))

    def release_all(self):
        self.left_up()


class FakeKb:
    def __init__(self):
        self.events = []
        self._ctrl_held = False

    def ctrl_down(self):
        self._ctrl_held = True
        self.events.append("ctrl_down")

    def ctrl_up(self):
        if self._ctrl_held:
            self._ctrl_held = False
            self.events.append("ctrl_up")

    def tap(self, *names):
        self.events.append(("tap",) + names)

    def alt_tab(self):
        self.events.append("alt_tab")

    def win_tab(self):
        self.events.append("win_tab")

    def win_d(self):
        self.events.append("win_d")

    def volume(self, up):
        self.events.append(("volume", up))

    def release_all(self):
        self.ctrl_up()


class FakeWin:
    def __init__(self):
        self.events = []

    def window_at(self, x, y):
        return 42

    def get_rect(self, hwnd):
        return (100, 100, 800, 600)

    def is_maximized(self, hwnd):
        return False

    def restore(self, hwnd):
        self.events.append("restore")

    def maximize(self, hwnd):
        self.events.append("maximize")

    def move_to(self, hwnd, x, y):
        self.events.append(("win_move", x, y))

    def snap(self, hwnd, side):
        self.events.append(("snap", side))


# --------------------------------------------------- synthetic hand data

TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIPS = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
MCPS = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
COLS = {"index": -0.06, "middle": -0.02, "ring": 0.02, "pinky": 0.06}


def make_hand(label="Right", extended=("index", "middle", "ring", "pinky"),
              pinched=(), cx=0.5, cy=0.5, dx=0.0, dy=0.0):
    """Build a plausible 21-landmark hand. `pinched` fingers get the thumb tip
    placed on their fingertip. cx/cy positions the hand; dx/dy offsets it."""
    cx, cy = cx + dx, cy + dy
    pts = np.zeros((21, 2), dtype=np.float32)
    pts[0] = (cx, cy + 0.25)                     # wrist
    for f, off in COLS.items():
        pts[MCPS[f]] = (cx + off, cy)
        pts[PIPS[f]] = (cx + off, cy - 0.06)
        if f in extended:
            pts[TIPS[f]] = (cx + off, cy - 0.16)
        else:
            pts[TIPS[f]] = (cx + off, cy + 0.05)  # curled toward wrist
    pts[1] = (cx - 0.10, cy + 0.20)              # thumb chain
    pts[2] = (cx - 0.13, cy + 0.14)
    pts[3] = (cx - 0.15, cy + 0.09)
    pts[4] = (cx - 0.16, cy + 0.05)              # thumb tip, far from fingers
    if pinched:
        f = pinched[0]
        target = pts[TIPS[f]].copy()
        pts[4] = target + (0.005, 0.005)
        for other in pinched[1:]:
            pts[TIPS[other]] = target + (0.01, 0.0)
    for i in (7, 11, 15, 19):                    # unused DIP joints
        pts[i] = pts[i - 1]
    return HandData(label, pts, 1.0)


class Runner:
    def __init__(self):
        cfg = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "config" / "gestures.yaml")
            .read_text())
        self.mouse = FakeMouse()
        self.kb = FakeKb()
        self.win = FakeWin()
        engine_mod.winctl = self.win
        self.eng = Engine(cfg, self.mouse, self.kb)
        self.t = 100.0

    def feed(self, hands, frames=1):
        for _ in range(frames):
            self.t += 0.02  # 50 fps
            self.eng.update(hands if isinstance(hands, list) else [hands], self.t)


FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# ----------------------------------------------------------------- tests

def test_cursor():
    r = Runner()
    r.feed(make_hand(extended=("index",)), 3)
    p1 = r.mouse.pos
    r.feed(make_hand(extended=("index",), dx=0.10), 6)
    check("cursor follows index finger", r.mouse.pos[0] > p1[0],
          f"{p1} -> {r.mouse.pos}")


def test_left_click_tap():
    r = Runner()
    r.feed(make_hand(extended=("index",)), 3)
    r.feed(make_hand(extended=("index",), pinched=("index",)), 3)
    r.feed(make_hand(extended=("index",)), 3)
    check("index+thumb tap = left click", "click" in r.mouse.events,
          str(r.mouse.events))


def test_drag():
    r = Runner()
    r.feed(make_hand(extended=("index",)), 3)
    r.feed(make_hand(extended=("index",), pinched=("index",)), 16)  # 320ms
    down = "left_down" in r.mouse.events
    r.feed(make_hand(extended=("index",)), 4)
    check("index pinch hold = drag (down then up)",
          down and "left_up" in r.mouse.events, str(r.mouse.events))


def test_right_click():
    r = Runner()
    r.feed(make_hand(extended=("index",)), 3)
    r.feed(make_hand(extended=(), pinched=("index", "middle")), 5)
    r.feed(make_hand(extended=("index",)), 4)
    ev = r.mouse.events
    check("two-finger pinch = right click",
          "right_down" in ev and "right_up" in ev, str(ev))


def test_double_click():
    r = Runner()
    r.feed(make_hand(extended=("index",)), 3)
    r.feed(make_hand(extended=("index",), pinched=("middle",)), 3)
    r.feed(make_hand(extended=("index",)), 3)
    check("middle+thumb tap = double-click", "double_click" in r.mouse.events,
          str(r.mouse.events))


def test_window_move():
    r = Runner()
    r.feed(make_hand(extended=("index",)), 3)
    r.feed(make_hand(extended=("index",), pinched=("middle",)), 26)  # >450ms
    check("middle pinch-hold enters WINMOVE", r.eng.mode == "WINMOVE", r.eng.mode)
    r.feed(make_hand(extended=("index",), pinched=("middle",), dx=0.05), 5)
    moved = any(e[0] == "win_move" for e in r.win.events if isinstance(e, tuple))
    check("window follows the hand", moved, str(r.win.events[:3]))
    r.feed(make_hand(extended=("index",)), 4)
    check("release ends window move", r.eng.mode == "NONE", r.eng.mode)


def test_scroll():
    r = Runner()
    r.feed(make_hand(extended=("index",)), 3)
    for i in range(12):
        r.feed(make_hand(extended=("index", "middle"), dy=-0.012 * i))
    wheels = [e for e in r.mouse.events if isinstance(e, tuple) and e[0] == "wheel"]
    check("scroll pose + hand up = wheel up",
          len(wheels) > 0 and all(w[1] > 0 for w in wheels),
          f"{len(wheels)} wheel events: {wheels[:4]}")


def test_zoom():
    r = Runner()
    lh = make_hand("Left", extended=(), pinched=("index",), cx=0.42)
    rh = make_hand("Right", extended=(), pinched=("index",), cx=0.58)
    r.feed([lh, rh], 4)
    check("both-hands pinch enters ZOOM with ctrl held",
          r.eng.mode == "ZOOM" and r.kb._ctrl_held,
          f"mode={r.eng.mode} ctrl={r.kb._ctrl_held}")
    for i in range(10):  # spread hands apart
        lh = make_hand("Left", extended=(), pinched=("index",), cx=0.42 - 0.012 * i)
        rh = make_hand("Right", extended=(), pinched=("index",), cx=0.58 + 0.012 * i)
        r.feed([lh, rh])
    wheels = [e for e in r.mouse.events if isinstance(e, tuple) and e[0] == "wheel"]
    check("spreading hands zooms in (positive wheel)",
          len(wheels) >= 3 and all(w[1] > 0 for w in wheels),
          f"{len(wheels)} wheels: {wheels[:4]}")
    n1 = len(wheels)
    for i in range(10):  # small spread -> should produce fewer wheel units
        r.feed([make_hand("Left", extended=(), pinched=("index",), cx=0.31 - 0.002 * i),
                make_hand("Right", extended=(), pinched=("index",), cx=0.69 + 0.002 * i)])
    wheels2 = [e for e in r.mouse.events if isinstance(e, tuple) and e[0] == "wheel"]
    check("zoom is proportional to spread distance", len(wheels2) - n1 < n1,
          f"big spread {n1} vs small spread {len(wheels2) - n1}")
    r.feed(make_hand("Right", extended=("index",)), 3)
    check("zoom exit releases ctrl", not r.kb._ctrl_held and "ctrl_up" in r.kb.events,
          str(r.kb.events))


def test_two_palm_swipes():
    r = Runner()
    for i in range(8):  # both palms swipe up fast
        r.feed([make_hand("Left", cx=0.35, dy=-0.05 * i),
                make_hand("Right", cx=0.65, dy=-0.05 * i)])
    check("two-palm swipe up = Task View", "win_tab" in r.kb.events,
          str(r.kb.events))
    r2 = Runner()
    for i in range(8):
        r2.feed([make_hand("Left", cx=0.35, dy=0.05 * i),
                 make_hand("Right", cx=0.65, dy=0.05 * i)])
    check("two-palm swipe down = Show Desktop", "win_d" in r2.kb.events,
          str(r2.kb.events))


def test_fist_silence():
    r = Runner()
    r.feed(make_hand(extended=("index",)), 3)
    r.feed(make_hand(extended=()), 55)  # fist > 1s
    check("fist hold pauses media and mutes",
          ("tap", "play_pause") in r.kb.events and ("tap", "mute") in r.kb.events,
          str(r.kb.events))
    n = r.kb.events.count(("tap", "mute"))
    r.feed(make_hand(extended=()), 55)  # still holding: must not repeat
    check("holding the fist doesn't repeat-fire",
          r.kb.events.count(("tap", "mute")) == n, str(r.kb.events))
    r.feed(make_hand(extended=("index",)), 3)  # unfist re-arms
    r.feed(make_hand(extended=()), 55)
    check("second fist-hold toggles again (restore)",
          r.kb.events.count(("tap", "mute")) == n + 1, str(r.kb.events))


def test_no_hands_pauses_and_resumes():
    r = Runner()
    r.feed(make_hand(extended=("index",)), 3)
    r.feed([], 3)  # hands leave the frame
    check("no hands auto-pauses", "paused" in r.eng.status, r.eng.status)
    p1 = r.mouse.pos
    r.feed(make_hand(extended=("index",), dx=0.10), 6)  # hands come back
    check("hands back in frame resume control", r.mouse.pos[0] > p1[0],
          f"{p1} -> {r.mouse.pos}")


def test_volume():
    r = Runner()
    r.feed(make_hand(extended=("index",)), 3)
    # real ring-pinch: other fingers stay extended, away from the thumb
    for i in range(10):
        r.feed(make_hand(extended=("index", "middle", "pinky"),
                         pinched=("ring",), dy=-0.015 * i))
    vols = [e for e in r.kb.events if isinstance(e, tuple) and e[0] == "volume"]
    check("ring pinch + hand up = volume up",
          len(vols) > 0 and all(v[1] for v in vols), str(vols))


def test_drag_anchors_at_pinch_start():
    r = Runner()
    r.feed(make_hand(extended=("index",)), 5)
    # keep drifting the hand right while the pinch is held through the grace
    for i in range(16):
        r.feed(make_hand(extended=("index",), pinched=("index",), dx=0.02 * i))
    check("drag button goes down where the pinch started",
          r.mouse.down_positions
          and r.mouse.down_positions[0][0] < r.mouse.pos[0],
          f"down at {r.mouse.down_positions}, cursor now {r.mouse.pos}")


def test_window_fling_snap():
    r = Runner()
    r.feed(make_hand(extended=("index",)), 3)
    r.feed(make_hand(extended=("index",), pinched=("middle",)), 26)
    for i in range(5):  # fast rightward move, then release
        r.feed(make_hand(extended=("index",), pinched=("middle",), dx=0.05 * i))
    r.feed(make_hand(extended=("index",), dx=0.25), 4)
    check("fast fling on release snaps the window",
          ("snap", "right") in r.win.events, str(r.win.events))


def test_zoom_out():
    r = Runner()
    r.feed([make_hand("Left", extended=(), pinched=("index",), cx=0.25),
            make_hand("Right", extended=(), pinched=("index",), cx=0.75)], 4)
    for i in range(10):  # bring hands together
        r.feed([make_hand("Left", extended=(), pinched=("index",), cx=0.25 + 0.015 * i),
                make_hand("Right", extended=(), pinched=("index",), cx=0.75 - 0.015 * i)])
    wheels = [e for e in r.mouse.events if isinstance(e, tuple) and e[0] == "wheel"]
    check("contracting hands zooms out (negative wheel)",
          len(wheels) >= 3 and all(w[1] < 0 for w in wheels),
          f"{len(wheels)} wheels: {wheels[:4]}")


def test_palm_swipe_alt_tab():
    r = Runner()
    for i in range(8):
        r.feed(make_hand(dx=0.05 * i))  # open palm moving fast right
    check("fast palm swipe = alt-tab", "alt_tab" in r.kb.events,
          str(r.kb.events))


def test_volume_freezes_cursor():
    r = Runner()
    r.feed(make_hand(extended=("index", "middle", "pinky")), 3)
    r.feed(make_hand(extended=("index", "middle", "pinky"),
                     pinched=("ring",)), 3)
    check("ring pinch enters VOLUME", r.eng.mode == "VOLUME", r.eng.mode)
    frozen = r.mouse.pos
    for i in range(6):
        r.feed(make_hand(extended=("index", "middle", "pinky"),
                         pinched=("ring",), dy=-0.02 * i))
    check("cursor is frozen while adjusting volume", r.mouse.pos == frozen,
          f"{frozen} -> {r.mouse.pos}")


def test_hand_lost_releases():
    r = Runner()
    r.feed(make_hand(extended=("index",)), 3)
    r.feed(make_hand(extended=("index",), pinched=("index",)), 16)  # dragging
    r.feed([], 2)
    check("hand lost releases held buttons",
          not r.mouse._left_down and "left_up" in r.mouse.events,
          str(r.mouse.events))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        print(t.__name__)
        t()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("all engine tests passed")
