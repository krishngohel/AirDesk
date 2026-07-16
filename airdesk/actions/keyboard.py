import ctypes

from .mouse import INPUT, KEYBDINPUT, user32

_INPUT_KEYBOARD = 1
_KEYUP = 0x0002
_UNICODE = 0x0004
_EXTENDEDKEY = 0x0001

VK = {
    "ctrl": 0x11, "alt": 0x12, "shift": 0x10, "win": 0x5B,
    "enter": 0x0D, "tab": 0x09, "space": 0x20, "backspace": 0x08,
    "delete": 0x2E, "escape": 0x1B, "esc": 0x1B,
    "home": 0x24, "end": 0x23, "pageup": 0x21, "pagedown": 0x22,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "plus": 0xBB, "minus": 0xBD,
    "vol_up": 0xAF, "vol_down": 0xAE, "mute": 0xAD,
    "play_pause": 0xB3, "next_track": 0xB0, "prev_track": 0xB1,
}
for _c in "abcdefghijklmnopqrstuvwxyz0123456789":
    VK[_c] = ord(_c.upper())
for _i in range(1, 13):
    VK[f"f{_i}"] = 0x6F + _i

# keys that need KEYEVENTF_EXTENDEDKEY or some apps misinterpret them
# (arrows/nav/delete/win/media — e.g. Win+arrow snapping breaks without it)
_EXTENDED_VKS = {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E,
                 0x5B, 0x5C, 0xAD, 0xAE, 0xAF, 0xB0, 0xB1, 0xB2, 0xB3}


def _key(vk, up=False):
    flags = (_KEYUP if up else 0) | (_EXTENDEDKEY if vk in _EXTENDED_VKS else 0)
    inp = INPUT(type=_INPUT_KEYBOARD)
    inp.u.ki = KEYBDINPUT(vk, 0, flags, 0, 0)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def _unicode_key(code_unit, up=False):
    inp = INPUT(type=_INPUT_KEYBOARD)
    inp.u.ki = KEYBDINPUT(0, code_unit, _UNICODE | (_KEYUP if up else 0), 0, 0)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


class Keyboard:
    def __init__(self):
        self._ctrl_held = False

    def tap(self, *names):
        vks = [VK[n] for n in names]
        for vk in vks:
            _key(vk)
        for vk in reversed(vks):
            _key(vk, up=True)

    def type_text(self, text):
        """Type arbitrary unicode text into the focused window."""
        for ch in text:
            if ch == "\n":
                self.tap("enter")
                continue
            if ch == "\t":
                self.tap("tab")
                continue
            # UTF-16 code units so emoji/surrogate pairs work too
            data = ch.encode("utf-16-le")
            for i in range(0, len(data), 2):
                cu = int.from_bytes(data[i:i + 2], "little")
                _unicode_key(cu)
                _unicode_key(cu, up=True)

    def backspace(self, n=1):
        for _ in range(min(n, 400)):
            self.tap("backspace")

    def ctrl_down(self):
        if not self._ctrl_held:
            self._ctrl_held = True
            _key(VK["ctrl"])

    def ctrl_up(self):
        if self._ctrl_held:
            self._ctrl_held = False
            _key(VK["ctrl"], up=True)

    def alt_tab(self):
        self.tap("alt", "tab")

    def win_tab(self):
        self.tap("win", "tab")

    def win_d(self):
        self.tap("win", "d")

    def volume(self, up):
        self.tap("vol_up" if up else "vol_down")

    def release_all(self):
        self.ctrl_up()
