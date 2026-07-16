import ctypes
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)

_INPUT_MOUSE = 0
_MOVE = 0x0001
_ABSOLUTE = 0x8000
_VIRTUALDESK = 0x4000
_LEFTDOWN, _LEFTUP = 0x0002, 0x0004
_RIGHTDOWN, _RIGHTUP = 0x0008, 0x0010
_MIDDLEDOWN, _MIDDLEUP = 0x0020, 0x0040
_WHEEL = 0x0800


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG), ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _IUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _IUNION)]


def _send_mouse(flags, dx=0, dy=0, data=0):
    inp = INPUT(type=_INPUT_MOUSE)
    inp.u.mi = MOUSEINPUT(dx, dy, data & 0xFFFFFFFF, flags, 0, 0)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


class VirtualScreen:
    """Bounding box of all monitors, in physical pixels."""

    def __init__(self):
        self.left = user32.GetSystemMetrics(76)    # SM_XVIRTUALSCREEN
        self.top = user32.GetSystemMetrics(77)
        self.width = user32.GetSystemMetrics(78)
        self.height = user32.GetSystemMetrics(79)


class Mouse:
    def __init__(self):
        self.vs = VirtualScreen()
        self._left_down = False
        self._right_down = False
        self.pos = (self.vs.left + self.vs.width // 2,
                    self.vs.top + self.vs.height // 2)

    def move_to(self, x, y):
        x, y = int(x), int(y)
        self.pos = (x, y)
        nx = round((x - self.vs.left) * 65535 / max(self.vs.width - 1, 1))
        ny = round((y - self.vs.top) * 65535 / max(self.vs.height - 1, 1))
        _send_mouse(_MOVE | _ABSOLUTE | _VIRTUALDESK, nx, ny)

    def left_down(self):
        if not self._left_down:
            self._left_down = True
            _send_mouse(_LEFTDOWN)

    def left_up(self):
        if self._left_down:
            self._left_down = False
            _send_mouse(_LEFTUP)

    def right_down(self):
        if not self._right_down:
            self._right_down = True
            _send_mouse(_RIGHTDOWN)

    def right_up(self):
        if self._right_down:
            self._right_down = False
            _send_mouse(_RIGHTUP)

    def click(self):
        self.left_down()
        self.left_up()

    def double_click(self):
        self.click()
        self.click()

    def right_click(self):
        self.right_down()
        self.right_up()

    def middle_click(self):
        _send_mouse(_MIDDLEDOWN)
        _send_mouse(_MIDDLEUP)

    def wheel(self, delta):
        _send_mouse(_WHEEL, data=int(delta))

    def release_all(self):
        self.left_up()
        self.right_up()
