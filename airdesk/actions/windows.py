import ctypes
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)

GA_ROOT = 2
SWP_NOSIZE = 0x0001
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SW_RESTORE = 9
SW_MAXIMIZE = 3
MONITOR_DEFAULTTONEAREST = 2

# never grab the desktop or taskbar
_SKIP_CLASSES = {"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"}


class _POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class _RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                ("right", wintypes.LONG), ("bottom", wintypes.LONG)]


class _MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", _RECT),
                ("rcWork", _RECT), ("dwFlags", wintypes.DWORD)]


user32.WindowFromPoint.restype = wintypes.HWND
user32.WindowFromPoint.argtypes = [_POINT]
user32.GetAncestor.restype = wintypes.HWND
# without an explicit restype ctypes truncates handles to 32-bit ints
user32.MonitorFromWindow.restype = ctypes.c_void_p


def window_at(x, y):
    hwnd = user32.WindowFromPoint(_POINT(int(x), int(y)))
    if not hwnd:
        return None
    root = user32.GetAncestor(hwnd, GA_ROOT)
    if not root:
        return None
    buf = ctypes.create_unicode_buffer(64)
    user32.GetClassNameW(root, buf, 64)
    if buf.value in _SKIP_CLASSES:
        return None
    return root


def get_rect(hwnd):
    r = _RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    return r.left, r.top, r.right - r.left, r.bottom - r.top


def is_maximized(hwnd):
    return bool(user32.IsZoomed(hwnd))


def restore(hwnd):
    user32.ShowWindow(hwnd, SW_RESTORE)


def maximize(hwnd):
    user32.ShowWindow(hwnd, SW_MAXIMIZE)


def move_to(hwnd, x, y):
    user32.SetWindowPos(hwnd, None, int(x), int(y), 0, 0,
                        SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE)


def work_area(hwnd):
    mon = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
    mi = _MONITORINFO()
    mi.cbSize = ctypes.sizeof(_MONITORINFO)
    user32.GetMonitorInfoW(mon, ctypes.byref(mi))
    r = mi.rcWork
    return r.left, r.top, r.right - r.left, r.bottom - r.top


def snap(hwnd, side):
    """Snap to half-screen ('left'/'right') or maximize ('up')."""
    if side == "up":
        maximize(hwnd)
        return
    if is_maximized(hwnd):
        restore(hwnd)
    l, t, w, h = work_area(hwnd)
    if side == "left":
        x, y, cw, ch = l, t, w // 2, h
    else:
        x, y, cw, ch = l + w // 2, t, w - w // 2, h
    user32.SetWindowPos(hwnd, None, x, y, cw, ch,
                        SWP_NOZORDER | SWP_NOACTIVATE)
