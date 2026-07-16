"""Action toasts: a small always-on-top, click-through window in the
bottom-left corner showing what AirDesk just did (gestures and voice).

Tk lives on its own daemon thread; push() is safe from any thread.
"""
import ctypes
import queue
import threading
import time
from ctypes import wintypes

_MARGIN = 16
_GA_ROOT = 2
_GWL_EXSTYLE = -20
_WS_EX_TRANSPARENT = 0x00000020   # clicks pass through to whatever is below
_WS_EX_TOOLWINDOW = 0x00000080    # no taskbar button, no alt-tab entry
_WS_EX_NOACTIVATE = 0x08000000    # never steals focus

_BG = "#101418"
_FG_FRESH = "#e8edf2"
_FG_OLD = "#8a9199"


def _work_area():
    """Desktop rect minus the taskbar, in physical pixels."""
    rect = wintypes.RECT()
    ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)
    return rect


class Hud:
    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.ttl = float(cfg.get("ttl_s", 2.5))
        self.max_items = int(cfg.get("max_items", 4))
        self.opacity = float(cfg.get("opacity", 0.88))
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def push(self, text):
        self._q.put(str(text))

    def stop(self):
        self._q.put(None)
        self._thread.join(timeout=2)

    # ------------------------------------------------------------- tk thread

    def _click_through(self, root):
        user32 = ctypes.windll.user32
        hwnd = user32.GetAncestor(root.winfo_id(), _GA_ROOT)
        ex = user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
        user32.SetWindowLongW(hwnd, _GWL_EXSTYLE,
                              ex | _WS_EX_TRANSPARENT | _WS_EX_TOOLWINDOW
                              | _WS_EX_NOACTIVATE)

    def _run(self):
        import tkinter as tk

        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", self.opacity)
        root.configure(bg=_BG)
        frame = tk.Frame(root, bg=_BG, padx=12, pady=7)
        frame.pack()
        root.update_idletasks()
        self._click_through(root)
        root.withdraw()

        items = []  # oldest first: [text, expires_monotonic, repeat_count]

        def render():
            for w in frame.winfo_children():
                w.destroy()
            now = time.monotonic()
            for text, expires, count in items:
                msg = text if count == 1 else f"{text}  ×{count}"
                fresh = (expires - now) / self.ttl > 0.4
                tk.Label(frame, text=msg, bg=_BG,
                         fg=_FG_FRESH if fresh else _FG_OLD,
                         anchor="w", font=("Segoe UI", 10)).pack(fill="x")
            root.update_idletasks()
            wa = _work_area()
            root.geometry(f"+{wa.left + _MARGIN}"
                          f"+{wa.bottom - root.winfo_reqheight() - _MARGIN}")

        def tick():
            now = time.monotonic()
            try:
                while True:
                    msg = self._q.get_nowait()
                    if msg is None:
                        root.destroy()
                        return
                    if items and items[-1][0] == msg:
                        items[-1][1] = now + self.ttl
                        items[-1][2] += 1
                    else:
                        items.append([msg, now + self.ttl, 1])
                        del items[:-self.max_items]
            except queue.Empty:
                pass
            items[:] = [it for it in items if it[1] > now]
            if items:
                render()
                root.deiconify()
            else:
                root.withdraw()
            root.after(120, tick)

        tick()
        root.mainloop()
        # the tick/render closures keep the Tk object in a reference cycle;
        # collect it HERE so the Tcl interpreter dies on its own thread
        # (otherwise: "Tcl_AsyncDelete: async handler deleted by the wrong
        # thread" at exit)
        del frame, root, render, tick
        import gc
        gc.collect()
