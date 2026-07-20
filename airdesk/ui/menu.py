"""Clickable gesture menu drawn on the preview window.

Click a row (real mouse or the gesture cursor) to toggle that control on
the running engine; the change is also written back to config/gestures.yaml
so it survives restarts. Press `g` in the preview to show/hide the panel.
"""
import re

import cv2

ROWS = [
    ("cursor", "cursor"),
    ("click", "click / drag"),
    ("right_click", "right click"),
    ("window_move", "window move"),
    ("scroll", "scroll"),
    ("volume", "volume"),
    ("mic_toggle", "mic toggle"),
    ("fist_mute", "fist silence"),
    ("palm_swipe", "palm swipe: tracks"),
    ("point_skip", "point skip: tracks"),
    ("zoom", "zoom (2 hands)"),
    ("two_palm_swipe", "two-palm swipe"),
]

_ROW_H = 24
_PANEL_W = 200
_PAD = 8
_HEADER_H = 24
_ON = (80, 220, 80)
_OFF = (90, 90, 90)
_TEXT = (235, 235, 235)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


class Menu:
    def __init__(self, engine, config_path):
        self.engine = engine
        self.config_path = config_path
        self.visible = True
        self._origin = (0, 0)

    def toggle_visible(self):
        self.visible = not self.visible

    # --------------------------------------------------------------- draw

    def draw(self, frame):
        h, w = frame.shape[:2]
        if not self.visible:
            cv2.putText(frame, "g: menu", (w - 78, 20), _FONT, 0.45, _OFF, 1)
            return frame
        x0, y0 = w - _PANEL_W - 8, 8
        self._origin = (x0, y0)
        panel_h = _HEADER_H + len(ROWS) * _ROW_H + _PAD
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + _PANEL_W, y0 + panel_h),
                      (28, 22, 16), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.putText(frame, "controls  (click / g hides)",
                    (x0 + _PAD, y0 + 17), _FONT, 0.42, _TEXT, 1)
        for i, (key, label) in enumerate(ROWS):
            on = self.engine.gestures.get(key, True)
            ry = y0 + _HEADER_H + i * _ROW_H
            bx, by = x0 + _PAD, ry + 5
            if on:
                cv2.rectangle(frame, (bx, by), (bx + 12, by + 12), _ON, -1)
            else:
                cv2.rectangle(frame, (bx, by), (bx + 12, by + 12), _OFF, 1)
            cv2.putText(frame, label, (bx + 20, by + 11), _FONT, 0.45,
                        _TEXT if on else _OFF, 1)
        return frame

    # -------------------------------------------------------------- click

    def on_click(self, x, y):
        if not self.visible:
            return
        x0, y0 = self._origin
        if not (x0 <= x <= x0 + _PANEL_W):
            return
        i = (y - y0 - _HEADER_H) // _ROW_H
        if not (0 <= i < len(ROWS)):
            return
        key, label = ROWS[i]
        on = not self.engine.gestures.get(key, True)
        self.engine.gestures[key] = on
        self.engine.notify(f"{label} {'on' if on else 'OFF'}")
        self._persist(key, on)

    def _persist(self, key, value):
        """Flip the `  key: true/false` line under gestures: in the yaml,
        preserving all comments/formatting. Best-effort."""
        try:
            text = self.config_path.read_text()
            new, n = re.subn(
                rf"^(\s+{key}:\s*)(true|false)\b",
                lambda m: m.group(1) + ("true" if value else "false"),
                text, count=1, flags=re.M)
            if n:
                self.config_path.write_text(new)
        except OSError:
            pass
