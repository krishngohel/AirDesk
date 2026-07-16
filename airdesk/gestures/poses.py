import numpy as np

WRIST = 0
THUMB_TIP = 4
INDEX_PIP, INDEX_TIP = 6, 8
MIDDLE_PIP, MIDDLE_TIP = 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20


def _d(pts, a, b):
    return float(np.linalg.norm(pts[a] - pts[b]))


class HandState:
    """Geometry of one hand -> pinch distances, finger extension, coarse pose."""

    def __init__(self, hand):
        self.hand = hand
        p = hand.ptsi
        self.span = _d(p, 0, 9) or 1e-6  # wrist -> middle knuckle
        self.pinch = {
            "index": _d(p, THUMB_TIP, INDEX_TIP) / self.span,
            "middle": _d(p, THUMB_TIP, MIDDLE_TIP) / self.span,
            "ring": _d(p, THUMB_TIP, RING_TIP) / self.span,
            "pinky": _d(p, THUMB_TIP, PINKY_TIP) / self.span,
        }
        self.ext = {
            name: _d(p, tip, WRIST) > _d(p, pip, WRIST) * 1.15
            for name, tip, pip in (
                ("index", INDEX_TIP, INDEX_PIP),
                ("middle", MIDDLE_TIP, MIDDLE_PIP),
                ("ring", RING_TIP, RING_PIP),
                ("pinky", PINKY_TIP, PINKY_PIP),
            )
        }
        n = sum(self.ext.values())
        if n == 0:
            self.pose = "fist"
        elif (self.ext["index"] and self.ext["middle"]
              and not self.ext["ring"] and not self.ext["pinky"]
              and self.pinch["index"] > 0.6 and self.pinch["middle"] > 0.6):
            self.pose = "scroll"
        elif n == 4:
            self.pose = "palm"
        elif self.ext["index"]:
            self.pose = "point"
        else:
            self.pose = "other"


class Hysteresis:
    """Debounced on/off from a continuous distance with two thresholds:
    engages below `engage`, releases only above `release`."""

    def __init__(self, engage, release, on_frames=2, off_frames=2):
        self.engage, self.release = engage, release
        self.on_frames, self.off_frames = on_frames, off_frames
        self._count = 0
        self.active = False

    def update(self, dist):
        if not self.active:
            if dist < self.engage:
                self._count += 1
                if self._count >= self.on_frames:
                    self.active = True
                    self._count = 0
            else:
                self._count = 0
        else:
            if dist > self.release:
                self._count += 1
                if self._count >= self.off_frames:
                    self.active = False
                    self._count = 0
            else:
                self._count = 0
        return self.active

    def reset(self):
        self.active = False
        self._count = 0
