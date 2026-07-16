import cv2

from ..vision.landmarks import HAND_CONNECTIONS as _CONNECTIONS

_GREEN = (80, 220, 80)
_RED = (60, 60, 230)
_WHITE = (240, 240, 240)


def draw(frame, hands, engine, fps):
    h, w = frame.shape[:2]
    for hd in hands:
        pts = (hd.pts * (w, h)).astype(int)
        for a, b in _CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), _GREEN, 1)
        for i, p in enumerate(pts):
            r = 5 if i in (4, 8) else 2  # highlight thumb + index tips
            cv2.circle(frame, tuple(p), r, _WHITE, -1)

    box = engine.cfg["cursor"]["box"]
    cv2.rectangle(frame,
                  (int(box["left"] * w), int(box["top"] * h)),
                  (int(box["right"] * w), int(box["bottom"] * h)),
                  (120, 120, 120), 1)

    color = _GREEN if engine.enabled else _RED
    cv2.putText(frame, engine.status, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    mic_color = _GREEN if engine.mic_on else (160, 160, 160)
    cv2.putText(frame, engine.voice_status, (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, mic_color, 1)
    cv2.putText(frame, f"{fps:.0f} fps", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1)
    return frame
