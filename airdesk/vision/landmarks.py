import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/latest/hand_landmarker.task")

# bone connections between the 21 hand landmarks (for the debug overlay)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),
]


class HandData:
    __slots__ = ("label", "pts", "ptsi", "palm")

    def __init__(self, label, pts, aspect):
        self.label = label          # 'Left' / 'Right' in the mirrored (selfie) view
        self.pts = pts              # (21, 2) normalized [0..1] frame coords
        # isotropic coords (x scaled by aspect) so distances aren't squashed
        self.ptsi = pts * np.array([aspect, 1.0], dtype=np.float32)
        self.palm = (pts[0] + pts[5] + pts[17]) / 3.0


def _model_path():
    path = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print("downloading hand landmark model (one time, ~8 MB)...")
        urllib.request.urlretrieve(_MODEL_URL, path)
    return str(path)


class HandTracker:
    def __init__(self, max_hands=2, det_conf=0.6, track_conf=0.6):
        self._lm = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=_model_path()),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=det_conf,
                min_tracking_confidence=track_conf,
            )
        )
        self._last_ts = 0

    def process(self, frame_bgr):
        """frame_bgr must already be mirrored (selfie view)."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # VIDEO mode needs strictly increasing timestamps
        ts = max(int(time.perf_counter() * 1000), self._last_ts + 1)
        self._last_ts = ts
        res = self._lm.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        hands = []
        for lms, handed in zip(res.hand_landmarks, res.handedness):
            pts = np.array([(p.x, p.y) for p in lms], dtype=np.float32)
            hands.append(HandData(handed[0].category_name, pts, w / h))
        return hands

    def close(self):
        self._lm.close()
