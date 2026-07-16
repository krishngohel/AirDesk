"""Probe camera indices 0-4 so you can pick the RGB webcam (not the IR one)
in config/gestures.yaml. IR cameras show up as near-grayscale."""
import cv2
import numpy as np

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
    if not cap.isOpened():
        continue
    ok, frame = cap.read()
    if ok and frame is not None:
        h, w = frame.shape[:2]
        brightness = float(frame.mean())
        # color spread across channels: ~0 means grayscale (likely the IR cam)
        colorfulness = float(np.std(frame.astype(np.float32), axis=2).mean())
        kind = "likely IR/grayscale" if colorfulness < 2.0 else "color (use this)"
        if brightness < 15:
            kind += " — frame is nearly black (lens covered / dark room?)"
        print(f"index {i}: {w}x{h}  brightness={brightness:.0f}  "
              f"colorfulness={colorfulness:.1f}  -> {kind}")
    else:
        print(f"index {i}: opened but no frame")
    cap.release()
