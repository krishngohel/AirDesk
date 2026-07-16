import threading

import cv2


class Camera:
    """Threaded webcam reader that always serves the freshest frame."""

    def __init__(self, index=0, width=640, height=480, fps=30):
        self.cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {index} — run scripts/list_cameras.py "
                "and set camera.index in config/gestures.yaml"
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self._frame = None
        self._fresh = threading.Event()
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            with self._lock:
                self._frame = frame
            self._fresh.set()

    def read(self, timeout=1.0):
        """Block until a frame newer than the last read() is available."""
        if not self._fresh.wait(timeout):
            return None
        self._fresh.clear()
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def close(self):
        self._running = False
        self._thread.join(timeout=1)
        self.cap.release()
