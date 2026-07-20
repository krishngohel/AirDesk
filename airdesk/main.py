import argparse
import ctypes
import os
import threading
import time
from ctypes import wintypes
from pathlib import Path

import cv2
import yaml

from .vision.camera import Camera
from .vision.landmarks import HandTracker
from .gestures.engine import Engine
from .actions.mouse import Mouse
from .actions.keyboard import Keyboard
from .ui.debug_view import draw

ROOT = Path(__file__).resolve().parents[1]


def _already_running():
    """Named mutex so a launch hotkey can't start a second instance
    (two processes fighting over the webcam)."""
    ctypes.windll.kernel32.CreateMutexW(None, False, "AirDesk.single_instance")
    return ctypes.windll.kernel32.GetLastError() == 183  # ERROR_ALREADY_EXISTS


def _dpi_aware():
    user32 = ctypes.windll.user32
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # per-monitor v2
    except (AttributeError, OSError):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            pass


def _hotkey_loop(engine):
    """Global hotkeys: Ctrl+Alt+Space = pause/resume, Ctrl+Alt+M = mic."""
    user32 = ctypes.WinDLL("user32")
    MOD_CONTROL, MOD_ALT, WM_HOTKEY = 0x2, 0x1, 0x0312
    if not user32.RegisterHotKey(None, 1, MOD_CONTROL | MOD_ALT, 0x20):
        print("warning: Ctrl+Alt+Space is held by another app — "
              "hide your hands to pause instead")
        engine.notify("Ctrl+Alt+Space taken — hide hands to pause")
    if not user32.RegisterHotKey(None, 2, MOD_CONTROL | MOD_ALT, 0x4D):  # M
        print("warning: Ctrl+Alt+M is held by another app — "
              "pinky pinch still toggles the mic")
        engine.notify("Ctrl+Alt+M taken — use pinky pinch")
    msg = wintypes.MSG()
    while user32.GetMessageW(ctypes.byref(msg), None, 0, 0):
        if msg.message != WM_HOTKEY:
            continue
        if msg.wParam == 1:
            engine.toggle_enabled()
            print("AirDesk", "ENABLED" if engine.enabled else "PAUSED")
        elif msg.wParam == 2:
            engine.set_mic(not engine.mic_on)
            print("AirDesk mic", "ON" if engine.mic_on else "OFF")


def main():
    ap = argparse.ArgumentParser(description="AirDesk — gesture computer control")
    ap.add_argument("--config", default=str(ROOT / "config" / "gestures.yaml"))
    ap.add_argument("--selftest", type=int, default=0, metavar="N",
                    help="process N frames with NO input injection, then report")
    args = ap.parse_args()

    if not args.selftest and _already_running():
        print("AirDesk is already running.")
        return

    cfg = yaml.safe_load(Path(args.config).read_text())
    _dpi_aware()

    cam_cfg = cfg["camera"]
    cam = Camera(cam_cfg["index"], cam_cfg["width"], cam_cfg["height"], cam_cfg["fps"])
    tr = cfg["tracking"]
    tracker = HandTracker(tr["max_hands"], tr["det_confidence"],
                          tr["track_confidence"])

    engine = None
    voice = None
    hud = None
    if not args.selftest:
        hud_cfg = cfg["ui"].get("hud", {})
        if hud_cfg.get("enabled", True):
            from .ui.hud import Hud
            hud = Hud(hud_cfg)
            hud.start()
        engine = Engine(cfg, Mouse(), Keyboard(),
                        notify=hud.push if hud else None)
        threading.Thread(target=_hotkey_loop, args=(engine,), daemon=True).start()
        if cfg.get("voice", {}).get("enabled"):
            try:
                from .voice.controller import VoiceController
                voice = VoiceController(cfg, engine, engine.kb, engine.mouse)
                voice.start()
            except ImportError:
                print("voice deps missing — run: "
                      ".venv\\Scripts\\pip install -r requirements-voice.txt")
        print("AirDesk running.  Hands out of frame = pause, "
              "fist-hold 1s = pause media + mute, "
              "Ctrl+Alt+M / pinky pinch = mic, Esc in preview = quit.")

    preview = cfg["ui"]["show_preview"] and not args.selftest
    menu = None
    if preview:
        win_name = "AirDesk  (Esc quits)"
        cv2.namedWindow(win_name)
        if engine:
            from .ui.menu import Menu
            menu = Menu(engine, Path(args.config))

            def _mouse(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    menu.on_click(x, y)

            cv2.setMouseCallback(win_name, _mouse)
    frames = 0
    hands_seen = 0
    fps = 0.0
    t_prev = time.perf_counter()
    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            frame = cv2.flip(frame, 1)  # mirror -> selfie view
            hands = tracker.process(frame)
            t = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(t - t_prev, 1e-4))
            t_prev = t

            if engine:
                engine.update(hands, t)

            frames += 1
            hands_seen += len(hands)
            if args.selftest and frames >= args.selftest:
                print(f"selftest ok: {frames} frames, avg fps {fps:.1f}, "
                      f"hand detections {hands_seen}")
                break

            if preview:
                img = draw(frame, hands, engine, fps)
                if menu:
                    img = menu.draw(img)
                cv2.imshow(win_name, img)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key == ord("g") and menu:
                    menu.toggle_visible()
    finally:
        if voice:
            voice.stop()
        if hud:
            hud.stop()
        if engine:
            engine._cancel_current()
        cam.close()
        cv2.destroyAllWindows()
    # tracker.close() and interpreter teardown are deliberately skipped:
    # mediapipe's serial dispatcher intermittently hard-crashes the process
    # (0xC0000005 / 0xC0000096) tearing down the HandLandmarker. All
    # user-visible state is already released above, so exit immediately.
    os._exit(0)


if __name__ == "__main__":
    main()
