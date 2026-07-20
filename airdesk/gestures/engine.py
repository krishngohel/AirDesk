import math
from collections import deque

import numpy as np

from ..vision.filters import OneEuro, OneEuro2D
from ..actions import windows as winctl
from .poses import HandState, Hysteresis

FINGERS = ("index", "middle", "ring", "pinky")


class Engine:
    """Turns per-frame hand states into mouse/keyboard/window actions.

    Modes: NONE, PENDING (pinch started, disambiguating), DRAG (left button
    held), RIGHT (right button held), WINMOVE (dragging a window), SCROLL,
    VOLUME, ZOOM (two-handed).
    """

    def __init__(self, cfg, mouse, kb, notify=None):
        self.cfg = cfg
        self.mouse = mouse
        self.kb = kb
        self.notify = notify or (lambda msg: None)
        self.gestures = cfg.get("gestures") or {}
        pc = cfg["pinch"]
        self.pinch = {
            label: {f: Hysteresis(pc["engage"], pc["release"]) for f in FINGERS}
            for label in ("Left", "Right")
        }
        self.cursor_filter = OneEuro2D(**cfg["cursor"]["one_euro"])
        self.zoom_filter = OneEuro(min_cutoff=1.5, beta=0.02)

        self.enabled = True
        self.mic_on = False
        self.mode = "NONE"
        self.status = ""
        self.voice_status = "voice off"

        self._pending = None        # {"kind": "left"|"index", "t0", "pos"}
        self._win = None            # {"hwnd", "rect", "grab"}
        self._trail = deque(maxlen=6)       # (t, x, y) cursor history px
        self._palm_trail = deque(maxlen=6)  # (t, x) primary palm, normalized
        self._two_trail = deque(maxlen=6)   # (t, y) avg of both palms
        self._scroll_prev = None
        self._scroll_accum = 0.0
        self._scroll_off = 0
        self._scroll_on = 0
        self._vol_prev = None
        self._vol_accum = 0.0
        self._zoom_prev = None
        self._zoom_accum = 0.0
        self._fist_t0 = None
        self._fist_armed = True
        self._swipe_cd = 0.0
        self._pinky_prev = False
        self._palm_miss = 0
        self.swipe_vx = 0.0     # live palm-swipe velocity, shown in preview
        self._point_n = 0       # consecutive frames of a sideways point
        self._point_dir = 0
        self._point_armed = True

    def _on(self, name):
        """A gesture is active unless explicitly disabled in config."""
        return self.gestures.get(name, True)

    # ------------------------------------------------------------- update

    def update(self, hands, t):
        if not hands:
            self._lost()
            return
        by_label = {}
        for h in hands:
            s = HandState(h)
            by_label.setdefault(h.label, s)

        pin = {
            label: {f: self.pinch[label][f].update(s.pinch[f]) for f in FINGERS}
            for label, s in by_label.items()
        }

        pref = self.cfg["tracking"]["primary_hand"]
        primary = by_label.get(pref) or next(iter(by_label.values()))

        # fist silence gesture works even while paused
        if self._on("fist_mute"):
            self._fist_logic(primary, t)
        if not self.enabled:
            self.status = "PAUSED"
            return

        # ---- two-hand gestures take priority ----
        left, right = by_label.get("Left"), by_label.get("Right")
        if left and right:
            if pin["Left"]["index"] and pin["Right"]["index"] and self._on("zoom"):
                self._zoom_step(left, right, t)
                return
            if self.mode == "ZOOM":
                self._end_zoom()
            if (left.pose == "palm" and right.pose == "palm"
                    and self._on("two_palm_swipe")):
                self._two_palm_swipe(left, right, t)
                return
            self._two_trail.clear()
        elif self.mode == "ZOOM":
            self._end_zoom()

        self._single(primary, pin[primary.hand.label], t)

    # ------------------------------------------------------- single hand

    def _single(self, s, pin, t):
        pose = s.pose
        # cursor follows the index fingertip (frozen while scrolling or
        # adjusting volume — those reuse vertical hand motion)
        if (self._on("cursor") and pose not in ("palm", "fist")
                and self.mode not in ("SCROLL", "VOLUME")):
            x, y = self._map(s.hand.pts[8], t)
            self.mouse.move_to(x, y)
            self._trail.append((t, x, y))

        pi, pm, pr, pk = pin["index"], pin["middle"], pin["ring"], pin["pinky"]
        grace = self.cfg["pinch"]["grace_ms"] / 1000.0
        hold = self.cfg["pinch"]["window_hold_ms"] / 1000.0

        # pinky+thumb tap toggles the mic
        if (pk and not self._pinky_prev and self.mode == "NONE"
                and not (pi or pm or pr) and self._on("mic_toggle")):
            self.set_mic(not self.mic_on)
        self._pinky_prev = pk

        m = self.mode
        if m == "NONE":
            self._from_idle(s, pi, pm, pr, pose, t)
        elif m == "PENDING":
            self._resolve_pending(pi, pm, t, grace, hold)
        elif m == "DRAG":
            if not pi:
                self.mouse.left_up()
                self.mode = "NONE"
                self.notify("drop")
        elif m == "RIGHT":
            if not pi and not pm:
                self.mouse.right_up()
                self.mode = "NONE"
        elif m == "WINMOVE":
            if pm:
                self._winmove_step()
            else:
                self._end_winmove(t)
        elif m == "SCROLL":
            if pose == "scroll":
                self._scroll_off = 0
                self._scroll_step(s)
            else:
                self._scroll_off += 1
                if self._scroll_off >= 3:
                    self.mode = "NONE"
                    self._scroll_prev = None
        elif m == "VOLUME":
            if pr:
                self._volume_step(s)
            else:
                self.mode = "NONE"
                self._vol_prev = None

        self.status = f"{self.mode}  mic:{'on' if self.mic_on else 'off'}"

    def _from_idle(self, s, pi, pm, pr, pose, t):
        if pose != "point":
            self._point_n = 0
            self._point_armed = True
        if pi and pm and self._on("right_click"):
            self._start_right()
        elif pi and self._on("click"):  # index pinch: tap = click, hold = drag
            self._pending = {"kind": "click", "t0": t, "pos": self.mouse.pos}
            self.mode = "PENDING"
        elif pm and self._on("window_move"):
            # middle pinch: tap = double-click, hold = move window
            self._pending = {"kind": "window", "t0": t, "pos": self.mouse.pos}
            self.mode = "PENDING"
        elif pr and self._on("volume"):
            self.mode = "VOLUME"
            self.notify("volume")
            self._vol_prev = float(s.hand.palm[1])
            self._vol_accum = 0.0
        elif pose == "scroll" and self._on("scroll"):
            self._scroll_on += 1
            if self._scroll_on >= 3:
                self.mode = "SCROLL"
                self.notify("scroll")
                self._scroll_prev = float(s.hand.palm[1])
                self._scroll_accum = 0.0
                self._scroll_off = 0
                self._scroll_on = 0
        else:
            self._scroll_on = 0
            if (pose == "point" and self._on("point_skip")
                    and not self._on("cursor")):
                self._point_skip(s, t)
            elif pose == "palm" and self._on("palm_swipe"):
                self._palm_miss = 0
                self._palm_swipe(s, t)
            else:
                # tolerate brief tracking flickers mid-swipe before
                # giving up on the trail
                self._palm_miss += 1
                if self._palm_miss >= 3:
                    self._palm_trail.clear()
                    self.swipe_vx = 0.0

    def _resolve_pending(self, pi, pm, t, grace, hold):
        p = self._pending
        if pi and pm:
            self._pending = None
            self._start_right()
            return
        if p["kind"] == "click":  # index pinch
            if not pi:  # quick tap -> click where the pinch started
                self.mouse.move_to(*p["pos"])
                self.mouse.click()
                self.notify("click")
                self._pending = None
                self.mode = "NONE"
            elif t - p["t0"] >= grace:
                # button goes down where the pinch started, not where the
                # cursor has drifted to since
                self.mouse.move_to(*p["pos"])
                self.mouse.left_down()
                self.notify("drag")
                self._pending = None
                self.mode = "DRAG"
        else:  # middle pinch
            if not pm:  # quick tap -> double-click
                self.mouse.move_to(*p["pos"])
                self.mouse.double_click()
                self.notify("double click")
                self._pending = None
                self.mode = "NONE"
            elif t - p["t0"] >= hold:  # held -> grab the window
                self._grab_window(p["pos"])
                self._pending = None

    # ------------------------------------------------------- right click

    def _start_right(self):
        self.mouse.right_down()
        self.notify("right click")
        self.mode = "RIGHT"

    # ------------------------------------------------------- window move

    def _grab_window(self, pos):
        hwnd = winctl.window_at(*pos)
        self.mode = "WINMOVE"
        if hwnd is None:
            self._win = None
            return
        if winctl.is_maximized(hwnd):
            winctl.restore(hwnd)
            _, _, w, _ = winctl.get_rect(hwnd)
            winctl.move_to(hwnd, pos[0] - w // 2, pos[1] - 12)
        self._win = {"hwnd": hwnd, "rect": winctl.get_rect(hwnd),
                     "grab": self.mouse.pos}
        self.notify("move window")

    def _winmove_step(self):
        if not self._win:
            return
        gx, gy = self._win["grab"]
        cx, cy = self.mouse.pos
        rx, ry, _, _ = self._win["rect"]
        winctl.move_to(self._win["hwnd"], rx + (cx - gx), ry + (cy - gy))

    def _end_winmove(self, t):
        win, self._win = self._win, None
        self.mode = "NONE"
        if not win or len(self._trail) < 2:
            return
        (t0, x0, y0), (t1, x1, y1) = self._trail[0], self._trail[-1]
        dt = max(t1 - t0, 1e-3)
        vx, vy = (x1 - x0) / dt, (y1 - y0) / dt
        speed = math.hypot(vx, vy)
        if speed < self.cfg["window"]["fling_px_s"]:
            return
        if abs(vx) > abs(vy):
            side = "right" if vx > 0 else "left"
            winctl.snap(win["hwnd"], side)
            self.notify(f"snap {side}")
        elif vy < 0:
            winctl.snap(win["hwnd"], "up")
            self.notify("maximize")

    # ------------------------------------------------------------ scroll

    def _scroll_step(self, s):
        c = self.cfg["scroll"]
        y = float(s.hand.palm[1])
        if self._scroll_prev is None:
            self._scroll_prev = y
            return
        delta = self._scroll_prev - y  # positive when the hand moves up
        self._scroll_prev = y
        if abs(delta) < c["deadzone"]:
            return
        if c["natural"]:
            delta = -delta
        self._scroll_accum += delta * c["gain"]
        step = c["step"]
        while abs(self._scroll_accum) >= step:
            chunk = step if self._scroll_accum > 0 else -step
            self.mouse.wheel(chunk)
            self._scroll_accum -= chunk

    # ------------------------------------------------------------ volume

    def _volume_step(self, s):
        y = float(s.hand.palm[1])
        if self._vol_prev is None:
            self._vol_prev = y
            return
        self._vol_accum += self._vol_prev - y  # up = louder
        self._vol_prev = y
        step = self.cfg["volume"]["step_norm"]
        while abs(self._vol_accum) >= step:
            self.kb.volume(self._vol_accum > 0)
            self._vol_accum -= step if self._vol_accum > 0 else -step

    # ------------------------------------------------- two-handed: zoom

    def _zoom_step(self, left, right, t):
        d = float(np.linalg.norm(left.hand.ptsi[8] - right.hand.ptsi[8]))
        if self.mode != "ZOOM":
            self._cancel_current()
            self.mode = "ZOOM"
            self.kb.ctrl_down()
            self.zoom_filter.reset()
            self._zoom_prev = self.zoom_filter(d, t)
            self._zoom_accum = 0.0
            self.status = "ZOOM"
            self.notify("zoom")
            return
        d = self.zoom_filter(d, t)
        # analog zoom: wheel amount is proportional to how far the hands
        # spread or contract this frame, flushed in small smooth chunks
        self._zoom_accum += (d - self._zoom_prev) * self.cfg["zoom"]["gain"]
        self._zoom_prev = d
        step = self.cfg["zoom"]["step"]
        while abs(self._zoom_accum) >= step:
            chunk = step if self._zoom_accum > 0 else -step
            self.mouse.wheel(chunk)
            self._zoom_accum -= chunk

    def _end_zoom(self):
        self.kb.ctrl_up()
        self.mode = "NONE"
        self._zoom_prev = None

    # ------------------------------------------- two-handed: palm swipes

    def _two_palm_swipe(self, left, right, t):
        y = (float(left.hand.palm[1]) + float(right.hand.palm[1])) / 2.0
        self._two_trail.append((t, y))
        if t < self._swipe_cd or len(self._two_trail) < 4:
            return
        (t0, y0), (t1, y1) = self._two_trail[0], self._two_trail[-1]
        vy = (y1 - y0) / max(t1 - t0, 1e-3)  # frame-heights per second
        thresh = self.cfg["swipe"]["vy_thresh"]
        if vy < -thresh:
            self.kb.win_tab()       # both palms swipe up -> Task View
            self.notify("task view")
        elif vy > thresh:
            self.kb.win_d()         # both palms swipe down -> Show Desktop
            self.notify("show desktop")
        else:
            return
        self._swipe_cd = t + self.cfg["swipe"]["cooldown_s"]
        self._two_trail.clear()

    def _palm_swipe(self, s, t):
        x = float(s.hand.palm[0])
        self._palm_trail.append((t, x))
        if t < self._swipe_cd or len(self._palm_trail) < 4:
            return
        (t0, x0), (t1, x1) = self._palm_trail[0], self._palm_trail[-1]
        vx = (x1 - x0) / max(t1 - t0, 1e-3)  # frame-widths per second
        self.swipe_vx = vx
        if abs(vx) < self.cfg["swipe"]["vx_thresh"]:
            return
        # frame is mirrored, so vx > 0 matches the hand moving right on screen
        if self.cfg["swipe"].get("palm_action", "media") == "media":
            if vx > 0:
                self.kb.tap("next_track")
                self.notify("next track")
            else:
                self.kb.tap("prev_track")
                self.notify("previous track")
        else:
            self.kb.alt_tab()
            self.notify("switch window")
        self._swipe_cd = t + self.cfg["swipe"]["cooldown_s"]
        self._palm_trail.clear()
        self.swipe_vx = 0.0

    # -------------------------------------------------- point: skip track

    def _point_skip(self, s, t):
        """Index finger pointing sideways, held still a beat = next/prev
        track. Pose-based, so unlike the palm swipe it doesn't depend on
        the tracker keeping up with fast hand motion."""
        c = self.cfg.get("point_skip") or {}
        v = s.hand.ptsi[8] - s.hand.ptsi[5]  # index knuckle -> fingertip
        dx, dy = float(v[0]), float(v[1])
        flat = (abs(dx) >= abs(dy) * c.get("min_tilt", 2.0)
                and abs(dx) >= s.span * c.get("min_reach", 0.5))
        if not flat:
            self._point_n = 0
            self._point_armed = True
            return
        d = 1 if dx > 0 else -1
        if d != self._point_dir:
            self._point_n = 0
            self._point_armed = True
        self._point_dir = d
        if t < self._swipe_cd:
            return
        self._point_n += 1
        # fires once per distinct point; drop the point and re-aim to repeat
        if self._point_n < c.get("hold_frames", 4) or not self._point_armed:
            return
        self._point_armed = False
        if d > 0:  # frame is mirrored, so dx > 0 = pointing right on screen
            self.kb.tap("next_track")
            self.notify("next track")
        else:
            self.kb.tap("prev_track")
            self.notify("previous track")
        self._swipe_cd = t + self.cfg["swipe"]["cooldown_s"]

    # ------------------------------------------------------- fist pause

    def _fist_logic(self, primary, t):
        """Fist held = silence: toggles media play/pause and speaker mute
        (both are system toggles, so a second fist-hold restores them)."""
        if primary.pose == "fist":
            if self._fist_t0 is None:
                self._fist_t0 = t
            elif self._fist_armed and t - self._fist_t0 >= self.cfg["fist_mute_hold_s"]:
                self._fist_armed = False
                self.kb.tap("play_pause")
                self.kb.tap("mute")
                self.notify("fist: play/pause + mute")
        else:
            self._fist_t0 = None
            self._fist_armed = True

    def toggle_enabled(self):
        self.enabled = not self.enabled
        self.notify("gestures resumed" if self.enabled else "gestures paused")
        if not self.enabled:
            self._cancel_current()

    def set_mic(self, on):
        self.mic_on = on
        self.notify("mic on" if on else "mic off")

    # ---------------------------------------------------------- cleanup

    def _cancel_current(self):
        self.mouse.release_all()
        self.kb.release_all()
        self._pending = None
        self._win = None
        self._scroll_prev = None
        self._vol_prev = None
        self._zoom_prev = None
        self._scroll_on = 0
        self.mode = "NONE"

    def _lost(self):
        """All hands left the frame: release everything and auto-pause;
        hands coming back into frame resume control instantly."""
        if self.mode != "NONE" or self.mouse._left_down or self.kb._ctrl_held:
            self._cancel_current()
        for trackers in self.pinch.values():
            for h in trackers.values():
                h.reset()
        self._fist_t0 = None
        self._trail.clear()
        self._palm_trail.clear()
        self._two_trail.clear()
        self._palm_miss = 0
        self.swipe_vx = 0.0
        self._point_n = 0
        self._point_armed = True
        self.status = "no hands — paused"

    # ---------------------------------------------------------- mapping

    def _map(self, pt, t):
        fx, fy = self.cursor_filter(float(pt[0]), float(pt[1]), t)
        box = self.cfg["cursor"]["box"]
        u = min(max((fx - box["left"]) / (box["right"] - box["left"]), 0.0), 1.0)
        v = min(max((fy - box["top"]) / (box["bottom"] - box["top"]), 0.0), 1.0)
        vs = self.mouse.vs
        return vs.left + u * (vs.width - 1), vs.top + v * (vs.height - 1)
