import queue
import threading

from .audio import Microphone, Segmenter
from .router import Router


class VoiceController:
    """Owns the mic capture and transcription threads. Audio only flows while
    engine.mic_on is True (pinky pinch / Ctrl+Alt+M / 'stop listening')."""

    def __init__(self, cfg, engine, kb, mouse):
        self.cfg = cfg["voice"]
        self.engine = engine
        self.kb = kb
        self.mouse = mouse
        self.running = True
        self._q = queue.Queue(maxsize=4)
        engine.mic_on = bool(self.cfg.get("start_on", False))
        engine.voice_status = "voice: loading model..."

    def start(self):
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._capture, daemon=True).start()

    def _capture(self):
        try:
            mic = Microphone(self.cfg.get("input_device"))
        except Exception as e:
            self.engine.voice_status = f"voice: no microphone ({e})"
            return
        seg = Segmenter(self.cfg["vad"])
        was_on = False
        while self.running:
            block = mic.read()
            on = self.engine.mic_on
            if not on:
                if was_on:
                    seg.reset()
                was_on = False
                continue
            was_on = True
            utt = seg.push(block)
            if utt is not None:
                try:
                    self._q.put_nowait(utt)
                except queue.Full:
                    self.engine.voice_status = "voice: busy, utterance dropped"

    def _worker(self):
        # model load happens here so the gesture loop starts instantly
        try:
            from .transcriber import Transcriber
            transcriber = Transcriber(self.cfg, log=self._log)
        except Exception as e:
            self.engine.voice_status = f"voice: model failed ({e})"
            return
        router = Router(self.cfg, self.kb, self.mouse, self.engine)
        self.engine.voice_status = (
            f"voice ready ({transcriber.device}) — {router.mode} mode, "
            f"mic {'on' if self.engine.mic_on else 'off'}")
        self.engine.notify(f"voice ready ({transcriber.device})")
        while self.running:
            utt = self._q.get()
            text = transcriber.transcribe(utt)
            if text:
                router.handle(text)

    def _log(self, msg):
        self.engine.voice_status = f"voice: {msg}"
        print(f"[voice] {msg}")

    def stop(self):
        self.running = False
