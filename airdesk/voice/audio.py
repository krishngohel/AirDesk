import queue
from collections import deque

import numpy as np

RATE = 16000
BLOCK = 480  # 30 ms


class Microphone:
    """Mic reader that always yields 16 kHz mono float32 blocks.

    Uses the callback API (the blocking API isn't supported on WDM-KS
    devices), survives machines with no default input device (picks the
    first capture device, preferring one named 'microphone'), and devices
    that won't open at 16 kHz (captures at 48k/44.1k/native, resamples).
    """

    def __init__(self, device=None):
        import sounddevice as sd
        self._q = queue.Queue(maxsize=256)
        self._acc = np.zeros(0, dtype=np.float32)
        candidates = ([device] if device is not None
                      else self._candidate_devices(sd))
        err = None
        for dev in candidates:
            rates = [RATE, 48000, 44100]
            try:
                native = int(sd.query_devices(dev)["default_samplerate"])
                if native not in rates:
                    rates.append(native)
            except Exception:
                pass
            for rate in rates:
                try:
                    self.stream = sd.InputStream(
                        samplerate=rate, channels=1, dtype="float32",
                        blocksize=int(rate * BLOCK / RATE), device=dev,
                        callback=self._on_audio)
                    self.stream.start()
                    self.rate = rate
                    self.device = dev
                    return
                except Exception as e:
                    err = e
        raise RuntimeError(f"could not open any microphone: {err}")

    def _on_audio(self, indata, frames, time_info, status):
        try:
            self._q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass  # consumer stalled; dropping is better than blocking audio

    @staticmethod
    def _candidate_devices(sd):
        devs = sd.query_devices()
        inputs = [i for i, d in enumerate(devs) if d["max_input_channels"] > 0]
        if not inputs:
            raise RuntimeError("no audio input devices found")
        order = []
        try:
            idx = sd.default.device[0]
            if idx is not None and idx >= 0:
                order.append(idx)
        except Exception:
            pass
        # prefer devices that are actual microphones by name
        order += [i for i in inputs
                  if "microphone" in devs[i]["name"].lower() and i not in order]
        order += [i for i in inputs if i not in order]
        return order

    def read(self):
        """Return the next 30 ms block at 16 kHz (blocks until available)."""
        n = int(self.rate * BLOCK / RATE)
        while len(self._acc) < n:
            self._acc = np.concatenate([self._acc, self._q.get()])
        chunk, self._acc = self._acc[:n], self._acc[n:]
        if self.rate != RATE:
            if self.rate % RATE == 0:  # integer decimation with crude LPF
                k = self.rate // RATE
                chunk = chunk.reshape(-1, k).mean(axis=1)
        else:
            return chunk.astype(np.float32)
        if self.rate % RATE != 0:      # e.g. 44.1 kHz: linear resample
            x = np.linspace(0.0, 1.0, len(chunk), endpoint=False)
            xi = np.linspace(0.0, 1.0, BLOCK, endpoint=False)
            chunk = np.interp(xi, x, chunk)
        return chunk.astype(np.float32)

    def close(self):
        self.stream.stop()
        self.stream.close()


class Segmenter:
    """Energy-based utterance endpointing with pre-roll so the first syllable
    is never clipped, an end-of-speech hangover, and a max-length flush."""

    def __init__(self, cfg):
        self.rms_start = cfg["rms_start"]
        self.rms_end = cfg["rms_end"]
        self.silence_end_s = cfg["silence_end_s"]
        self.min_s = cfg["min_utterance_s"]
        self.max_s = cfg["max_utterance_s"]
        n_pre = max(1, int(cfg["pre_roll_s"] * RATE / BLOCK))
        self._pre = deque(maxlen=n_pre)
        self._buf = []
        self._silence = 0.0
        self.active = False

    def push(self, block):
        """Feed one block; returns a finished utterance array or None."""
        rms = float(np.sqrt(np.mean(block ** 2)))
        dur = len(block) / RATE
        if not self.active:
            self._pre.append(block)
            if rms >= self.rms_start:
                self.active = True
                self._buf = list(self._pre)
                self._silence = 0.0
            return None

        self._buf.append(block)
        if rms < self.rms_end:
            self._silence += dur
        else:
            self._silence = 0.0

        total = len(self._buf) * dur
        if self._silence >= self.silence_end_s or total >= self.max_s:
            utt = np.concatenate(self._buf)
            speech_len = total - self._silence
            self.reset()
            if total >= self.max_s:      # long monologue: stay hot for the rest
                self.active = True
            if speech_len >= self.min_s:
                return utt
        return None

    def reset(self):
        self._buf = []
        self._silence = 0.0
        self.active = False
        self._pre.clear()
