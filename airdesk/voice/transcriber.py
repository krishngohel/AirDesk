import os
import site
from pathlib import Path

import numpy as np

RATE = 16000

# classic Whisper hallucinations on silence / breath noise
_HALLUCINATIONS = {
    "you", "so", "bye", "the end", "thank you", "thanks for watching",
    "thank you for watching", "thank you very much",
    "subtitles by the amara org community", "1", "okay",
}


def _add_nvidia_dll_dirs():
    """Make pip-installed cuBLAS/cuDNN visible to CTranslate2 on Windows."""
    roots = set(site.getsitepackages())
    try:
        roots.add(site.getusersitepackages())
    except AttributeError:
        pass
    for sp in roots:
        base = Path(sp) / "nvidia"
        if not base.is_dir():
            continue
        for d in base.glob("**/bin"):
            if d.is_dir():
                try:
                    os.add_dll_directory(str(d))
                    os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
                except OSError:
                    pass


def _resolve_model(name):
    """Prefer a fully-local copy in <project>/models so no network is needed."""
    if Path(name).is_dir():
        return name
    root = Path(__file__).resolve().parents[2]
    for cand in (root / "models" / name, root / "models" / f"whisper-{name}"):
        if cand.is_dir():
            return str(cand)
    return name


class Transcriber:
    """faster-whisper wrapper: device fallback, quality gates, vocab biasing."""

    def __init__(self, cfg, log=print):
        from faster_whisper import WhisperModel

        _add_nvidia_dll_dirs()
        name = _resolve_model(cfg.get("model", "turbo"))
        self.language = cfg.get("language", "en") or None
        vocab = cfg.get("vocabulary") or []
        self._prompt = f"Vocabulary: {', '.join(map(str, vocab))}." if vocab else None

        # on CPU a large model is painfully slow, so a lighter one can be
        # configured for the no-GPU case only (cpu_model: null = same model)
        cpu_name = _resolve_model(cfg.get("cpu_model") or name)
        if cfg.get("device", "auto") == "cpu":
            attempts = [("cpu", cpu_name, "int8")]
        else:
            attempts = [("cuda", name, "float16"), ("cpu", cpu_name, "int8")]
        err = None
        for device, model_name, compute in attempts:
            try:
                log(f"loading whisper '{Path(model_name).name}' on {device} ({compute})...")
                model = WhisperModel(model_name, device=device, compute_type=compute)
                # warm up with real audio: CUDA problems only surface at inference
                list(model.transcribe(np.zeros(RATE, dtype=np.float32),
                                      beam_size=1)[0])
                self.model, self.device = model, device
                log(f"whisper ready on {device}")
                return
            except Exception as e:  # missing cuDNN, no GPU, OOM...
                err = e
                log(f"  {device} failed: {e}")
        raise RuntimeError(f"could not load whisper model: {err}")

    def transcribe(self, audio):
        """audio: float32 mono 16 kHz. Returns '' when nothing real was said."""
        if audio is None or len(audio) < int(0.25 * RATE):
            return ""
        audio = np.asarray(audio, dtype=np.float32)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.0025:  # essentially silence; don't invite hallucinations
            return ""
        # padding helps very short utterances ("yes", "stop") decode reliably
        pad = np.zeros(int(0.3 * RATE), dtype=np.float32)
        audio = np.concatenate([pad, audio, pad])

        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=5,
            temperature=[0.0, 0.2, 0.4],
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            condition_on_previous_text=False,
            initial_prompt=self._prompt,
        )
        parts = []
        for seg in segments:
            if seg.no_speech_prob > 0.6 and seg.avg_logprob < -0.8:
                continue
            if seg.compression_ratio > 2.4:  # repetitive-loop hallucination
                continue
            parts.append(seg.text.strip())
        text = " ".join(p for p in parts if p).strip()

        norm = "".join(c for c in text.lower() if c.isalnum() or c == " ").strip()
        if norm in _HALLUCINATIONS and len(audio) < 4 * RATE:
            return ""
        return text
