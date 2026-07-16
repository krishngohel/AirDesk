# AirDesk — control your computer with your hands and voice

Fully local gesture + voice control for Windows: a webcam + MediaPipe hand
tracking replaces the mouse; Whisper (large-v3-turbo, GPU-accelerated)
replaces the keyboard. Nothing ever leaves your machine.

## Run

```powershell
.\run.ps1            # first run creates .venv and installs deps
.venv\Scripts\pip install -r requirements-voice.txt   # once, for voice
```

Take your hands out of frame to pause (control resumes the moment they're
back); quit with Esc in the preview window. `Ctrl+Alt+M` toggles the mic
(`Ctrl+Alt+Space` pause also exists but another app owns that hotkey on
this machine). Your physical mouse and keyboard always keep working.

A small click-through **HUD** in the bottom-left corner shows every action
as it happens ("click", "opening chrome", "new tab ×2", "mic on", ...).
Tune or disable it under `ui.hud` in `config/gestures.yaml`.

## Gestures

Cursor follows your **index fingertip** (right hand by default).

| Gesture | Action |
|---|---|
| Move hand while pointing | Move cursor |
| Index + thumb pinch (tap) | Left click |
| Index + thumb pinch (hold) | Drag / select |
| Index + middle + thumb pinch together | Right click (hold, release to finish) |
| Middle + thumb pinch (quick tap) | Double-click |
| Middle + thumb pinch (hold ≥ 0.26s) | Grab + move the window under the cursor |
| ...then fling toward an edge on release | Snap window left/right, up = maximize |
| Index + middle extended ("scroll pose"), move hand up/down | Scroll (analog) |
| Ring + thumb pinch, move hand up/down | Volume |
| Pinky + thumb pinch (tap) | Toggle mic |
| Open palm, fast horizontal swipe | Alt+Tab |
| Open palm (still) | Neutral — reposition your hand without moving the cursor |
| Hands out of frame | Auto-pause; control resumes when they're back |
| Fist held 1s | Silence: pause media + mute speakers (fist again to restore) |

### Two-handed

| Gesture | Action |
|---|---|
| Both hands index-pinched, spread apart / together | Zoom in / out — smooth and proportional to how far you spread (Ctrl+scroll) |
| Both palms open, swipe up | Task View (Win+Tab) |
| Both palms open, swipe down | Show desktop (Win+D) |

## Voice

The mic starts **muted**. Unmute with a pinky pinch, `Ctrl+Alt+M`, and mute
back with those or by saying **"stop listening"**. Two modes, switchable
anytime by saying **"dictation mode"** / **"command mode"**:

- **Dictation** (default): everything you say is typed into the focused
  window with automatic punctuation and smart spacing. Short spoken commands
  (up to 4 words) still work inline — say "new line", "delete that"
  (removes the last utterance), "select all", "press enter", or
  "open chrome" (known apps only, so dictating "open the door" is still
  just typed). Longer sentences are always typed, even if they contain
  command words.
- **Command**: nothing is typed unless you say "type ..." — everything else
  is interpreted as a command; unrecognized speech is ignored.

Command highlights (see `airdesk/voice/router.py` for the full grammar):
"press enter / escape / tab...", "press control shift t" (any combo),
"select all / copy / paste / cut / undo / save", "close window",
"switch window", "new tab / close tab / next tab", "go back / forward",
"refresh", "scroll up / down", "page up / down", "go to top / bottom",
"zoom in / out / reset zoom", "click / double click / right click",
"snap left / right", "maximize / minimize", "show desktop", "task view",
"volume up / down", "mute", "play / pause", "next track",
"open chrome / notepad / terminal / settings / ...", "search for <anything>".

### Accuracy

- Whisper **large-v3-turbo** on your GPU (CUDA float16), the state of the
  art for accent-robust local recognition. With no GPU it automatically
  drops to `cpu_model` (small, 2.6x realtime on this machine) so dictation
  stays snappy; set `cpu_model: null` to force turbo on CPU (max accent
  robustness, ~7s lag per sentence).
- `language: en` is locked by default so accents are never misdetected as
  another language (set `null` for multilingual autodetect).
- Beam search (5) with temperature fallback; VAD-filtered decoding.
- Hallucination gates: silence/noise never types ("thank you for
  watching..."-style Whisper artifacts are filtered by no-speech
  probability, compression ratio, and a known-artifact list).
- Utterance capture keeps 0.4 s of pre-roll (first syllables never clip),
  pads short utterances, splits >25 s monologues, ignores keyboard-click
  blips, and survives mics that can't record at 16 kHz.
- Add your names/jargon to `voice.vocabulary` to bias recognition
  (e.g. `[Necto, Jajapur]`).

## Tuning

Everything lives in `config/gestures.yaml`:

- Cursor feels jittery → raise `cursor.one_euro.min_cutoff` slightly down, or lower `beta`.
- Cursor lags fast moves → raise `beta`.
- Pinches trigger too easily / not enough → adjust `pinch.engage` (lower = must pinch tighter).
- Zoom/scroll too fast or slow → `zoom.gain`, `scroll.gain`.
- Too much arm travel → shrink `cursor.box`.
- Wrong camera (IR instead of RGB) → `python scripts/list_cameras.py`, set `camera.index`.
- Low FPS → lower `camera.width`/`camera.height`.

## Troubleshooting (this machine)

- **GPU**: the RTX 4060 currently shows as a *phantom device* (not active),
  so Whisper runs on CPU. To restore GPU inference: run
  `pnputil /scan-devices` from an **admin** terminal, check Device Manager
  for the disabled GPU, check for an "Eco mode"/iGPU-only switch in your
  laptop's vendor app (MSI Center / Vantage / Armoury Crate), or reinstall
  the NVIDIA driver. With `device: auto`, AirDesk uses CUDA automatically
  the moment it's back.
- **Mic**: input is routed through "SteelSeries Sonar - Microphone"
  (auto-detected). If dictation hears nothing, make sure SteelSeries GG is
  running and Windows Settings → Privacy → Microphone allows desktop apps;
  or set `voice.input_device` to a specific index
  (`python -c "import sounddevice; print(sounddevice.query_devices())"`).
- **Model downloads**: Python's HTTP client gets connection-reset on this
  network sometimes. `scripts/download_model.py <name>` retries/resumes; the
  model is loaded from `models/whisper-turbo/` locally so normal runs need
  no network at all.

## Safety

- Physical keyboard/mouse are never blocked.
- Hands leaving the frame auto-pause and release all held buttons/keys
  instantly; Esc in the preview quits.

## Roadmap

- Calibration wizard, per-app profiles, tray icon.
