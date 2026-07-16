"""Download the Whisper model with retries + resume (flaky-network safe).
Usage: python scripts/download_model.py [model-name]"""
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import snapshot_download

# faster-whisper's own alias table (model name -> HF repo)
from faster_whisper.utils import _MODELS

name = sys.argv[1] if len(sys.argv) > 1 else "turbo"
repo = _MODELS.get(name, name)
print(f"downloading {repo} ...")

for attempt in range(1, 11):
    try:
        path = snapshot_download(repo, max_workers=2)
        print("model ready at", path)
        break
    except Exception as e:
        print(f"attempt {attempt}/10 failed: {e}")
        time.sleep(min(10 * attempt, 60))
else:
    sys.exit(1)
