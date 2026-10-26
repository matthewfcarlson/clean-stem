"""Audio decoding via ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

WHISPER_SAMPLE_RATE = 16000


def decode_mono(path: Path, sample_rate: int = WHISPER_SAMPLE_RATE) -> np.ndarray:
    """Decode an audio file to mono float32 samples at the given rate."""
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "-",
    ]  # fmt: skip
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)
