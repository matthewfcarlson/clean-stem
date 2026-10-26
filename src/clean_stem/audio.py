"""Audio decoding via ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

WHISPER_SAMPLE_RATE = 16000


def _decode(path: Path, channels: int, sample_rate: int) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
        "-f", "f32le", "-ac", str(channels), "-ar", str(sample_rate), "-",
    ]  # fmt: skip
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def decode_mono(path: Path, sample_rate: int = WHISPER_SAMPLE_RATE) -> np.ndarray:
    """Decode an audio file to mono float32 samples at the given rate."""
    return _decode(path, 1, sample_rate)


def decode_stereo(path: Path, sample_rate: int) -> np.ndarray:
    """Decode an audio file to float32 samples shaped (2, n)."""
    return _decode(path, 2, sample_rate).reshape(-1, 2).T.copy()


def resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    import julius
    import torch

    return julius.resample_frac(torch.from_numpy(np.ascontiguousarray(samples)), from_rate, to_rate).numpy()
