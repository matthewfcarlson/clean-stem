"""Split audio windows into vocals and instrumental with Demucs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_MODEL = "htdemucs"


@dataclass(frozen=True)
class Stems:
    start: float  # seconds into the song
    end: float
    vocals: np.ndarray  # (2, n) float32 at the model's sample rate
    instrumental: np.ndarray  # everything except vocals, same shape

    def slice(self, start: float, end: float, sample_rate: int) -> Stems:
        lo = int(round((start - self.start) * sample_rate))
        hi = lo + int(round((end - start) * sample_rate))
        return Stems(start, end, self.vocals[:, lo:hi], self.instrumental[:, lo:hi])


def merge_windows(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping (start, end) windows."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


class Separator:
    """Runs Demucs on song windows and remembers the stems for splicing."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from demucs.pretrained import get_model

        self.model = get_model(model_name)
        self.model.eval()
        self.sample_rate: int = self.model.samplerate
        self.stems: list[Stems] = []

    def prefetch(self, stereo: np.ndarray, sample_rate: int, windows: list[tuple[float, float]]) -> None:
        """Separate the union of `windows` up front, so overlapping requests
        are served from one Demucs pass instead of several."""
        for start, end in merge_windows(windows):
            self.separate(stereo, sample_rate, start, end)

    def separate(self, stereo: np.ndarray, sample_rate: int, start: float, end: float) -> Stems:
        """Separate stereo[:, start:end] (seconds); stereo must be at self.sample_rate.

        Reuses an already-separated window that covers the request.
        """
        for stems in self.stems:
            if stems.start <= start and end <= stems.end:
                return stems.slice(start, end, sample_rate)

        import torch
        from demucs.apply import apply_model

        if sample_rate != self.sample_rate:
            raise ValueError(f"expected {self.sample_rate} Hz audio, got {sample_rate}")
        segment = torch.from_numpy(stereo[:, int(start * sample_rate) : int(end * sample_rate)].copy())

        # Demucs expects input normalised the way its CLI does it.
        ref = segment.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            sources = apply_model(self.model, ((segment - mean) / std)[None], device="cpu", progress=False)[0]
        sources = sources * std + mean

        vocals_idx = self.model.sources.index("vocals")
        vocals = sources[vocals_idx]
        instrumental = sources.sum(0) - vocals
        stems = Stems(start, end, vocals.numpy(), instrumental.numpy())
        self.stems.append(stems)
        return stems
