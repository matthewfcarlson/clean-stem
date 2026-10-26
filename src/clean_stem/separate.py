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


class Separator:
    """Runs Demucs on song windows and remembers the stems for splicing."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from demucs.pretrained import get_model

        self.model = get_model(model_name)
        self.model.eval()
        self.sample_rate: int = self.model.samplerate
        self.stems: list[Stems] = []

    def separate(self, stereo: np.ndarray, sample_rate: int, start: float, end: float) -> Stems:
        """Separate stereo[:, start:end] (seconds); stereo must be at self.sample_rate."""
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
