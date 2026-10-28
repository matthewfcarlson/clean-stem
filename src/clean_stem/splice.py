"""Replace flagged words in a song with the matching instrumental audio."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from clean_stem.audio import resample
from clean_stem.separate import Stems

# Raised-cosine fade (seconds) into and out of each replaced span, placed just
# outside the span so the word itself is fully covered.
CROSSFADE = 0.03

STANDARD_MP3_BITRATES = (32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320)


@dataclass(frozen=True)
class Span:
    start: float
    end: float


def merge_spans(spans: list[Span], gap: float = 2 * CROSSFADE) -> list[Span]:
    """Merge spans that overlap or sit closer together than `gap` seconds."""
    merged: list[Span] = []
    for span in sorted(spans, key=lambda s: s.start):
        if merged and span.start - merged[-1].end <= gap:
            merged[-1] = Span(merged[-1].start, max(merged[-1].end, span.end))
        else:
            merged.append(span)
    return merged


def _covering_stems(span: Span, stems: list[Stems]) -> Stems:
    """The separated window that best covers the span."""

    def overlap(s: Stems) -> float:
        return min(s.end, span.end) - max(s.start, span.start)

    return max(stems, key=overlap)


def replace_spans(
    audio: np.ndarray,
    sample_rate: int,
    spans: list[Span],
    stems: list[Stems],
    stems_rate: int,
) -> np.ndarray:
    """Return a copy of `audio` (channels, n) with each span swapped for instrumental."""
    out = audio.copy()
    fade = int(CROSSFADE * sample_rate)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade, dtype=np.float32)) if fade else np.zeros(0, np.float32)

    for span in merge_spans(spans):
        window = _covering_stems(span, stems)
        instrumental = window.instrumental
        if stems_rate != sample_rate:
            instrumental = resample(instrumental, stems_rate, sample_rate)
        if instrumental.shape[0] != out.shape[0]:
            instrumental = np.broadcast_to(instrumental.mean(axis=0, keepdims=True), (out.shape[0], instrumental.shape[1]))
        window_start = int(window.start * sample_rate)

        # Mix weight for the instrumental across the song: 0 outside, 1 inside
        # the span, with crossfades on either side. Clamped to the window.
        lo = max(window_start, int(span.start * sample_rate) - fade)
        hi = min(window_start + instrumental.shape[1], int(span.end * sample_rate) + fade, out.shape[1])
        if hi <= lo:
            continue
        weight = np.ones(hi - lo, dtype=np.float32)
        n_in = min(fade, int(span.start * sample_rate) - lo)
        n_out = min(fade, hi - int(span.end * sample_rate))
        if n_in > 0:
            weight[:n_in] = ramp[-n_in:]
        if n_out > 0:
            weight[-n_out:] = ramp[::-1][:n_out]

        inst = instrumental[:, lo - window_start : hi - window_start]
        out[:, lo:hi] = out[:, lo:hi] * (1 - weight) + inst * weight

    return out


def encode_like(original: Path, audio: np.ndarray, sample_rate: int, dest: Path) -> None:
    """Encode (channels, n) float audio to `dest` in the same format as `original`."""
    import mutagen

    info = mutagen.File(original).info
    suffix = original.suffix.lower()
    if suffix == ".mp3":
        kbps = min(STANDARD_MP3_BITRATES, key=lambda b: abs(b - info.bitrate / 1000))
        codec = ["-c:a", "libmp3lame", "-b:a", f"{kbps}k", "-f", "mp3"]
    elif suffix == ".flac":
        # Match the original bit depth; unmodified samples then round-trip exactly.
        depth = ["-sample_fmt", "s16"] if info.bits_per_sample <= 16 else ["-sample_fmt", "s32", "-bits_per_raw_sample", "24"]
        codec = ["-c:a", "flac", *depth, "-compression_level", "8", "-f", "flac"]
    else:
        raise ValueError(f"unsupported format: {suffix}")

    channels = audio.shape[0]
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-y",
        "-f", "f32le", "-ac", str(channels), "-ar", str(sample_rate), "-i", "-",
        *codec, str(dest),
    ]  # fmt: skip
    pcm = np.clip(audio, -1.0, 1.0).T.astype(np.float32).tobytes()
    subprocess.run(cmd, input=pcm, check=True, capture_output=True)
    copy_tags(original, dest)


def copy_tags(original: Path, dest: Path) -> None:
    """Copy every tag (lyrics, artwork, ...) from `original` onto `dest`."""
    import mutagen
    from mutagen.flac import FLAC
    from mutagen.id3 import ID3

    if original.suffix.lower() == ".mp3":
        ID3(original).save(dest)
        return
    src, dst = mutagen.File(original), mutagen.File(dest)
    dst.delete()
    for key, value in (src.tags or {}).items():
        dst[key] = value
    if isinstance(src, FLAC):
        for picture in src.pictures:
            dst.add_picture(picture)
    dst.save()
