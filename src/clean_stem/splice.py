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
        codec = ["-c:a", "flac", "-f", "flac"]
    elif suffix == ".m4a" and info.codec == "alac":
        # Lossless; ffmpeg's ALAC encoder stores 16 bits from s16p and 24 from s32p.
        codec = ["-c:a", "alac", "-sample_fmt", "s32p" if info.bits_per_sample > 16 else "s16p", "-f", "ipod"]
    elif suffix == ".m4a" and info.codec.startswith("mp4a"):
        bps = info.bitrate or int(original.stat().st_size * 8 / info.length)
        codec = [*_aac_encoder(info.codec), "-b:a", f"{round(bps / 1000)}k", "-f", "ipod"]
        # ffmpeg decodes the original's end padding as audio; drop it so the length holds.
        priming, _, valid = _gapless(original)
        if priming and sample_rate == info.sample_rate:
            audio = audio[:, :valid]
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
    from mutagen.mp4 import MP4, MP4FreeForm

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
    elif isinstance(src, MP4) and ITUNSMPB in dst:
        # iTunSMPB records the encoder's priming/padding and ffmpeg trusts it
        # over the edit list, so the original's would shift the new encode.
        smpb = " 00000000 {:08X} {:08X} {:016X}".format(*_gapless(dest)) + " 00000000" * 8
        dst[ITUNSMPB] = [MP4FreeForm(smpb.encode(), dataformat=dst[ITUNSMPB][0].dataformat)]
    dst.save()


ITUNSMPB = "----:com.apple.iTunes:iTunSMPB"


def _aac_encoder(codec: str) -> list[str]:
    """ffmpeg args for the best available AAC encoder (fdk keeps an HE-AAC profile)."""
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], check=True, capture_output=True, text=True).stdout
    if any(line.split()[1:2] == ["libfdk_aac"] for line in encoders.splitlines()):
        profile = {"mp4a.40.5": "aac_he", "mp4a.40.29": "aac_he_v2"}.get(codec, "aac_low")
        return ["-c:a", "libfdk_aac", "-profile:a", profile]
    return ["-c:a", "aac"]


def _gapless(path: Path) -> tuple[int, int, int]:
    """(priming, padding, valid) samples of an MP4 audio track, from its edit list.

    AAC encoders prepend priming samples and pad the last frame; the edit list
    says which samples are real. Apple players go by it; ffmpeg prefers
    iTunSMPB for the priming and never trims the end padding.
    """
    import struct

    from mutagen.mp4 import Atoms

    with path.open("rb") as f:
        atoms = Atoms(f)

        def payload(*names: bytes) -> bytes | None:
            try:
                return atoms.path(b"moov", *names)[-1].read(f)[1]
            except KeyError:
                return None

        def timescale(box: bytes) -> int:  # mvhd / mdhd: version 1 has 64-bit times
            return struct.unpack_from(">I", box, 20 if box[0] == 1 else 12)[0]

        movie_scale = timescale(payload(b"mvhd"))
        media_scale = timescale(payload(b"trak", b"mdia", b"mdhd"))
        stts = payload(b"trak", b"mdia", b"minf", b"stbl", b"stts")
        entries = struct.unpack_from(f">{2 * struct.unpack_from('>I', stts, 4)[0]}I", stts, 8)
        counts, deltas = entries[::2], entries[1::2]
        # ffmpeg shortens the last packet's duration, so pad to whole frames.
        coded = sum(counts) * deltas[0]
        priming, valid, padding = 0, sum(n * d for n, d in zip(counts, deltas)), 0
        edts = payload(b"trak", b"edts")  # mutagen doesn't descend into edts
        if edts and edts[4:8] == b"elst":
            elst = edts[8 : struct.unpack_from(">I", edts)[0]]
            fmt = ">Qq4x" if elst[0] == 1 else ">Ii4x"
            for duration, media_time in struct.iter_unpack(fmt, elst[8:]):
                if media_time >= 0:  # skip empty edits
                    priming, valid = media_time, round(duration * media_scale / movie_scale)
                    padding = max(coded - priming - valid, 0)
                    break
    return priming, padding, valid
