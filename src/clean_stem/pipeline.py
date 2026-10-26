"""The end-to-end cleaning pipeline for a single file."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from clean_stem import align, sidecar
from clean_stem.lookup import FetchedLyrics, LrclibProvider, TrackInfo, read_track_info
from clean_stem.lyrics import read_embedded_lyrics
from clean_stem.profanity import WORD_LIST_VERSION, find_profanity

SUPPORTED_SUFFIXES = (".mp3", ".flac")


@dataclass
class Models:
    """Loads Whisper and Demucs on first use, then reuses them across files."""

    whisper_model: str = align.DEFAULT_MODEL
    separate: bool = True

    @cached_property
    def transcriber(self) -> align.Transcriber:
        return align.Transcriber(self.whisper_model)

    @cached_property
    def separator(self):
        from clean_stem.separate import Separator

        return Separator()


@dataclass
class Located:
    track: TrackInfo
    lyrics: FetchedLyrics
    words: list[align.FlaggedWord]
    stems: list = field(default_factory=list)  # separate.Stems for each window


def locate(path: Path, models: Models) -> Located | str:
    """Find flagged words and their timings. Returns a skip reason on failure."""
    from clean_stem.audio import decode_mono, decode_stereo

    track = read_track_info(path)
    if track is None:
        return "missing artist/title tags"
    fetched = LrclibProvider().fetch(track)
    if fetched is None or not fetched.synced:
        return "no synced lyrics"

    listeners = [("mix", align.mix_listener(decode_mono(path), models.transcriber))]
    stems = []
    if models.separate:
        separator = models.separator
        separator.stems = stems  # collect this file's windows only
        stereo = decode_stereo(path, separator.sample_rate)
        listeners.insert(
            0, ("vocals", align.vocals_listener(stereo, separator.sample_rate, separator, models.transcriber))
        )
    words = align.align_flagged_words(fetched.synced, track.duration, listeners)
    return Located(track, fetched, words, stems)


def clean(path: Path, models: Models) -> dict:
    """Censor one file in place, writing a .dirtystem sidecar. Returns a result dict."""
    from clean_stem.audio import decode_stereo
    from clean_stem.splice import Span, encode_like, replace_spans

    result = {"file": str(path)}
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        return {**result, "status": "skipped", "reason": f"unsupported format {path.suffix}"}
    if sidecar.sidecar_path(path).exists():
        return {**result, "status": "skipped", "reason": "already processed"}

    # Bail early on clean songs before loading any models. Embedded lyrics
    # count if present; otherwise the synced lyrics decide.
    embedded = read_embedded_lyrics(path)
    if embedded is not None and not find_profanity(embedded.text):
        return {**result, "status": "clean", "lyrics_source": embedded.source}

    if not models.separate:
        raise ValueError("cleaning needs Demucs stems; separation can't be disabled")
    located = locate(path, models)
    if isinstance(located, str):
        return {**result, "status": "skipped", "reason": located}
    if not located.words:
        return {**result, "status": "clean", "lyrics_source": located.lyrics.source}

    # Splice at the file's own sample rate.
    import mutagen

    info = mutagen.File(path).info
    sample_rate, channels = info.sample_rate, info.channels
    audio = decode_stereo(path, sample_rate)
    if channels == 1:
        audio = audio[:1]
    spans = [Span(w.start, w.end) for w in located.words]
    cleaned = replace_spans(audio, sample_rate, spans, located.stems, models.separator.sample_rate)

    tmp = path.with_name(f".{path.stem}.cleaning{path.suffix}")
    try:
        encode_like(path, cleaned, sample_rate, tmp)
        details = {
            "word_list_version": WORD_LIST_VERSION,
            "lyrics_source": located.lyrics.source,
            "whisper_model": models.whisper_model,
            "replacements": [
                {"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3), "method": w.method, "source": w.source}
                for w in located.words
            ],
        }
        shutil.copymode(path, tmp)
        sidecar.write_sidecar(path, tmp, details)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()

    return {**result, "status": "cleaned", "lyrics_source": located.lyrics.source, "replacements": details["replacements"]}
