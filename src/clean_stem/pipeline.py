"""The end-to-end cleaning pipeline for a single file."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from clean_stem import align, sidecar
from clean_stem.lookup import (
    CachedProvider,
    LookupFailed,
    LrclibProvider,
    LyricsOvhProvider,
    TrackInfo,
    lookup_lyrics,
    read_track_info,
)
from clean_stem.lyrics import read_embedded_lyrics
from clean_stem.profanity import WORD_LIST_VERSION, find_profanity, is_profane
from clean_stem.state import StateDB

SUPPORTED_SUFFIXES = (".mp3", ".flac", ".ogg", ".opus")


@dataclass
class Models:
    """Shared across files: Whisper and Demucs (loaded on first use) and the
    lyrics providers (cached in `state` when given)."""

    whisper_model: str = align.DEFAULT_MODEL
    state: StateDB | None = None

    def __post_init__(self):
        wrap = (lambda p: CachedProvider(p, self.state)) if self.state else (lambda p: p)
        self.lrclib = wrap(LrclibProvider())
        self.ovh = wrap(LyricsOvhProvider())

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
    lyrics_source: str | None
    words: list[align.FlaggedWord]
    synced: list = field(default_factory=list)  # lookup.SyncedLine, if used
    stems: list = field(default_factory=list)  # separate.Stems for each window


# How `locate` may use lyrics: "auto" (synced if available, else plain),
# "plain" (ignore line timings) or "none" (transcription only). The last two
# exist mainly to exercise the fallbacks on songs that do have synced lyrics.
LYRICS_MODES = ("auto", "plain", "none")


class _AudioSetup:
    """Decoded audio and listeners for one file; built only once it's needed."""

    def __init__(self, path: Path, models: Models):
        from clean_stem.audio import decode_mono, decode_stereo

        self.mix_audio = decode_mono(path)
        self.mix = align.mix_listener(self.mix_audio, models.transcriber)
        self.separator = models.separator
        self.stems: list = []
        self.separator.stems = self.stems  # collect this file's windows only
        self.stereo = decode_stereo(path, self.separator.sample_rate)
        self.vocals = align.vocals_listener(self.stereo, self.separator.sample_rate, self.separator, models.transcriber)


def locate(path: Path, models: Models, lyrics_mode: str = "auto") -> Located | str:
    """Find flagged words and their timings. Returns a skip reason on failure."""
    from clean_stem import scan

    track = read_track_info(path)
    if track is None:
        return "missing artist/title tags"

    fetched = models.lrclib.fetch(track) if lyrics_mode == "auto" else None
    synced = fetched.synced if fetched else []
    if synced:
        # Other sources sometimes include ad-libs the synced lyrics drop.
        others = [lyr.text for lyr in (read_embedded_lyrics(path), lookup_lyrics(track, (models.ovh,))) if lyr]
        synced = align.enrich_synced(synced, others)
        if not any(find_profanity(line.text) for line in synced):
            # Clean lyrics: bail before any audio work. (Unlisted ad-libs in
            # otherwise clean songs aren't worth a full transcription each.)
            return Located(track, fetched.source, [], synced)

    if synced:
        audio = _AudioSetup(path, models)
        mix, vocals, separator, stems, stereo = audio.mix, audio.vocals, audio.separator, audio.stems, audio.stereo
        separator.prefetch(stereo, separator.sample_rate, align.flagged_windows(synced, track.duration))
        words = align.align_flagged_words(synced, track.duration, [("vocals", vocals), ("mix", mix)])
        # Catch profanity the lyrics leave out: scan the full mix, then confirm
        # anything unexplained on the vocals.
        lyric_tokens = align.tokenize("\n".join(line.text for line in synced))
        candidates = [
            w
            for w in scan.unexplained_profanity(lyric_tokens, mix(0.0, track.duration))
            if not scan.overlaps(w.start, w.end, words)
        ]
        words += scan.confirm_on_vocals(candidates, vocals, stems, separator.sample_rate, track.duration)
        return Located(track, fetched.source, sorted(words, key=lambda w: w.start), synced, stems)

    # No line timings. Use plain lyrics if we can find them.
    lyrics_text = lyrics_source = None
    if lyrics_mode != "none":
        if (embedded := read_embedded_lyrics(path)) is not None:
            lyrics_text, lyrics_source = embedded.text, embedded.source
        elif (plain := lookup_lyrics(track, (models.lrclib, models.ovh))) is not None:
            lyrics_text, lyrics_source = plain.text, plain.source
    if lyrics_text is not None and not find_profanity(lyrics_text):
        return Located(track, lyrics_source, [])
    audio = _AudioSetup(path, models)
    vocals, separator, stems = audio.vocals, audio.separator, audio.stems
    if lyrics_text is None:
        # Cheap check before separating the whole song: a confident full-mix
        # transcript with no profanity means the song is clean.
        heard, confidence = models.transcriber.transcribe(audio.mix_audio)
        if confidence >= scan.CONFIDENT_PROBABILITY and not any(is_profane(w.text) for w in heard):
            return Located(track, "transcript", [])

    vocals_transcript = vocals(0.0, track.duration)
    words = scan.locate_without_sync(lyrics_text, vocals_transcript, stems, separator.sample_rate, track.duration)
    return Located(track, lyrics_source or "transcript", words, [], stems)


def clean(path: Path, models: Models, lyrics_mode: str = "auto") -> dict:
    """Censor one file in place, writing a .dirtystem sidecar. Returns a result dict."""
    from clean_stem.audio import decode_stereo
    from clean_stem.splice import OPUS_SAMPLE_RATE, Span, encode_like, replace_spans

    result = {"file": str(path)}
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        return {**result, "status": "skipped", "reason": f"unsupported format {path.suffix}"}
    if sidecar.sidecar_path(path).exists():
        return {**result, "status": "skipped", "reason": "already processed"}

    # Bail early on clean songs before loading any models. Embedded lyrics
    # count if present; otherwise the synced lyrics decide.
    embedded = read_embedded_lyrics(path) if lyrics_mode != "none" else None
    if embedded is not None and not find_profanity(embedded.text):
        return {**result, "status": "clean", "lyrics_source": embedded.source}

    try:
        located = locate(path, models, lyrics_mode)
    except LookupFailed as e:
        return {**result, "status": "error", "reason": f"lyrics lookup failed, will retry: {e}"}
    if isinstance(located, str):
        return {**result, "status": "skipped", "reason": located}
    if not located.words:
        return {**result, "status": "clean", "lyrics_source": located.lyrics_source}

    # Splice at the file's own sample rate.
    import mutagen

    info = mutagen.File(path).info
    sample_rate, channels = getattr(info, "sample_rate", None), info.channels
    if sample_rate is None:  # Opus: mutagen has no rate; it always decodes at 48 kHz
        sample_rate = OPUS_SAMPLE_RATE
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
            "lyrics_source": located.lyrics_source,
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

    return {**result, "status": "cleaned", "lyrics_source": located.lyrics_source, "replacements": details["replacements"]}
