"""The end-to-end cleaning pipeline for a single file."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from clean_stem import align, sidecar
from clean_stem.lookup import (
    INSTRUMENTAL_SUFFIX,
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

SUPPORTED_SUFFIXES = (".mp3", ".flac", ".m4a", ".ogg", ".opus")

# How censored spans are filled: "reversed" (instrumental plus the word's
# vocal reversed) or "sung" (a clean word sung in the artist's voice where the
# swear is sung, "reversed" elsewhere; needs Seed-VC).
FILLS = ("reversed", "sung")


@dataclass
class Models:
    """Shared across files: Whisper and Demucs (loaded on first use) and the
    lyrics providers (cached in `state` when given)."""

    whisper_model: str = align.DEFAULT_MODEL
    state: StateDB | None = None
    fill: str = "reversed"
    seedvc_dir: Path | None = None

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

    @cached_property
    def sung(self):
        from clean_stem.sung import SeedVCError, SungFill, default_seedvc_dir

        seedvc_dir = self.seedvc_dir or default_seedvc_dir()
        if seedvc_dir is None:
            raise SeedVCError("the sung fill needs a Seed-VC checkout: pass --seedvc or set $CLEAN_STEM_SEEDVC")
        return SungFill(seedvc_dir)


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
        words = align.fit_spans(words, stems, separator.sample_rate, track.duration)
        return Located(track, fetched.source, sorted(words, key=lambda w: w.start), synced, stems)

    # No line timings. Use plain lyrics if we can find them.
    lyrics_text = lyrics_source = None
    instrumental = False  # a lyrics site says so; not trusted on its own
    if lyrics_mode != "none":
        if (embedded := read_embedded_lyrics(path)) is not None:
            lyrics_text, lyrics_source = embedded.text, embedded.source
        elif (plain := lookup_lyrics(track, (models.lrclib, models.ovh))) is not None:
            if plain.source.endswith(INSTRUMENTAL_SUFFIX):
                instrumental, lyrics_source = True, plain.source
            else:
                lyrics_text, lyrics_source = plain.text, plain.source
    if lyrics_text is not None and not find_profanity(lyrics_text):
        return Located(track, lyrics_source, [])
    audio = _AudioSetup(path, models)
    vocals, separator, stems = audio.vocals, audio.separator, audio.stems
    if lyrics_text is None:
        # Cheap check before separating the whole song: the song is clean if
        # the full-mix transcript has no profanity and is confident, or is
        # too sparse to be anything but an instrumental, or the lyrics site
        # says it's instrumental (they're sometimes wrong, hence the listen).
        heard, confidence = models.transcriber.transcribe(audio.mix_audio)
        if not any(is_profane(w.text) for w in heard):
            if instrumental:
                return Located(track, lyrics_source, [])
            if confidence >= scan.CONFIDENT_PROBABILITY:
                return Located(track, "transcript", [])
            if len(heard) < scan.INSTRUMENTAL_WORDS_PER_MINUTE * track.duration / 60:
                return Located(track, "transcript (instrumental)", [])

    vocals_transcript = vocals(0.0, track.duration)
    words = scan.locate_without_sync(lyrics_text, vocals_transcript, stems, separator.sample_rate, track.duration)
    words = align.fit_spans(words, stems, separator.sample_rate, track.duration)
    return Located(track, lyrics_source or "transcript", words, [], stems)


@dataclass(frozen=True)
class SungPlan:
    """Whether a located word gets the sung fill, before any conversion."""

    substitute: str | None  # the word to sing, or None to keep the default fill
    reason: str | None  # why not, when substitute is None
    max_lead: float = 0.0  # how early (s) the new word may start


def plan_sung(words: list[align.FlaggedWord]) -> list[SungPlan]:
    from clean_stem.splice import CROSSFADE
    from clean_stem.sung import substitute

    plans = []
    for i, w in enumerate(words):
        sub = substitute(w.word)
        # Spans this close get merged when spliced, and lose their sung vocal.
        crowded = any(j != i and v.start - 2 * CROSSFADE <= w.end and w.start <= v.end + 2 * CROSSFADE for j, v in enumerate(words))
        if sub is None:
            plans.append(SungPlan(None, "no substitute"))
        elif crowded:
            plans.append(SungPlan(None, "next to another word"))
        else:
            # It may start a little early, but not so early it merges with a neighbour.
            before = max((v.end for v in words if v.end <= w.start), default=0.0)
            plans.append(SungPlan(sub, None, w.start - before - 2 * CROSSFADE - 0.01))
    return plans


def sing_spans(located: Located, models: Models, sample_rate: int) -> tuple[list, list[str]]:
    """Spans for the located words, with sung vocals where the sung fill
    applies, and each word's fill ("sung:<word>" or "reversed (<why>)")."""
    from clean_stem.splice import Span, _covering_stems
    from clean_stem.sung import SAMPLE_RATE, SeedVCError, build_reference

    words = located.words
    spans = [Span(w.start, w.end) for w in words]
    if sample_rate != SAMPLE_RATE:
        return spans, [f"reversed (stems at {sample_rate} Hz)"] * len(words)

    ref_id = None
    fills = []
    for i, (w, plan) in enumerate(zip(words, plan_sung(words))):
        if plan.substitute is None:
            fills.append(f"reversed ({plan.reason})")
            continue
        if ref_id is None:
            reference = build_reference(located.stems, [(v.start, v.end) for v in words], sample_rate)
            ref_id = models.sung.set_reference(reference)
        try:
            result = models.sung.convert(
                ref_id, _covering_stems(spans[i], located.stems), w.start, w.end, plan.substitute, sample_rate, plan.max_lead
            )
        except SeedVCError as e:
            fills.append(f"reversed (conversion failed: {e})")
            continue
        if isinstance(result, str):
            fills.append(f"reversed ({result})")
            continue
        spans[i] = Span(min(w.start, result.word_start), w.end, sung=result)
        fills.append(f"sung:{plan.substitute}")
    return spans, fills


# How far a cleaned file's length may drift from the original's (seconds).
MAX_LENGTH_DRIFT = 0.1


def check_encoded(encoded: Path, seconds: float, sample_rate: int) -> None:
    """Refuse to replace the original with a file that doesn't decode to the
    original's length (`seconds`, as decoded at `sample_rate`)."""
    import subprocess

    from clean_stem.audio import decode_mono

    try:
        decoded = len(decode_mono(encoded, sample_rate)) / sample_rate
    except subprocess.CalledProcessError as e:
        raise sidecar.SidecarError(f"cleaned file doesn't decode: {e.stderr.decode(errors='replace').strip()[:200]}") from e
    if abs(decoded - seconds) > MAX_LENGTH_DRIFT:
        raise sidecar.SidecarError(f"cleaned file decodes to {decoded:.2f}s, the original {seconds:.2f}s")


def back_up(path: Path, root: Path) -> Path:
    """Copy `path` under `root` (mirroring its absolute path) and check the copy."""
    dest = root / path.resolve().relative_to(path.resolve().anchor)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.copy2(path, tmp)
    if sidecar.sha256(tmp) != sidecar.sha256(path):
        tmp.unlink()
        raise sidecar.SidecarError(f"backup of {path} doesn't match it")
    os.replace(tmp, dest)
    return dest


def clean(path: Path, models: Models, lyrics_mode: str = "auto", backup: Path | None = None) -> dict:
    """Censor one file in place, writing a .dirtystem sidecar (and, with
    `backup`, a copy of the original under that directory). Returns a result dict."""
    from clean_stem.audio import decode_stereo
    from clean_stem.splice import OPUS_SAMPLE_RATE, REVERSED_VOCAL_GAIN, Span, encode_like, replace_spans

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
    if models.fill == "sung":
        spans, fills = sing_spans(located, models, models.separator.sample_rate)
        fill = {"style": "sung", "fallback": "instrumental+reversed-vocal", "reversed_vocal_gain": REVERSED_VOCAL_GAIN}
    else:
        spans = [Span(w.start, w.end) for w in located.words]
        fills = ["reversed"] * len(spans)
        fill = {"style": "instrumental+reversed-vocal", "reversed_vocal_gain": REVERSED_VOCAL_GAIN}
    cleaned = replace_spans(audio, sample_rate, spans, located.stems, models.separator.sample_rate)

    tmp = path.with_name(f".{path.stem}.cleaning{path.suffix}")
    try:
        encode_like(path, cleaned, sample_rate, tmp)
        details = {
            "word_list_version": WORD_LIST_VERSION,
            "lyrics_source": located.lyrics_source,
            "whisper_model": models.whisper_model,
            "fill": fill,
            "replacements": [
                {"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3), "method": w.method, "source": w.source, "fill": f}
                for w, f in zip(located.words, fills)
            ],
        }
        shutil.copymode(path, tmp)
        check_encoded(tmp, audio.shape[1] / sample_rate, sample_rate)
        if backup is not None:
            back_up(path, backup)
        sidecar.write_sidecar(path, tmp, details)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    if sidecar.sha256(path) != sidecar.read_manifest(path)["modified_sha256"]:
        raise sidecar.SidecarError(f"{path} doesn't match its sidecar after replacing; restore with `clean-stem reverse --force`")

    return {**result, "status": "cleaned", "lyrics_source": located.lyrics_source, "replacements": details["replacements"]}
