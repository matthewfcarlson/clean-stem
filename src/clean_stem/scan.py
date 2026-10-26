"""Whole-song transcription passes that catch what line-by-line alignment can't.

Two situations need these:

* Swearing the lyrics leave out (ad-libs, extra repeats). After the synced
  lines are aligned, the full mix is transcribed once; any profanity that the
  lyrics don't account for becomes a candidate, and is only accepted if it is
  also heard on the isolated vocals around that moment.
* Songs without synced lyrics. Whisper transcribes the song's isolated vocals
  end to end; flagged lyric words (if plain lyrics exist) are located by
  aligning against that transcript, and any other profanity heard is added.

Whisper invents words over silence, so words whose vocals are near-silent are
discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np

from clean_stem.align import FlaggedWord, Listener, TimedWord, match_words, normalize, pad_span, tokenize
from clean_stem.profanity import is_profane

# A candidate is re-checked on the vocals this far either side of it.
CONFIRM_PAD = 1.5
# Vocals quieter than this (dBFS RMS) count as silence.
SILENCE_DB = -40.0
# Without lyrics, the full-mix pass decides whether separation is needed:
# if it hears no profanity and is at least this confident, the song is clean.
CONFIDENT_PROBABILITY = 0.6


@dataclass(frozen=True)
class Transcript:
    words: list[TimedWord]
    mean_probability: float


def unexplained_profanity(lyric_tokens: list[str], heard: list[TimedWord]) -> list[TimedWord]:
    """Profane words in the transcript that the lyrics don't account for.

    A heard word that lines up one-for-one with a different lyric word is a
    mishearing ("some whole other" heard as "some hoe other") and is ignored.
    Words heard where the lyrics have nothing, or more words than the lyrics
    have, are ad-libs and repeats.
    """
    sm = SequenceMatcher(None, [normalize(t) for t in lyric_tokens], [normalize(w.text) for w in heard], autojunk=False)
    result = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            continue
        result.extend(w for w in heard[j1:j2] if is_profane(w.text))
    return result


def overlaps(start: float, end: float, words: list[FlaggedWord], slack: float = 0.3) -> bool:
    return any(start < w.end + slack and end > w.start - slack for w in words)


def vocals_level_db(stems: list, start: float, end: float, sample_rate: int) -> float:
    """RMS level of the isolated vocals over [start, end], in dBFS."""
    covering = [s for s in stems if s.start <= start and end <= s.end]
    if not covering:
        return 0.0  # no stems to judge by; treat as audible
    s = covering[0]
    lo = int((start - s.start) * sample_rate)
    hi = max(lo + 1, int((end - s.start) * sample_rate))
    rms = float(np.sqrt(np.mean(np.square(s.vocals[:, lo:hi], dtype=np.float64))))
    return 20 * np.log10(rms + 1e-12)


def confirm_on_vocals(
    candidates: list[TimedWord],
    vocals: Listener,
    stems: list,
    sample_rate: int,
    duration: float,
) -> list[FlaggedWord]:
    """Keep candidates that are also heard, audibly, on the isolated vocals."""
    found: list[FlaggedWord] = []
    for cand in sorted(candidates, key=lambda w: w.start):
        if overlaps(cand.start, cand.end, found):
            continue
        start = max(0.0, cand.start - CONFIRM_PAD)
        end = min(duration, cand.end + CONFIRM_PAD)
        for w in vocals(start, end):
            if not is_profane(w.text) or overlaps(w.start, w.end, found):
                continue
            if w.end < cand.start - 0.5 or w.start > cand.end + 0.5:
                continue
            if vocals_level_db(stems, w.start, w.end, sample_rate) < SILENCE_DB:
                continue
            found.append(FlaggedWord(w.text, -1, *pad_span(w.start, w.end, duration), "heard", "vocals-scan"))
    return found


def locate_without_sync(
    lyrics_text: str | None,
    vocals_transcript: list[TimedWord],
    stems: list,
    sample_rate: int,
    duration: float,
) -> list[FlaggedWord]:
    """Locate profanity using a full-song vocals transcript (no line timings)."""
    heard = [w for w in vocals_transcript if vocals_level_db(stems, w.start, w.end, sample_rate) >= SILENCE_DB]
    found: list[FlaggedWord] = []

    if lyrics_text:
        tokens = tokenize(lyrics_text)
        flagged = [i for i, t in enumerate(tokens) if is_profane(t)]
        for i, (start, end, method) in sorted(match_words(tokens, heard, flagged).items()):
            found.append(FlaggedWord(tokens[i], -1, *pad_span(start, end, duration), method, "vocals"))
        extras = unexplained_profanity(tokens, heard)
    else:
        extras = [w for w in heard if is_profane(w.text)]

    for w in extras:
        if not overlaps(w.start, w.end, found):
            found.append(FlaggedWord(w.text, -1, *pad_span(w.start, w.end, duration), "heard", "vocals-scan"))
    return sorted(found, key=lambda w: w.start)
