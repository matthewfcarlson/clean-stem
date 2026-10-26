"""Find when each flagged word is sung, using line-synced lyrics and Whisper.

Synced lyrics tell us roughly when each line starts. For every line that
contains profanity we transcribe a short window around it with word
timestamps, then line the transcript up against the known lyrics. Each
window is tried with a list of "listeners" in order (normally isolated
vocals, then the full mix) until every flagged word in it is heard. Words
Whisper hears correctly get exact timings ("heard"); words it mishears are
placed between the neighbouring words it did hear ("inferred"). If Whisper
can't make out the line at all, the word's position is estimated from where
it falls in the line's text ("estimated").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable, Iterable, Sequence

import numpy as np

from clean_stem.audio import WHISPER_SAMPLE_RATE, resample
from clean_stem.lookup import SyncedLine
from clean_stem.profanity import is_profane

DEFAULT_MODEL = "small"
# Seconds of audio to include before a line starts and after the next begins;
# LRC timestamps are often slightly early or late.
WINDOW_PAD = 1.0
# Extra margin either side of a word whose position is only estimated.
ESTIMATE_PAD = 0.5
# Whisper word timestamps are tight and clip consonants at either end, so
# widen located words. Tuned by muting sample tracks and listening for leaks.
WORD_PAD_BEFORE = 0.15
WORD_PAD_AFTER = 0.25
MIN_WORD_LENGTH = 0.5

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’]*")


@dataclass(frozen=True)
class TimedWord:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class FlaggedWord:
    word: str
    line: int  # index into the synced lines
    start: float
    end: float
    method: str  # "heard", "inferred" or "estimated"
    source: str  # which listener located it, e.g. "vocals" or "mix"


# Transcribes the song between two times (seconds); returned word times are
# absolute, not relative to the window.
Listener = Callable[[float, float], list[TimedWord]]

_METHOD_RANK = {"heard": 0, "inferred": 1}


def normalize(word: str) -> str:
    return word.lower().replace("’", "'").strip("'")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


class Transcriber:
    """Thin wrapper so tests can substitute a fake."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")

    def words(self, audio: np.ndarray) -> list[TimedWord]:
        return self.transcribe(audio)[0]

    def transcribe(self, audio: np.ndarray) -> tuple[list[TimedWord], float]:
        """Words with window-relative times, plus Whisper's mean word probability."""
        segments, _ = self.model.transcribe(
            audio,
            language="en",
            word_timestamps=True,
            condition_on_previous_text=False,
        )
        result, probabilities = [], []
        for segment in segments:
            for w in segment.words or []:
                probabilities.append(w.probability)
                # Whisper attaches punctuation; keep only the word-ish part.
                for token in tokenize(w.word):
                    result.append(TimedWord(token, w.start, w.end))
        return result, (sum(probabilities) / len(probabilities) if probabilities else 0.0)


def mix_listener(audio: np.ndarray, transcriber: Transcriber) -> Listener:
    """Listen to the full mix (mono, WHISPER_SAMPLE_RATE)."""

    def listen(start: float, end: float) -> list[TimedWord]:
        window = audio[int(start * WHISPER_SAMPLE_RATE) : int(end * WHISPER_SAMPLE_RATE)]
        return [TimedWord(w.text, w.start + start, w.end + start) for w in transcriber.words(window)]

    return listen


def vocals_listener(stereo: np.ndarray, sample_rate: int, separator, transcriber: Transcriber) -> Listener:
    """Listen to Demucs-isolated vocals. `separator` keeps the stems for splicing."""

    def listen(start: float, end: float) -> list[TimedWord]:
        stems = separator.separate(stereo, sample_rate, start, end)
        vocals = resample(stems.vocals.mean(axis=0), sample_rate, WHISPER_SAMPLE_RATE)
        return [TimedWord(w.text, w.start + start, w.end + start) for w in transcriber.words(vocals)]

    return listen


def match_words(lyric_tokens: list[str], heard: list[TimedWord], wanted: Iterable[int]) -> dict[int, tuple[float, float, str]]:
    """Map lyric token indices to (start, end, method) using the transcript."""
    sm = SequenceMatcher(
        None, [normalize(t) for t in lyric_tokens], [normalize(w.text) for w in heard], autojunk=False
    )
    timings: dict[int, tuple[float, float, str]] = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        for i in range(i1, i2):
            if tag == "equal":
                w = heard[j1 + (i - i1)]
                timings[i] = (w.start, w.end, "heard")
            elif j2 > j1:
                # Replaced: the lyric words i1..i2 were heard as something else
                # over heard[j1..j2]. Assume they line up proportionally, and
                # take the heard words covering this one's share (rounding
                # outwards; censoring a little extra beats missing the word).
                n, m = i2 - i1, j2 - j1
                k = i - i1
                first = j1 + (k * m) // n
                last = j1 + max(-(-(k + 1) * m // n) - 1, (k * m) // n)
                timings[i] = (heard[first].start, heard[last].end, "inferred")
            else:
                # Not heard at all: it sits between the neighbouring words.
                before = heard[j1 - 1].end if j1 > 0 else None
                after = heard[j1].start if j1 < len(heard) else None
                if before is not None and after is not None:
                    timings[i] = (before, after, "inferred")
    return {i: timings[i] for i in wanted if i in timings}


def pad_span(start: float, end: float, duration: float) -> tuple[float, float]:
    start, end = start - WORD_PAD_BEFORE, end + WORD_PAD_AFTER
    if end - start < MIN_WORD_LENGTH:
        middle = (start + end) / 2
        start, end = middle - MIN_WORD_LENGTH / 2, middle + MIN_WORD_LENGTH / 2
    return max(0.0, start), min(duration, end)


def estimate_timing(text: str, token_index: int, line_start: float, line_end: float) -> tuple[float, float]:
    """Guess a word's timing by assuming the line is sung at an even pace."""
    spans = [m.span() for m in _TOKEN_RE.finditer(text)]
    total = spans[-1][1]
    word_start, word_end = spans[token_index]
    length = line_end - line_start
    return (
        max(line_start, line_start + length * word_start / total - ESTIMATE_PAD),
        min(line_end, line_start + length * word_end / total + ESTIMATE_PAD),
    )


def align_flagged_words(
    lines: list[SyncedLine],
    duration: float,
    listeners: Sequence[tuple[str, Listener]],
) -> list[FlaggedWord]:
    """Locate every profane word in the synced lyrics.

    `listeners` are (name, listener) pairs tried in order for each window;
    later ones only run if an earlier one left some flagged word unheard.
    """
    found: list[FlaggedWord] = []

    for idx, line in enumerate(lines):
        tokens = tokenize(line.text)
        flagged = [i for i, t in enumerate(tokens) if is_profane(t)]
        if not flagged:
            continue

        line_end = lines[idx + 1].time if idx + 1 < len(lines) else duration
        start = max(0.0, line.time - WINDOW_PAD)
        end = min(duration, line_end + WINDOW_PAD)

        # The window spills into the neighbouring lines, so align against them
        # too; otherwise their words get matched to this line's.
        prev_tokens = tokenize(lines[idx - 1].text) if idx > 0 else []
        next_tokens = tokenize(lines[idx + 1].text) if idx + 1 < len(lines) else []
        offset = len(prev_tokens)
        lyric_tokens = prev_tokens + tokens + next_tokens

        best: dict[int, tuple[float, float, str, str]] = {}
        for name, listen in listeners:
            timings = match_words(lyric_tokens, listen(start, end), [offset + i for i in flagged])
            for i in flagged:
                timing = timings.get(offset + i)
                if timing is None:
                    continue
                if i not in best or _METHOD_RANK[timing[2]] < _METHOD_RANK[best[i][2]]:
                    best[i] = (*timing, name)
            if all(i in best and best[i][2] == "heard" for i in flagged):
                break

        for i in flagged:
            if i in best:
                word_start, word_end, method, name = best[i]
                found.append(FlaggedWord(tokens[i], idx, *pad_span(word_start, word_end, duration), method, name))
            else:
                est_start, est_end = estimate_timing(line.text, i, line.time, line_end)
                found.append(FlaggedWord(tokens[i], idx, est_start, est_end, "estimated", ""))

    return found


def enrich_synced(lines: list[SyncedLine], other_texts: Iterable[str]) -> list[SyncedLine]:
    """Add profanity that other lyric sources have but the synced lyrics lack.

    Synced lyrics often drop ad-libs ("(Let's fucking go)") that a site's
    plain lyrics or the file's embedded lyrics include. Each other text is
    aligned against the synced lyrics word by word; profane words it has
    extra are inserted into the synced line next to where they appear.
    """
    lines = list(lines)
    for other in other_texts:
        line_tokens = [tokenize(line.text) for line in lines]
        flat = [(li, ti) for li, tokens in enumerate(line_tokens) for ti in range(len(tokens))]
        if not flat:
            break
        ours = [normalize(line_tokens[li][ti]) for li, ti in flat]
        theirs = tokenize(other)
        sm = SequenceMatcher(None, ours, [normalize(t) for t in theirs], autojunk=False)
        insertions: dict[int, list[tuple[int, str]]] = {}  # line -> [(token position, word)]
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal" or tag == "delete":
                continue
            extra = [t for t in theirs[j1:j2] if is_profane(t)]
            if not extra or (tag == "replace" and any(is_profane(ours[i]) for i in range(i1, i2))):
                continue
            # Attach after the last of our words before the gap (or before the
            # first word, at the very start).
            li, ti = flat[i2 - 1] if i2 > 0 else (flat[0][0], -1)
            insertions.setdefault(li, []).extend((ti + 1, t) for t in extra)
        for li, inserts in insertions.items():
            tokens = list(line_tokens[li])
            for pos, word in sorted(inserts, key=lambda x: x[0], reverse=True):
                tokens.insert(pos, word)
            lines[li] = SyncedLine(lines[li].time, " ".join(tokens))
    return lines
