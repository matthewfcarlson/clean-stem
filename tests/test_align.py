import numpy as np

from clean_stem.align import TimedWord, align_flagged_words, estimate_timing, match_words, pad_span, tokenize
from clean_stem.lookup import SyncedLine


def heard(spec: str, offset: float = 0.0) -> list[TimedWord]:
    """Build a transcript from "word@start word@start ..."; each word lasts 0.2s."""
    words = []
    for item in spec.split():
        text, start = item.rsplit("@", 1)
        words.append(TimedWord(text, float(start) + offset, float(start) + offset + 0.2))
    return words


class FakeTranscriber:
    """Returns canned transcripts (times relative to the window) in call order."""

    def __init__(self, *transcripts: str):
        self.transcripts = list(transcripts)

    def words(self, audio):
        return heard(self.transcripts.pop(0))


def test_match_words_heard_exactly():
    tokens = tokenize("Bloodsucker, fame fucker")
    timings = match_words(tokens, heard("Bloodsucker@1.0 fame@1.5 fucker@2.0"), [2])
    assert timings == {2: (2.0, 2.2, "heard")}


def test_match_words_infers_misheard_word_from_neighbours():
    tokens = tokenize("A yee nigga couldn't try me")
    timings = match_words(tokens, heard("I@0.0 eat@0.3 milk@0.6 couldn't@1.0 try@1.3 me@1.5"), [2])
    start, end, method = timings[2]
    assert method == "inferred"
    assert (start, end) == (0.0, 0.8)  # the whole misheard span before "couldn't"


def test_match_words_infers_unheard_word_between_neighbours():
    tokens = tokenize("like a goddamn vampire")
    timings = match_words(tokens, heard("like@0.0 a@0.3 vampire@1.2"), [2])
    assert timings == {2: (0.5, 1.2, "inferred")}


def test_estimate_timing_assumes_even_pace():
    start, end = estimate_timing("aaaa bbbb", 1, 10.0, 19.0)
    assert (round(start, 2), round(end, 2)) == (14.5, 19.0)


def test_align_flagged_words_uses_window_offsets_and_skips_clean_lines():
    lines = [
        SyncedLine(10.0, "Just what you wanted"),
        SyncedLine(12.0, "Bloodsucker, fame fucker"),
        SyncedLine(14.0, "Bleeding me dry"),
    ]
    # Only the profane line is transcribed; its window starts at 12.0 - 1.0 = 11.0.
    transcriber = FakeTranscriber("wanted@0.2 Bloodsucker@1.0 fame@1.5 fucker@2.0 Bleeding@3.0")
    words = align_flagged_words(np.zeros(16000 * 20, dtype=np.float32), lines, transcriber, 20.0)
    assert [(w.word, w.line, w.method) for w in words] == [("fucker", 1, "heard")]
    # Heard at 13.0-13.2, then padded out to the minimum length.
    assert (round(words[0].start, 2), round(words[0].end, 2)) == (12.85, 13.45)


def test_pad_span_widens_and_clamps():
    assert tuple(round(x, 2) for x in pad_span(10.0, 10.0, 60.0)) == (9.8, 10.3)
    assert tuple(round(x, 2) for x in pad_span(10.0, 11.0, 60.0)) == (9.85, 11.25)
    assert pad_span(0.05, 0.1, 0.2) == (0.0, 0.2)


def test_align_flagged_words_falls_back_to_estimate():
    lines = [SyncedLine(0.0, "You bald headed fag, blow me"), SyncedLine(4.0, "next")]
    words = align_flagged_words(np.zeros(16000 * 5, dtype=np.float32), lines, FakeTranscriber(""), 5.0)
    assert len(words) == 1 and words[0].method == "estimated"
    assert 0.0 < words[0].start < words[0].end < 4.0
