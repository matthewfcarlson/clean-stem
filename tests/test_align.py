from clean_stem.align import TimedWord, align_flagged_words, estimate_timing, match_words, pad_span, tokenize
from clean_stem.lookup import SyncedLine


def heard(spec: str, offset: float = 0.0) -> list[TimedWord]:
    """Build a transcript from "word@start word@start ..."; each word lasts 0.2s."""
    words = []
    for item in spec.split():
        text, start = item.rsplit("@", 1)
        words.append(TimedWord(text, float(start) + offset, float(start) + offset + 0.2))
    return words


class FakeListener:
    """Returns canned transcripts (times relative to the window) in call order."""

    def __init__(self, *transcripts: str):
        self.transcripts = list(transcripts)
        self.calls = []

    def __call__(self, start, end):
        self.calls.append((start, end))
        return heard(self.transcripts.pop(0), offset=start)


def test_match_words_heard_exactly():
    tokens = tokenize("Bloodsucker, fame fucker")
    timings = match_words(tokens, heard("Bloodsucker@1.0 fame@1.5 fucker@2.0"), [2])
    assert timings == {2: (2.0, 2.2, "heard")}


def test_match_words_infers_misheard_word_from_neighbours():
    tokens = tokenize("A yee nigga couldn't try me")
    timings = match_words(tokens, heard("I@0.0 eat@0.3 milk@0.6 couldn't@1.0 try@1.3 me@1.5"), [2])
    start, end, method = timings[2]
    assert method == "inferred"
    assert (start, end) == (0.6, 0.8)  # third misheard word for the third lyric word


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
    listener = FakeListener("wanted@0.2 Bloodsucker@1.0 fame@1.5 fucker@2.0 Bleeding@3.0")
    words = align_flagged_words(lines, 20.0, [("vocals", listener)])
    # Only the profane line is transcribed: from 1s before it to 1s after the next line starts.
    assert listener.calls == [(11.0, 15.0)]
    assert [(w.word, w.line, w.method, w.source) for w in words] == [("fucker", 1, "heard", "vocals")]
    # Heard at 13.0-13.2, then padded out to the minimum length.
    assert (round(words[0].start, 2), round(words[0].end, 2)) == (12.85, 13.45)


def test_pad_span_widens_and_clamps():
    assert tuple(round(x, 2) for x in pad_span(10.0, 10.0, 60.0)) == (9.8, 10.3)
    assert tuple(round(x, 2) for x in pad_span(10.0, 11.0, 60.0)) == (9.85, 11.25)
    assert pad_span(0.05, 0.1, 0.2) == (0.0, 0.2)


def test_align_flagged_words_falls_back_to_estimate():
    lines = [SyncedLine(0.0, "You bald headed fag, blow me"), SyncedLine(4.0, "next")]
    words = align_flagged_words(lines, 5.0, [("vocals", FakeListener(""))])
    assert len(words) == 1 and words[0].method == "estimated"
    assert 0.0 < words[0].start < words[0].end < 4.0


def test_later_listeners_only_run_when_needed_and_best_result_wins():
    lines = [SyncedLine(0.0, "fame fucker, goddamn vampire"), SyncedLine(4.0, "next")]
    vocals = FakeListener("fame@0.5 fucker@1.0 goddam@1.5 vampire@2.0")  # "goddamn" misheard
    mix = FakeListener("fame@0.5 fuh@1.0 goddamn@1.6 vampire@2.0")  # "fucker" misheard
    words = align_flagged_words(lines, 5.0, [("vocals", vocals), ("mix", mix)])
    assert [(w.word, w.method, w.source) for w in words] == [
        ("fucker", "heard", "vocals"),
        ("goddamn", "heard", "mix"),
    ]

    vocals = FakeListener("fame@0.5 fucker@1.0 goddamn@1.5 vampire@2.0")
    mix = FakeListener()
    align_flagged_words(lines, 5.0, [("vocals", vocals), ("mix", mix)])
    assert mix.calls == []  # everything was heard on the vocals


def test_enrich_synced_adds_adlibs_from_other_sources():
    from clean_stem.align import enrich_synced

    lines = [SyncedLine(120.0, 'Like, "Ah"'), SyncedLine(124.0, 'Like, "Ah"'), SyncedLine(128.0, "All the time")]
    embedded = 'Like, "Ah"\nLike, "Ah" (Let\'s fucking go)\n\nAll the time'
    enriched = enrich_synced(lines, [embedded])
    assert [line.text for line in enriched] == ['Like, "Ah"', "Like Ah fucking", "All the time"]
    assert [line.time for line in enriched] == [120.0, 124.0, 128.0]


def test_enrich_synced_ignores_clean_differences_and_existing_profanity():
    from clean_stem.align import enrich_synced

    lines = [SyncedLine(1.0, "Bloodsucker, fame fucker"), SyncedLine(3.0, "I hate to give the satisfaction")]
    other = "Bloodsucker, fame-fucker\nHate to give the satisfaction, oh"
    assert enrich_synced(lines, [other]) == lines


def test_match_words_infers_within_long_misheard_stretch_proportionally():
    tokens = tokenize("one two three goddamn five six seven eight")
    # Everything misheard: eight lyric words over eight heard words, 1s apart.
    transcript = heard(" ".join(f"x{i}@{i}" for i in range(8)))
    start, end, method = match_words(tokens, transcript, [3])[3]
    assert method == "inferred"
    assert (start, end) == (3.0, 3.2)  # not the whole 0-7s stretch

    # Fewer heard words than lyric words: spans round outwards.
    start, end, _ = match_words(tokens, heard("x0@0 x1@1 x2@2 x3@3"), [3])[3]
    assert (start, end) == (1.0, 1.2)
