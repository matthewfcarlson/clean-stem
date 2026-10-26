import numpy as np

from clean_stem.align import FlaggedWord, TimedWord, tokenize
from clean_stem.scan import confirm_on_vocals, locate_without_sync, unexplained_profanity, vocals_level_db
from clean_stem.separate import Stems

RATE = 1000


def heard(spec: str, offset: float = 0.0) -> list[TimedWord]:
    words = []
    for item in spec.split():
        text, start = item.rsplit("@", 1)
        words.append(TimedWord(text, float(start) + offset, float(start) + offset + 0.2))
    return words


def loud_stems(duration=10.0, quiet=()):
    """Stems whose vocals are loud everywhere except the `quiet` (start, end) ranges."""
    vocals = np.full((2, int(duration * RATE)), 0.5, np.float32)
    for start, end in quiet:
        vocals[:, int(start * RATE) : int(end * RATE)] = 0.0
    return [Stems(0.0, duration, vocals, np.zeros_like(vocals))]


def test_unexplained_finds_adlibs_and_extra_repeats():
    lyrics = tokenize("Like, ah\nBye bitch (Bye bitch)")
    transcript = heard("like@0 ah@1 let's@2 fucking@2.5 go@3 bye@4 bitch@4.5 bye@5 bitch@5.5 bitch@6")
    assert [(w.text, w.start) for w in unexplained_profanity(lyrics, transcript)] == [("fucking", 2.5), ("bitch", 6.0)]


def test_unexplained_ignores_one_for_one_mishearings():
    lyrics = tokenize("get the fuck out my face, these some whole other politics")
    transcript = heard("get@0 the@1 fuck@2 out@3 my@4 fags@5 these@6 some@7 hoe@8 other@9 politics@10")
    assert unexplained_profanity(lyrics, transcript) == []


def test_unexplained_without_lyrics_flags_everything():
    assert [w.text for w in unexplained_profanity([], heard("oh@0 shit@1"))] == ["shit"]


def test_vocals_level_db():
    stems = loud_stems(quiet=[(2.0, 3.0)])
    assert vocals_level_db(stems, 2.2, 2.8, RATE) < -100
    assert round(vocals_level_db(stems, 5.0, 6.0, RATE)) == -6


def test_confirm_on_vocals_requires_audible_nearby_profanity():
    stems = loud_stems(quiet=[(7.0, 8.0)])
    candidates = heard("fucking@2.5 shit@5 damn@7.4")
    responses = {
        2.5: "let's@1.0 fucking@1.1 go@1.6",  # confirmed (window starts at 1.0)
        5.0: "it@1.0 is@1.5",  # not heard on vocals -> dropped
        7.4: "damn@1.5",  # heard, but the vocals are silent there -> dropped
    }

    def vocals(start, end):
        return heard(responses[round(start + 1.5, 1)], offset=start)

    found = confirm_on_vocals(candidates, vocals, stems, RATE, 10.0)
    assert [(w.word, w.method, w.source) for w in found] == [("fucking", "heard", "vocals-scan")]
    assert found[0].start < 2.1 < found[0].end  # timed from the vocals, not the mix


def test_locate_without_sync_uses_plain_lyrics_and_adds_extras():
    stems = loud_stems()
    transcript = heard("bloodsucker@1 fame@1.5 fuh@2 like@4 a@4.3 goddamn@4.6 vampire@5 shit@8")
    words = locate_without_sync("Bloodsucker, fame fucker\nlike a goddamn vampire", transcript, stems, RATE, 10.0)
    assert [(w.word, w.method, w.source) for w in words] == [
        ("fucker", "inferred", "vocals"),
        ("goddamn", "heard", "vocals"),
        ("shit", "heard", "vocals-scan"),
    ]


def test_locate_without_sync_or_lyrics_drops_silent_hallucinations():
    stems = loud_stems(quiet=[(0.0, 3.0)])
    words = locate_without_sync(None, heard("fuck@1 oh@5 shit@6"), stems, RATE, 10.0)
    assert [w.word for w in words] == ["shit"]
