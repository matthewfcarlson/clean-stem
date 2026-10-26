from clean_stem.lookup import (
    TrackInfo,
    clean_title,
    lyrics_similarity,
    parse_lrc,
    pick_lrclib_candidate,
    strip_section_headers,
)

TRACK = TrackInfo(artist="Kendrick Lamar", title="squabble up", album="GNX", duration=158.04)


def candidate(id, duration, album="GNX", synced=True, plain="lyrics", instrumental=False):
    return {
        "id": id,
        "duration": duration,
        "albumName": album,
        "plainLyrics": plain,
        "syncedLyrics": "[00:01.00] lyrics" if synced else None,
        "instrumental": instrumental,
    }


def test_picks_matching_album_within_duration():
    candidates = [
        candidate(1, 158.0, album="Top Songs Of 2025"),
        candidate(2, 290.0),  # right album, wrong length (extended mix)
        candidate(3, 157.0, album="GNX (explicit)"),
    ]
    assert pick_lrclib_candidate(candidates, TRACK)["id"] == 3


def test_prefers_synced_then_closest_duration():
    candidates = [candidate(1, 158.0, synced=False), candidate(2, 155.0), candidate(3, 157.5)]
    assert pick_lrclib_candidate(candidates, TRACK)["id"] == 3


def test_rejects_instrumental_missing_lyrics_and_unknown_duration():
    candidates = [
        candidate(1, 158.0, instrumental=True),
        candidate(2, 158.0, plain=None),
        candidate(3, None),
        candidate(4, 170.0),
    ]
    assert pick_lrclib_candidate(candidates, TRACK) is None


def test_clean_title_strips_explicit_marker():
    assert clean_title("vampire (Explicit)") == "vampire"
    assert clean_title("Song [Remastered 2011]") == "Song"
    assert clean_title("all-american bitch") == "all-american bitch"


def test_parse_lrc():
    lrc = "[00:14.45] I hate to give the satisfaction\n[01:02.10] Bloodsucker\nnot a timed line"
    lines = parse_lrc(lrc)
    assert [(round(l.time, 2), l.text) for l in lines] == [
        (14.45, "I hate to give the satisfaction"),
        (62.1, "Bloodsucker"),
    ]


def test_strip_section_headers():
    assert strip_section_headers("[Eminem]\nGuess who's back\n\n[Chorus]\nBack again") == "Guess who's back\n\nBack again"


def test_lyrics_similarity_ignores_case_and_punctuation():
    assert lyrics_similarity("Hate to give, the SATISFACTION", "hate to give the satisfaction") == 1.0
    assert lyrics_similarity("completely different words", "hate to give the satisfaction") < 0.2
