import pytest

from clean_stem.profanity import find_profanity, is_profane


@pytest.mark.parametrize(
    "word",
    ["fuck", "Fucking", "motherfucker", "bullshit", "bitches", "goddamn", "damn", "ass", "fuckin'", "fuckin’",
     "f**k", "F*cking", "sh*t", "b***h", "n****s"],
)
def test_flags_profanity(word):
    assert is_profane(word)


@pytest.mark.parametrize(
    "word",
    ["hell", "class", "assume", "Dickens", "cocktail", "pissed", "Scunthorpe", "hoedown", "shoe"],
)
def test_ignores_innocent_words(word):
    assert not is_profane(word)


def test_find_profanity_reports_positions():
    text = "first line is fine\nBloodsucker, fame fucker\nlike a goddamn vampire"
    matches = find_profanity(text)
    assert [(m.word, m.line, m.column) for m in matches] == [
        ("fucker", 2, 18),
        ("goddamn", 3, 7),
    ]


def test_clean_text_has_no_matches():
    assert find_profanity("Lacy, oh, Lacy\nAren't you the sweetest thing on this side of Hell?") == []


def test_censored_lyrics_are_found_whole():
    assert [m.word for m in find_profanity("Get the f**k out, this sh*t is real")] == ["f**k", "sh*t"]
    assert find_profanity("5 * 3 = fifteen") == []
