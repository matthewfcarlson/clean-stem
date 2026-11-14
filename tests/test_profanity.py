import pytest

from clean_stem.profanity import find_profanity, is_profane, profane_tokens


@pytest.mark.parametrize(
    "word",
    ["fuck", "Fucking", "motherfucker", "bullshit", "bitches", "goddamn", "damn", "asshole", "dick", "Dicks", "dickhead", "dick's", "dick'll", "fuckin'", "fuckin’",
     "f**k", "F*cking", "sh*t", "b***h", "n****s"],
)
def test_flags_profanity(word):
    assert is_profane(word)


@pytest.mark.parametrize(
    "word",
    ["hell", "class", "assume", "Dickens", "cocktail", "pissed", "Scunthorpe", "hoedown", "shoe", "ass", "Dickinson", "dickie", "hoes", "cocked"],
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


def test_lyrics_only_words_need_lyrics():
    assert is_profane("dick") and not is_profane("dick", heard=True) and not is_profane("Dickhead", heard=True)
    assert is_profane("shit", heard=True)


@pytest.mark.parametrize(
    "line, flagged",
    [
        ("she throws those glances at every tom, Tom, Dick and Joe", []),
        ("You can ask any Tom, Dick, or Stanley", []),
        ("Dick Turpin wore a mask, too", []),
        ("Grandpa Dick was always sick", []),
        ("Kept his dick wet", ["dick"]),
        ("Dick move, I'm a dick", ["Dick", "dick"]),
        ("SUCK MY DICK", ["DICK"]),
        ("Because my dick's bigger", ["dick's"]),
    ],
)
def test_dick_as_a_name_is_not_flagged(line, flagged):
    assert [m.word for m in find_profanity(line)] == flagged
    tokens = line.replace(",", "").split()
    assert [tokens[i] for i in profane_tokens(tokens)] == flagged
