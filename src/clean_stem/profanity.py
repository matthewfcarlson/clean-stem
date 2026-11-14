"""Find profanity in lyrics using a hardcoded default word list.

The target is "words a 2-year-old would repeat": explicit swears and slurs,
not euphemisms, innuendo, or mild words like "hell".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Roots that are flagged anywhere inside a word ("motherfucking", "bullshit").
ROOTS = (
    "fuck",
    "shit",
    "bitch",
    "nigga",
    "nigger",
)

# Words flagged only as whole words, optionally with a common suffix, because
# their letters show up inside innocent words ("class", "cocktail", "Scunthorpe").
WHOLE_WORDS = (
    "asshole",
    "damn",
    "goddamn",
    "cock",
    "dick",
    "dickhead",
    "pussy",
    "bastard",
    "cunt",
    "whore",
    "slut",
    "fag",
    "faggot",
)
WHOLE_WORD_SUFFIXES = ("", "s", "es", "ed", "in", "ing", "in'")
# Suffixed forms that are everyday words ("cocked your head to the side").
INNOCENT = {"cocked"}
# Flagged only where the lyrics have them, never from a transcript alone: Whisper
# hears them in innocent words too often.
LYRICS_ONLY_WORDS = ("dick", "dickhead")

# Bump when the list changes, so batch runs re-check files.
# 2: dropped "ass", "dick", "hoe" as too tame / too often misheard ("whole" -> "hoe").
# 3: "cocked" is innocent.
# 4: "dick" is back, from lyrics only, and not as a name ("Tom, Dick and Harry").
WORD_LIST_VERSION = "4"

# Asterisks are part of words so censored lyrics ("f**k", "sh*t") stay whole.
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’*]*")
_CENSORED_RE = re.compile(r"^[a-z]+\*+[a-z*]*$")
_WHOLE = {w + s for w in WHOLE_WORDS for s in WHOLE_WORD_SUFFIXES} - INNOCENT
_LYRICS_ONLY = {w + s for w in LYRICS_ONLY_WORDS for s in WHOLE_WORD_SUFFIXES}
# Contractions on a swear ("dick's bigger", "dick'll").
_CONTRACTION_RE = re.compile(r"'(?:s|ll|d|re|ve)$")


@dataclass(frozen=True)
class Match:
    word: str  # as written in the lyrics
    line: int  # 1-based line number
    column: int  # 0-based character offset within the line


def is_profane(word: str, heard: bool = False) -> bool:
    """Whether `word` is a swear. `heard`: it comes from a transcript, not lyrics."""
    normalized = word.lower().replace("’", "'").strip("'")
    if _CENSORED_RE.match(normalized):
        # Some lyrics sites censor words themselves; they only do that to swears.
        return True
    for form in (normalized, normalized.rstrip("'"), _CONTRACTION_RE.sub("", normalized)):
        if form in _WHOLE:
            return not (heard and form in _LYRICS_ONLY)
    return any(root in normalized for root in ROOTS)


def _titlecase(word: str) -> bool:
    return word[:1].isupper() and word[1:2].islower() and word != "I"


def profane_tokens(tokens: list[str]) -> list[int]:
    """Indices of the swears among consecutive lyrics words. A capitalized
    lyrics-only word next to another capitalized word is a name ("Tom, Dick",
    "Dick Turpin")."""
    flagged = []
    for i, t in enumerate(tokens):
        if not is_profane(t):
            continue
        base = _CONTRACTION_RE.sub("", t.lower().replace("’", "'"))
        if base in _LYRICS_ONLY and _titlecase(t):
            if (i > 0 and _titlecase(tokens[i - 1])) or (i + 1 < len(tokens) and _titlecase(tokens[i + 1])):
                continue
        flagged.append(i)
    return flagged


def find_profanity(text: str) -> list[Match]:
    matches = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        words = list(_WORD_RE.finditer(line))
        for i in profane_tokens([m.group() for m in words]):
            matches.append(Match(word=words[i].group(), line=line_no, column=words[i].start()))
    return matches
