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
    "pussy",
    "bastard",
    "cunt",
    "whore",
    "slut",
    "fag",
    "faggot",
)
WHOLE_WORD_SUFFIXES = ("", "s", "es", "ed", "in", "ing", "in'")

# Bump when the list changes, so batch runs re-check files.
# 2: dropped "ass", "dick", "hoe" as too tame / too often misheard ("whole" -> "hoe").
WORD_LIST_VERSION = "2"

# Asterisks are part of words so censored lyrics ("f**k", "sh*t") stay whole.
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’*]*")
_CENSORED_RE = re.compile(r"^[a-z]+\*+[a-z*]*$")
_WHOLE = {w + s for w in WHOLE_WORDS for s in WHOLE_WORD_SUFFIXES}


@dataclass(frozen=True)
class Match:
    word: str  # as written in the lyrics
    line: int  # 1-based line number
    column: int  # 0-based character offset within the line


def is_profane(word: str) -> bool:
    normalized = word.lower().replace("’", "'").strip("'")
    if _CENSORED_RE.match(normalized):
        # Some lyrics sites censor words themselves; they only do that to swears.
        return True
    if normalized in _WHOLE or normalized.rstrip("'") in _WHOLE:
        return True
    return any(root in normalized for root in ROOTS)


def find_profanity(text: str) -> list[Match]:
    matches = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for m in _WORD_RE.finditer(line):
            if is_profane(m.group()):
                matches.append(Match(word=m.group(), line=line_no, column=m.start()))
    return matches
