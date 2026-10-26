"""Read lyrics embedded in audio file tags."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

import mutagen
from mutagen.id3 import ID3
from mutagen.mp4 import MP4

# Vorbis comment keys (FLAC, OGG, Opus) that commonly hold unsynced lyrics.
VORBIS_LYRICS_KEYS = ("lyrics", "unsyncedlyrics", "unsynced lyrics")


@dataclass(frozen=True)
class Lyrics:
    text: str
    source: str  # e.g. "tag:USLT::XXX"


def read_embedded_lyrics(path: Path) -> Lyrics | None:
    """Return the lyrics embedded in the file's tags, or None if there are none."""
    audio = mutagen.File(path)
    if audio is None or audio.tags is None:
        return None

    tags = audio.tags
    found: tuple[str, str] | None = None

    if isinstance(tags, ID3):
        for key, frame in tags.items():
            if key.startswith("USLT") and str(frame).strip():
                found = (key, str(frame))
                break
    elif isinstance(audio, MP4):
        values = tags.get("\xa9lyr")
        if values:
            found = ("\xa9lyr", values[0])
    else:
        for key in VORBIS_LYRICS_KEYS:
            values = tags.get(key)
            if values:
                found = (key.upper(), values[0])
                break

    if found is None:
        return None
    key, text = found
    text = html.unescape(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    return Lyrics(text=text, source=f"tag:{key}") if text else None
