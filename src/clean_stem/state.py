"""Persistent state: a lyrics cache and a record of every file scanned.

One SQLite file holds both, so a periodic run over a big library only does
work for files that are new or changed, and doesn't re-query lyrics sites.
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from clean_stem.lookup import FetchedLyrics, TrackInfo, lyrics_from_json, lyrics_to_json

# "Not found" answers are re-asked after this long; sites add lyrics over time.
NOT_FOUND_TTL = 30 * 24 * 3600

DEFAULT_PATH = Path(os.environ.get("CLEAN_STEM_STATE", Path.home() / ".cache" / "clean-stem" / "state.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS lyrics_cache (
    provider   TEXT NOT NULL,
    track_key  TEXT NOT NULL,
    fetched_at REAL NOT NULL,
    lyrics     TEXT,            -- JSON, or NULL for "not found"
    PRIMARY KEY (provider, track_key)
);
CREATE TABLE IF NOT EXISTS files (
    path              TEXT PRIMARY KEY,
    size              INTEGER NOT NULL,
    mtime_ns          INTEGER NOT NULL,
    status            TEXT NOT NULL,   -- clean, cleaned, skipped, error
    reason            TEXT,
    word_list_version TEXT NOT NULL,
    processed_at      REAL NOT NULL
);
"""


@dataclass(frozen=True)
class FileRecord:
    size: int
    mtime_ns: int
    status: str
    reason: str | None
    word_list_version: str


def track_key(track: TrackInfo) -> str:
    # Durations drift slightly between rips; whole seconds are close enough.
    return "\x1f".join([track.artist.lower(), track.title.lower(), (track.album or "").lower(), str(round(track.duration))])


class StateDB:
    def __init__(self, path: Path | None = None):
        path = path or DEFAULT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # Lyrics cache

    def get_lyrics(self, provider: str, track: TrackInfo) -> tuple[bool, FetchedLyrics | None]:
        """(hit, lyrics). A cached "not found" is a hit with None until it expires."""
        row = self.conn.execute(
            "SELECT fetched_at, lyrics FROM lyrics_cache WHERE provider = ? AND track_key = ?",
            (provider, track_key(track)),
        ).fetchone()
        if row is None:
            return False, None
        fetched_at, data = row
        if data is None and time.time() - fetched_at > NOT_FOUND_TTL:
            return False, None
        return True, lyrics_from_json(data)

    def put_lyrics(self, provider: str, track: TrackInfo, lyrics: FetchedLyrics | None) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO lyrics_cache VALUES (?, ?, ?, ?)",
                (provider, track_key(track), time.time(), lyrics_to_json(lyrics)),
            )

    # Scanned files

    def get_file(self, path: Path) -> FileRecord | None:
        row = self.conn.execute(
            "SELECT size, mtime_ns, status, reason, word_list_version FROM files WHERE path = ?", (str(path),)
        ).fetchone()
        return FileRecord(*row) if row else None

    def put_file(self, path: Path, status: str, reason: str | None, word_list_version: str) -> None:
        st = path.stat()
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(path), st.st_size, st.st_mtime_ns, status, reason, word_list_version, time.time()),
            )
