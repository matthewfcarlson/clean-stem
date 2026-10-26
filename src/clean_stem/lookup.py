"""Look up lyrics for a track from online lyrics providers."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import mutagen

USER_AGENT = "clean-stem/0.1 (https://github.com/matthewfcarlson/clean-stem)"
TIMEOUT = 15
# How far a candidate's duration may drift from the file's before we reject it.
MAX_DURATION_DELTA = 4.0
# Waits (seconds) before retrying a rate-limited or failing request.
RETRY_DELAYS = (2, 8, 30)

_TITLE_NOISE_RE = re.compile(r"\s*[\(\[](explicit|clean|remaster(ed)?[^)\]]*)[\)\]]\s*", re.I)
_SECTION_HEADER_RE = re.compile(r"^\s*\[[^\]]*\]\s*$")
_LRC_LINE_RE = re.compile(r"^\[(\d+):(\d+(?:\.\d+)?)\]\s?(.*)$")


@dataclass(frozen=True)
class TrackInfo:
    artist: str
    title: str
    album: str | None
    duration: float


@dataclass(frozen=True)
class SyncedLine:
    time: float  # seconds from the start of the track
    text: str


@dataclass(frozen=True)
class FetchedLyrics:
    text: str
    source: str  # e.g. "lrclib:2599853"
    synced: list[SyncedLine] = field(default_factory=list)
    duration: float | None = None


def read_track_info(path: Path) -> TrackInfo | None:
    audio = mutagen.File(path, easy=True)
    if audio is None or audio.tags is None:
        return None
    artist = (audio.tags.get("artist") or [None])[0]
    title = (audio.tags.get("title") or [None])[0]
    if not artist or not title:
        return None
    album = (audio.tags.get("album") or [None])[0]
    return TrackInfo(artist=artist, title=clean_title(title), album=album, duration=audio.info.length)


def clean_title(title: str) -> str:
    return _TITLE_NOISE_RE.sub(" ", title).strip()


def strip_section_headers(text: str) -> str:
    """Drop lines like "[Chorus]" or "[Eminem]" that some providers include."""
    lines = [line for line in text.splitlines() if not _SECTION_HEADER_RE.match(line)]
    return "\n".join(lines).strip()


def parse_lrc(lrc: str) -> list[SyncedLine]:
    synced = []
    for line in lrc.splitlines():
        m = _LRC_LINE_RE.match(line.strip())
        if m:
            minutes, seconds, text = m.groups()
            synced.append(SyncedLine(time=int(minutes) * 60 + float(seconds), text=text))
    return synced


def lyrics_similarity(a: str, b: str) -> float:
    """Rough 0-1 similarity between two lyric texts, ignoring case and punctuation."""

    def words(text: str) -> list[str]:
        return re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))

    return SequenceMatcher(None, words(a), words(b), autojunk=False).ratio()


class LookupFailed(Exception):
    """A provider couldn't be reached. Unlike "not found", worth retrying later."""


def _get_json(url: str, retry_delays=RETRY_DELAYS):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(len(retry_delays) + 1):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code != 429 and e.code < 500:
                raise LookupFailed(f"HTTP {e.code} from {url}") from e
            error: Exception = e
            retry_after = e.headers.get("Retry-After") if e.headers else None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            error, retry_after = e, None
        if attempt < len(retry_delays):
            delay = retry_delays[attempt]
            if retry_after and retry_after.isdigit():
                delay = max(delay, min(int(retry_after), 300))
            time.sleep(delay)
    raise LookupFailed(f"{error} ({url})") from error


class LrclibProvider:
    """https://lrclib.net - open, keyless, often has line-synced lyrics."""

    name = "lrclib"
    base_url = "https://lrclib.net/api"

    def fetch(self, track: TrackInfo) -> FetchedLyrics | None:
        params = {"artist_name": track.artist, "track_name": track.title}
        candidates = _get_json(f"{self.base_url}/search?{urllib.parse.urlencode(params)}") or []
        best = pick_lrclib_candidate(candidates, track)
        if best is None:
            return None
        return FetchedLyrics(
            text=best["plainLyrics"],
            source=f"{self.name}:{best['id']}",
            synced=parse_lrc(best.get("syncedLyrics") or ""),
            duration=best.get("duration"),
        )


def pick_lrclib_candidate(candidates: list[dict], track: TrackInfo) -> dict | None:
    """Choose the search result that best matches the file.

    Candidates must have lyrics and a duration close to the file's. Among
    those, prefer a matching album, then synced lyrics, then the closest
    duration.
    """
    album = (track.album or "").lower()

    def score(c: dict) -> tuple:
        return (
            bool(album) and (c.get("albumName") or "").lower().startswith(album),
            bool(c.get("syncedLyrics")),
            -abs(c["duration"] - track.duration),
        )

    viable = [
        c
        for c in candidates
        if not c.get("instrumental")
        and c.get("plainLyrics")
        and c.get("duration") is not None
        and abs(c["duration"] - track.duration) <= MAX_DURATION_DELTA
    ]
    return max(viable, key=score, default=None)


class LyricsOvhProvider:
    """https://lyrics.ovh - keyless, plain text only, no duration to match on."""

    name = "lyrics.ovh"
    base_url = "https://api.lyrics.ovh/v1"

    def fetch(self, track: TrackInfo) -> FetchedLyrics | None:
        artist = urllib.parse.quote(track.artist, safe="")
        title = urllib.parse.quote(track.title, safe="")
        data = _get_json(f"{self.base_url}/{artist}/{title}")
        text = strip_section_headers((data or {}).get("lyrics") or "")
        return FetchedLyrics(text=text, source=self.name) if text else None


DEFAULT_PROVIDERS = (LrclibProvider(), LyricsOvhProvider())


def lookup_lyrics(track: TrackInfo, providers=DEFAULT_PROVIDERS) -> FetchedLyrics | None:
    """Return lyrics from the first provider that has them.

    Returns None only if every provider answered "not found"; if any of them
    failed and none had lyrics, raises LookupFailed so the caller can retry.
    """
    failure: LookupFailed | None = None
    for provider in providers:
        try:
            result = provider.fetch(track)
        except LookupFailed as e:
            failure = e
            continue
        if result is not None:
            return result
    if failure is not None:
        raise failure
    return None


class CachedProvider:
    """Wraps a provider with a persistent cache (see state.StateDB)."""

    def __init__(self, provider, cache):
        self.provider = provider
        self.cache = cache
        self.name = provider.name

    def fetch(self, track: TrackInfo) -> FetchedLyrics | None:
        hit, value = self.cache.get_lyrics(self.name, track)
        if hit:
            return value
        value = self.provider.fetch(track)  # LookupFailed propagates uncached
        self.cache.put_lyrics(self.name, track, value)
        return value


def lyrics_to_json(lyrics: FetchedLyrics | None) -> str | None:
    if lyrics is None:
        return None
    return json.dumps(
        {
            "text": lyrics.text,
            "source": lyrics.source,
            "synced": [[line.time, line.text] for line in lyrics.synced],
            "duration": lyrics.duration,
        }
    )


def lyrics_from_json(data: str | None) -> FetchedLyrics | None:
    if data is None:
        return None
    d = json.loads(data)
    return FetchedLyrics(d["text"], d["source"], [SyncedLine(t, text) for t, text in d["synced"]], d["duration"])
