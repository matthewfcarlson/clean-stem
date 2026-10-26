import io
import os
import time
import urllib.error

import pytest

from clean_stem import lookup, state as state_mod
from clean_stem.batch import is_unchanged, iter_audio_files
from clean_stem.lookup import CachedProvider, FetchedLyrics, LookupFailed, SyncedLine, TrackInfo, lookup_lyrics
from clean_stem.profanity import WORD_LIST_VERSION
from clean_stem.state import StateDB

TRACK = TrackInfo("Olivia Rodrigo", "vampire", "GUTS", 219.7)
LYRICS = FetchedLyrics("Bloodsucker, fame fucker", "lrclib:1", [SyncedLine(70.2, "Bloodsucker, fame fucker")], 220.0)


@pytest.fixture
def db(tmp_path):
    return StateDB(tmp_path / "state.db")


class FakeProvider:
    name = "fake"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def fetch(self, track):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_lyrics_cache_round_trip(db):
    assert db.get_lyrics("lrclib", TRACK) == (False, None)
    db.put_lyrics("lrclib", TRACK, LYRICS)
    assert db.get_lyrics("lrclib", TRACK) == (True, LYRICS)
    # Same song, slightly different rip length: still a hit.
    assert db.get_lyrics("lrclib", TrackInfo("olivia rodrigo", "Vampire", "GUTS", 219.9))[0]


def test_not_found_is_cached_until_it_expires(db, monkeypatch):
    db.put_lyrics("lrclib", TRACK, None)
    assert db.get_lyrics("lrclib", TRACK) == (True, None)
    later = time.time() + state_mod.NOT_FOUND_TTL + 1
    monkeypatch.setattr(state_mod.time, "time", lambda: later)
    assert db.get_lyrics("lrclib", TRACK) == (False, None)


def test_cached_provider_caches_answers_but_not_failures(db):
    provider = FakeProvider(LookupFailed("503"), LYRICS)
    cached = CachedProvider(provider, db)
    with pytest.raises(LookupFailed):
        cached.fetch(TRACK)
    assert cached.fetch(TRACK) == LYRICS
    assert cached.fetch(TRACK) == LYRICS
    assert provider.calls == 2


def test_lookup_lyrics_distinguishes_failure_from_not_found():
    assert lookup_lyrics(TRACK, (FakeProvider(None), FakeProvider(None))) is None
    assert lookup_lyrics(TRACK, (FakeProvider(LookupFailed("down")), FakeProvider(LYRICS))) == LYRICS
    with pytest.raises(LookupFailed):
        lookup_lyrics(TRACK, (FakeProvider(LookupFailed("down")), FakeProvider(None)))


def test_get_json_retries_rate_limits_then_gives_up(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req.full_url)
        if len(calls) < 3:
            raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, None)
        return io.BytesIO(b'{"ok": true}')

    monkeypatch.setattr(lookup.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(lookup.time, "sleep", lambda s: None)
    assert lookup._get_json("https://example.test/x") == {"ok": True}
    assert len(calls) == 3

    def always_down(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, None)

    monkeypatch.setattr(lookup.urllib.request, "urlopen", always_down)
    with pytest.raises(LookupFailed):
        lookup._get_json("https://example.test/x", retry_delays=(0, 0))


def test_iter_audio_files_recurses_and_skips_temp_files(tmp_path):
    (tmp_path / "Album").mkdir()
    for name in ["Album/01.mp3", "Album/02.FLAC", "Album/cover.jpg", "Album/.01.cleaning.mp3", "Album/._02.flac", "song.mp3.dirtystem"]:
        (tmp_path / name).write_bytes(b"x")
    found = [p.relative_to(tmp_path).as_posix() for p in iter_audio_files([tmp_path])]
    assert found == ["Album/01.mp3", "Album/02.FLAC"]


def test_is_unchanged(db, tmp_path):
    song = tmp_path / "song.mp3"
    song.write_bytes(b"audio")
    assert not is_unchanged(db, song)

    db.put_file(song, "clean", None, WORD_LIST_VERSION)
    assert is_unchanged(db, song)

    db.put_file(song, "error", "lyrics lookup failed", WORD_LIST_VERSION)
    assert not is_unchanged(db, song)  # errors are retried

    db.put_file(song, "clean", None, WORD_LIST_VERSION + "-old")
    assert not is_unchanged(db, song)  # word list changed since

    db.put_file(song, "clean", None, WORD_LIST_VERSION)
    song.write_bytes(b"new audio")
    os.utime(song, ns=(1, 1))
    assert not is_unchanged(db, song)  # file changed
