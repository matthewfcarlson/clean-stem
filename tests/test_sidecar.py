import os

import pytest

from clean_stem.sidecar import ReverseError, read_manifest, reverse, sidecar_path, write_sidecar


@pytest.fixture
def cleaned(tmp_path):
    """A file that has been "cleaned": sidecar written, modified bytes in place."""
    song = tmp_path / "song.mp3"
    song.write_bytes(b"original audio")
    os.chmod(song, 0o640)
    modified = tmp_path / "tmp.mp3"
    modified.write_bytes(b"cleaned audio")
    write_sidecar(song, modified, {"replacements": [{"word": "x", "start": 1.0, "end": 1.5}]})
    os.replace(modified, song)
    return song


def test_sidecar_records_manifest(cleaned):
    manifest = read_manifest(cleaned)
    assert manifest["file"] == "song.mp3"
    assert manifest["replacements"][0]["word"] == "x"
    assert manifest["original_sha256"] != manifest["modified_sha256"]


def test_reverse_restores_exact_bytes_and_mode(cleaned):
    os.chmod(cleaned, 0o640)
    assert reverse(cleaned) == "restored"
    assert cleaned.read_bytes() == b"original audio"
    assert oct(cleaned.stat().st_mode & 0o777) == oct(0o640)
    assert not sidecar_path(cleaned).exists()


def test_reverse_refuses_if_file_changed_unless_forced(cleaned):
    cleaned.write_bytes(b"edited after cleaning")
    with pytest.raises(ReverseError, match="changed"):
        reverse(cleaned)
    assert reverse(cleaned, force=True) == "restored"
    assert cleaned.read_bytes() == b"original audio"


def test_reverse_when_cleaning_was_interrupted(tmp_path):
    song = tmp_path / "song.mp3"
    song.write_bytes(b"original audio")
    other = tmp_path / "tmp.mp3"
    other.write_bytes(b"cleaned audio")
    write_sidecar(song, other, {})  # crashed before replacing the file
    assert reverse(song) == "already original; removed sidecar"
    assert song.read_bytes() == b"original audio"
