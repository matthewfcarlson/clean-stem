from clean_stem import cli, pipeline


def test_clean_retries_failed_lookups_at_the_end(tmp_path, monkeypatch, capsys):
    songs = [tmp_path / "a.mp3", tmp_path / "b.mp3"]
    for song in songs:
        song.write_bytes(b"")
    calls = []

    def fake_clean(path, models, lyrics_mode):
        calls.append(path.name)
        if path.name == "a.mp3" and calls.count("a.mp3") == 1:
            return {"file": str(path), "status": "error", "reason": "lyrics lookup failed, will retry: HTTP 503"}
        return {"file": str(path), "status": "clean", "lyrics_source": "lrclib:1"}

    monkeypatch.setattr(pipeline, "clean", fake_clean)
    monkeypatch.setattr(cli, "RETRY_PASS_DELAY", 0)
    assert cli.main(["clean", "--no-state", *map(str, songs)]) == 0
    assert calls == ["a.mp3", "b.mp3", "a.mp3"]
    out = capsys.readouterr().out
    assert "retrying 1 files" in out and out.strip().endswith("summary: clean 2")
