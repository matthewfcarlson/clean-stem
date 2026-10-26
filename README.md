# clean-stem
Automatically cleans up profanity in audio using AI generated stems

See [VISION.md](VISION.md) for goals and architecture.

## Development

Requires [uv](https://docs.astral.sh/uv/) and ffmpeg.

```sh
uv sync
uv run clean-stem check path/to/song.mp3    # scan lyrics for profanity (tags, then online)
uv run clean-stem lyrics path/to/song.mp3   # compare lyrics from each source
uv run clean-stem align path/to/song.mp3    # timestamp each flagged word (Whisper)
uv run clean-stem clean path/to/Music       # censor in place (files or folders), writes song.mp3.dirtystem
uv run clean-stem reverse path/to/song.mp3  # restore the original from the sidecar
uv run pytest                               # CLEAN_STEM_ONLINE=1 to include online lookups
```

### Sample tracks

Lyrics come from embedded tags first, then [LRCLIB](https://lrclib.net) (matched on artist, title, album and duration; often includes line timestamps), then [lyrics.ovh](https://lyrics.ovh).

`align` uses LRCLIB's line timestamps to pick a few seconds around each flagged line, isolates the vocals in that window with Demucs, transcribes them with faster-whisper, and lines the transcript up against the known lyrics. If a word isn't clearly heard on the vocals, the full mix is tried too. Each word is reported as `heard` (Whisper heard it), `inferred` (placed between the neighbouring words it did hear) or `estimated` (Whisper couldn't make out the line; position guessed from the line timing).

Lyrics gaps are covered three ways:

- **Ad-libs missing from the synced lyrics**: profanity that the embedded lyrics or lyrics.ovh have, but LRCLIB's synced lines lack, is merged into the nearest synced line before alignment.
- **Profanity in no lyrics at all**: the full mix is transcribed once; profanity the lyrics don't account for (excluding one-for-one mishearings like "whole" → "hoe") is re-checked on the isolated vocals around that moment and kept only if heard there.
- **No synced lyrics**: the whole song's vocals are separated and transcribed, then aligned against plain lyrics if any exist. With no lyrics at all, a quick full-mix pass first decides whether the song is clean enough to skip.

Words heard where the isolated vocals are near-silent are discarded, since Whisper invents words over silence. `--lyrics plain|none` forces the fallbacks, for testing.

`clean` replaces each flagged word with the Demucs instrumental for that moment (short crossfades either side), re-encodes in the original format and bitrate, copies all tags across, and swaps the file in place. Before swapping, it writes `song.mp3.dirtystem`: a zip with a `manifest.json` (what was replaced and why, plus hashes) and the original file's exact bytes, so `reverse` restores it bit-for-bit. Files that already have a sidecar are skipped.

`clean` keeps a SQLite state file (`~/.cache/clean-stem/state.db`, or `--state` / `$CLEAN_STEM_STATE`) with every file's size, mtime and outcome, plus cached lyrics lookups. Re-running over a library only touches new or changed files; lookups that failed (site down, rate-limited) are retried next run, and files restored with `reverse` are left alone.

`samples/manifest.tsv` lists the test tracks and what result each should give. The audio is copyrighted and is not committed to the repo. To copy it from your own library:

```sh
scripts/fetch_samples.sh /path/to/Music   # default: /Volumes/Media/Music
```
