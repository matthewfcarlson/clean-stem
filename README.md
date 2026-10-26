# clean-stem
Automatically cleans up profanity in audio using AI generated stems

See [VISION.md](VISION.md) for goals and architecture.

## Development

Requires [uv](https://docs.astral.sh/uv/) and ffmpeg.

```sh
uv sync
uv run clean-stem check path/to/song.mp3    # scan lyrics for profanity (tags, then online)
uv run clean-stem lyrics path/to/song.mp3   # compare lyrics from each source
uv run pytest                               # CLEAN_STEM_ONLINE=1 to include online lookups
```

### Sample tracks

Lyrics come from embedded tags first, then [LRCLIB](https://lrclib.net) (matched on artist, title, album and duration; often includes line timestamps), then [lyrics.ovh](https://lyrics.ovh).

`samples/manifest.tsv` lists the test tracks and what result each should give. The audio is copyrighted and is not committed to the repo. To copy it from your own library:

```sh
scripts/fetch_samples.sh /path/to/Music   # default: /Volumes/Media/Music
```
