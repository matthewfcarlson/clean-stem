# clean-stem
Automatically cleans up profanity in audio using AI generated stems

See [VISION.md](VISION.md) for goals and architecture.

## Development

Requires [uv](https://docs.astral.sh/uv/) and ffmpeg.

```sh
uv sync
uv run clean-stem check path/to/song.mp3   # scan embedded lyrics for profanity
uv run pytest
```

### Sample tracks

`samples/manifest.tsv` lists the test tracks and what result each should give. The audio is copyrighted and is not committed to the repo. To copy it from your own library:

```sh
scripts/fetch_samples.sh /path/to/Music   # default: /Volumes/Media/Music
```
