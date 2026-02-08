# clean-stem Vision

An open-source tool that batch-processes a local music library to censor profanity by replacing vocal segments with instrumentals. Designed for parents who want kid-safe listening without maintaining a separate "clean" copy of their library.

## Problem

You have a large local music collection served via Plex/Plexamp. You want to listen around young kids without worrying about explicit lyrics. Existing solutions either require buying duplicate "clean" versions of albums, rely on streaming services with incomplete clean catalogs, or destructively edit your files with no way back.

## Goals

- **Automatically detect and censor profanity in music files** by replacing flagged vocal segments with the underlying instrumentals (via AI stem separation).
- **Reversible** — every modification can be undone. Original audio snippets are preserved in `.dirtystem` sidecar files so the original file can be fully restored.
- **Non-destructive to library structure** — files are modified in-place so Plex picks them up on rescan. No separate directory of clean copies.
- **Bail early, bail often** — the pipeline is designed to skip work at every stage. Songs already processed get skipped. Songs with clean lyrics get skipped. Only the segments around flagged words go through the expensive stem separation.
- **Incremental** — runs periodically against a music directory, processing only new or changed files.
- **Best-effort** — this doesn't need to be perfect. The mental model for profanity is "words a 2-year-old would repeat." We are not filtering euphemisms, innuendo, or thematic content.
- **Open source** — MIT or similar permissive license.

## Non-Goals

- **Podcasts or spoken-word content** — this is for music only.
- **Real-time playback filtering** — we pre-process files, not intercept audio streams.
- **Desktop app or CLI** — this runs as a Docker container (ideal for Unraid/NAS setups), not on a laptop.
- **Comprehensive content filtering** — we are not rating songs for violence, drug references, sexual themes, etc. Just explicit profanity.
- **Adjustable profanity thresholds** — may be a future feature, but v1 targets a single sensible default word list.
- **GPU requirement** — must work on CPU-only hardware, even if it takes days for a large library.

## Target User

A parent with a large local music collection (10-20k songs) served via Plex, running on a NAS or home server (e.g., Unraid). Comfortable with Docker but not necessarily a developer.

## Architecture Overview

### Deployment

- **Docker container** with a lightweight web UI for monitoring progress and viewing configuration.
- Configuration via a TOML/YAML config file mounted into the container. The web UI displays config but does not modify it — edits are made by hand.
- Scans a mounted music directory periodically, processing new/changed files.

### Pipeline

The processing pipeline is designed to bail out as early as possible to avoid unnecessary work:

```
For each audio file in the music directory:
  1. SKIP if already processed (check for .dirtystem sidecar)
  2. SKIP if file hasn't changed since last scan

  3. LYRICS LOOKUP
     - Check embedded metadata/tags for lyrics
     - Fall back to external API (Genius, Musixmatch, etc.)

  4. PROFANITY CHECK
     - Scan lyrics against word list
     - SKIP if no profanity found (song is clean)

  5. WORD-LEVEL ALIGNMENT
     - Run Whisper (faster-whisper, INT8, CPU) on the raw audio
     - Align Whisper transcript against known lyrics
     - Identify timestamps for each flagged word

  6. SEGMENT EXTRACTION
     - For each flagged timestamp, extract a ~5-10 second window

  7. STEM SEPARATION
     - Run Demucs on only the extracted segments (not the whole song)
     - Produce instrumental-only versions of each segment

  8. SPLICE
     - Replace the vocal segments in the original file with instrumentals
     - Save the original audio snippets + timestamps to a .dirtystem sidecar file

  9. MARK COMPLETE
     - Record processing metadata so the file is skipped on future scans
```

### Songs Without Lyrics

For songs where lyrics cannot be found (no embedded tags, no API results):

- Run Whisper on the full raw audio to generate a transcript
- If Whisper confidence is low, escalate: run Demucs to isolate vocals, then re-run Whisper on the vocal track
- Proceed with profanity detection on the generated transcript

### Reversibility

Each modified file gets a sidecar: `song.mp3.dirtystem`

The sidecar contains:
- Original audio snippets that were replaced
- Timestamps and durations of each replacement
- Metadata (processing date, word list version, detected words)
- Enough information to fully reconstruct the original file

Running `clean-stem reverse <file>` (or a batch reverse) restores the original audio.

### Supported Formats

- MP3
- FLAC
- OGG
- M4A/AAC

### Key Technology Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Stem separation | [Demucs](https://github.com/facebookresearch/demucs) | Best open-source quality for vocal/instrumental separation |
| Speech-to-text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | ~3x real-time on CPU with INT8 quantization |
| Language | Python | ML ecosystem compatibility (Demucs, Whisper) |
| Packaging | Docker | Avoids Python dependency bitrot, runs on NAS/Unraid |
| Web UI | TBD | Lightweight — just progress monitoring and config display |
| Config | TOML or YAML | Simple, human-editable |

## Open Questions

- **Lyrics API choice** — Genius, Musixmatch, or something else? Licensing/rate limits may be a factor.
- **Profanity word list source** — Curate our own, or use an existing open-source list?
- **Whisper model size** — `base` is faster but less accurate; `medium` is slower but better. What's the right default for CPU-only?
- **Sidecar format** — Binary (smaller, faster) or JSON + raw audio blobs (more inspectable)?
- **Web UI framework** — Something minimal (Flask + htmx? FastAPI + simple HTML?) since it's just a monitoring dashboard.
- **Demucs model variant** — htdemucs is the default, but there are tradeoffs between speed and quality.
- **How to handle live files** — What if Plex is playing a file while we're trying to modify it?
- **Crossfade** — When splicing instrumentals over vocals, do we need a short crossfade to avoid audible clicks?

## Milestones

### v0.1 — Proof of Concept
- CLI that processes a single file end-to-end
- Hardcoded profanity list
- Lyrics from embedded tags only
- MP3 support only
- `.dirtystem` sidecar generation
- Reverse operation

### v0.2 — Batch Processing
- Directory scanning with skip logic
- External lyrics API integration
- All four format support (MP3, FLAC, OGG, M4A)
- Progress logging

### v0.3 — Docker + Web UI
- Dockerfile
- Web UI for progress monitoring
- Config file support
- Periodic scanning

### v1.0 — Ready for Daily Use
- Stable pipeline
- Good documentation
- Unraid community app template
- Battle-tested on a real 15k song library
