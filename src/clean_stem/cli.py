"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from clean_stem.lookup import (
    DEFAULT_PROVIDERS,
    LrclibProvider,
    lookup_lyrics,
    lyrics_similarity,
    read_track_info,
)
from clean_stem.lyrics import read_embedded_lyrics
from clean_stem.profanity import WORD_LIST_VERSION, find_profanity


def find_lyrics(path: Path, online: bool):
    """Embedded tags first, then online providers. Returns (text, source) or None."""
    lyrics = read_embedded_lyrics(path)
    if lyrics is not None:
        return lyrics.text, lyrics.source
    if online and (track := read_track_info(path)) is not None:
        if (fetched := lookup_lyrics(track)) is not None:
            return fetched.text, fetched.source
    return None


def check_file(path: Path, online: bool = True) -> dict:
    found = find_lyrics(path, online)
    if found is None:
        return {"file": str(path), "status": "no-lyrics"}
    text, source = found
    matches = find_profanity(text)
    return {
        "file": str(path),
        "status": "explicit" if matches else "clean",
        "lyrics_source": source,
        "word_list_version": WORD_LIST_VERSION,
        "matches": [vars(m) for m in matches],
    }


def print_result(result: dict) -> None:
    status = result["status"]
    source = f"  [{result['lyrics_source']}]" if "lyrics_source" in result else ""
    print(f"{status.upper():9} {result['file']}{source}")
    if status == "explicit":
        counts = Counter(m["word"].lower() for m in result["matches"])
        summary = ", ".join(f"{w} x{n}" if n > 1 else w for w, n in counts.items())
        print(f"          {len(result['matches'])} flagged: {summary}")


def cmd_check(args: argparse.Namespace) -> int:
    results = [check_file(p, online=not args.offline) for p in args.files]
    if args.json:
        json.dump(results, sys.stdout, indent=2)
        print()
    else:
        for result in results:
            print_result(result)
    return 0


def cmd_lyrics(args: argparse.Namespace) -> int:
    """Show what each source returns for a file and how well they agree."""
    for path in args.files:
        track = read_track_info(path)
        if track is None:
            print(f"{path}: missing artist/title tags")
            continue
        print(f"{path}")
        print(f"  {track.artist} - {track.title} ({track.album or 'no album'}, {track.duration:.1f}s)")

        texts = {}
        if (embedded := read_embedded_lyrics(path)) is not None:
            texts[embedded.source] = embedded.text
        for provider in DEFAULT_PROVIDERS:
            fetched = lookup_lyrics(track, providers=(provider,))
            if fetched is None:
                print(f"  {provider.name:12} not found")
                continue
            texts[fetched.source] = fetched.text
            detail = f"{len(fetched.text.split())} words"
            if fetched.duration is not None:
                detail += f", {fetched.duration:.0f}s"
            if fetched.synced:
                detail += f", {len(fetched.synced)} synced lines"
            print(f"  {provider.name:12} {fetched.source}: {detail}")

        sources = list(texts)
        for i, a in enumerate(sources):
            for b in sources[i + 1 :]:
                print(f"  agreement {a} vs {b}: {lyrics_similarity(texts[a], texts[b]):.0%}")
    return 0


def cmd_align(args: argparse.Namespace) -> int:
    """Find when each flagged word is sung, using synced lyrics + Whisper."""
    from clean_stem.align import DEFAULT_MODEL, Transcriber, align_flagged_words, mix_listener, vocals_listener
    from clean_stem.audio import decode_mono, decode_stereo

    transcriber = separator = None
    results = []
    for path in args.files:
        track = read_track_info(path)
        fetched = LrclibProvider().fetch(track) if track else None
        if not fetched or not fetched.synced:
            results.append({"file": str(path), "status": "no-synced-lyrics"})
            continue
        if transcriber is None:
            transcriber = Transcriber(args.model or DEFAULT_MODEL)
        listeners = [("mix", mix_listener(decode_mono(path), transcriber))]
        if not args.no_separate:
            if separator is None:
                from clean_stem.separate import Separator

                separator = Separator()
            stereo = decode_stereo(path, separator.sample_rate)
            listeners.insert(0, ("vocals", vocals_listener(stereo, separator.sample_rate, separator, transcriber)))
        words = align_flagged_words(fetched.synced, track.duration, listeners)
        results.append(
            {
                "file": str(path),
                "status": "aligned",
                "lyrics_source": fetched.source,
                "words": [{**vars(w), "line_text": fetched.synced[w.line].text} for w in words],
            }
        )

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        print()
        return 0
    for result in results:
        print(f"{result['file']}  [{result.get('lyrics_source', result['status'])}]")
        for w in result.get("words", []):
            how = f"{w['method']}/{w['source']}" if w["source"] else w["method"]
            print(f"  {w['start']:7.2f}-{w['end']:7.2f}  {how:16}  {w['word']:<14} {w['line_text']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clean-stem")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Scan lyrics (embedded, then online) for profanity")
    check.add_argument("files", nargs="+", type=Path)
    check.add_argument("--json", action="store_true", help="Output JSON")
    check.add_argument("--offline", action="store_true", help="Only use embedded lyrics")
    check.set_defaults(func=cmd_check)

    lyrics = sub.add_parser("lyrics", help="Compare lyrics from tags and online sources")
    lyrics.add_argument("files", nargs="+", type=Path)
    lyrics.set_defaults(func=cmd_lyrics)

    align = sub.add_parser("align", help="Find timestamps of flagged words (synced lyrics + Whisper)")
    align.add_argument("files", nargs="+", type=Path)
    align.add_argument("--model", help="Whisper model size (default: small)")
    align.add_argument("--no-separate", action="store_true", help="Skip Demucs; transcribe the full mix only")
    align.add_argument("--json", action="store_true", help="Output JSON")
    align.set_defaults(func=cmd_align)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
