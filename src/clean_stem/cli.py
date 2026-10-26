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
from clean_stem.pipeline import LYRICS_MODES
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
    """Find when each flagged word is sung."""
    from clean_stem.pipeline import Models, locate

    models = Models(whisper_model=args.model or Models.whisper_model, state=open_state(args))
    results = []
    for path in args.files:
        located = locate(path, models, args.lyrics)
        if isinstance(located, str):
            results.append({"file": str(path), "status": "skipped", "reason": located})
            continue
        results.append(
            {
                "file": str(path),
                "status": "aligned",
                "lyrics_source": located.lyrics_source,
                "words": [
                    {**vars(w), "line_text": located.synced[w.line].text if w.line >= 0 else None}
                    for w in located.words
                ],
            }
        )

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        print()
        return 0
    for result in results:
        print(f"{result['file']}  [{result.get('lyrics_source') or result['reason']}]")
        for w in result.get("words", []):
            how = f"{w['method']}/{w['source']}" if w["source"] else w["method"]
            context = w["line_text"] or ("(not in lyrics)" if w["source"] == "vocals-scan" else "")
            print(f"  {w['start']:7.2f}-{w['end']:7.2f}  {how:20}  {w['word']:<14} {context}")
    return 0


def open_state(args: argparse.Namespace):
    from clean_stem.state import StateDB

    return None if args.no_state else StateDB(args.state)


def cmd_clean(args: argparse.Namespace) -> int:
    """Censor files in place, keeping the originals in .dirtystem sidecars."""
    import time

    from clean_stem.batch import is_unchanged, iter_audio_files
    from clean_stem.pipeline import Models, clean

    state = open_state(args)
    models = Models(whisper_model=args.model or Models.whisper_model, state=state)
    files = list(iter_audio_files(args.files))
    results = []
    counts: Counter = Counter()
    for n, path in enumerate(files, start=1):
        if state and not args.force and is_unchanged(state, path):
            counts["unchanged"] += 1
            continue
        started = time.monotonic()
        try:
            result = clean(path, models, args.lyrics)
        except Exception as e:  # keep going through a big library
            result = {"file": str(path), "status": "error", "reason": f"{type(e).__name__}: {e}"}
        if state:
            state.put_file(path, result["status"], result.get("reason"), WORD_LIST_VERSION)
        counts[result["status"]] += 1
        results.append(result)
        if not args.json:
            detail = result.get("reason") or result.get("lyrics_source", "")
            if result["status"] == "cleaned":
                detail = f"{len(result['replacements'])} words replaced  [{detail}]"
            elapsed = time.monotonic() - started
            print(f"[{n}/{len(files)}] {result['status'].upper():8} {path}  {detail}  ({elapsed:.0f}s)", flush=True)
    if args.json:
        json.dump(results, sys.stdout, indent=2)
        print()
    else:
        print("summary: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) if counts else "nothing to do")
    return 1 if counts["error"] else 0


def cmd_reverse(args: argparse.Namespace) -> int:
    """Restore originals from their .dirtystem sidecars."""
    from clean_stem.sidecar import ReverseError, reverse, sidecar_path

    state = open_state(args)
    failed = 0
    for path in args.files:
        if not sidecar_path(path).exists():
            print(f"{'skipped':9} {path}  (not cleaned)")
            continue
        try:
            print(f"{reverse(path, force=args.force):9} {path}")
            if state:
                # Remember the choice, so later runs leave the original alone.
                state.put_file(path, "reversed", "restored by user", WORD_LIST_VERSION)
        except ReverseError as e:
            failed += 1
            print(f"FAILED    {path}: {e}")
    return 1 if failed else 0


LYRICS_HELP = "auto: synced lyrics if available; plain: ignore line timings; none: transcription only"


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
    align.add_argument("--lyrics", choices=LYRICS_MODES, default="auto", help=LYRICS_HELP)
    align.add_argument("--json", action="store_true", help="Output JSON")
    align.set_defaults(func=cmd_align)

    clean = sub.add_parser("clean", help="Censor files in place (originals kept in .dirtystem sidecars)")
    clean.add_argument("files", nargs="+", type=Path, help="Files or directories (searched recursively)")
    clean.add_argument("--force", action="store_true", help="Reprocess files even if unchanged since the last run")
    clean.add_argument("--model", help="Whisper model size (default: small)")
    clean.add_argument("--lyrics", choices=LYRICS_MODES, default="auto", help=LYRICS_HELP)
    clean.add_argument("--json", action="store_true", help="Output JSON")
    clean.set_defaults(func=cmd_clean)

    rev = sub.add_parser("reverse", help="Restore originals from .dirtystem sidecars")
    rev.add_argument("files", nargs="+", type=Path)
    rev.add_argument("--force", action="store_true", help="Restore even if the file changed since cleaning")
    for p in (align, clean, rev):
        p.add_argument("--state", type=Path, default=None, help="State database (default: $CLEAN_STEM_STATE or ~/.cache/clean-stem/state.db)")
        p.add_argument("--no-state", action="store_true", help="Don't read or record state, or cache lyrics")
    rev.set_defaults(func=cmd_reverse)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
