"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from clean_stem.lyrics import read_embedded_lyrics
from clean_stem.profanity import WORD_LIST_VERSION, find_profanity


def check_file(path: Path) -> dict:
    lyrics = read_embedded_lyrics(path)
    if lyrics is None:
        return {"file": str(path), "status": "no-lyrics"}
    matches = find_profanity(lyrics.text)
    return {
        "file": str(path),
        "status": "explicit" if matches else "clean",
        "lyrics_source": lyrics.source,
        "word_list_version": WORD_LIST_VERSION,
        "matches": [vars(m) for m in matches],
    }


def print_result(result: dict) -> None:
    status = result["status"]
    print(f"{status.upper():9} {result['file']}")
    if status == "explicit":
        counts = Counter(m["word"].lower() for m in result["matches"])
        summary = ", ".join(f"{w} x{n}" if n > 1 else w for w, n in counts.items())
        print(f"          {len(result['matches'])} flagged: {summary}")


def cmd_check(args: argparse.Namespace) -> int:
    results = [check_file(p) for p in args.files]
    if args.json:
        json.dump(results, sys.stdout, indent=2)
        print()
    else:
        for result in results:
            print_result(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clean-stem")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Scan embedded lyrics for profanity")
    check.add_argument("files", nargs="+", type=Path)
    check.add_argument("--json", action="store_true", help="Output JSON")
    check.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
