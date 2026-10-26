"""Walking a library and deciding which files need (re)processing."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from clean_stem.pipeline import SUPPORTED_SUFFIXES
from clean_stem.profanity import WORD_LIST_VERSION
from clean_stem.state import StateDB

# Statuses that mean "done with this file until it changes". Errors (e.g. a
# lyrics site being down) are retried on the next run.
FINAL_STATUSES = ("clean", "cleaned", "skipped", "reversed")


def iter_audio_files(paths: Iterable[Path]) -> Iterator[Path]:
    """Yield supported audio files, recursing into directories in sorted order."""
    for path in paths:
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and _is_candidate(child):
                    yield child
        elif _is_candidate(path):
            yield path


def _is_candidate(path: Path) -> bool:
    # Dotfiles include our own temp files and macOS "._" resource forks.
    return path.suffix.lower() in SUPPORTED_SUFFIXES and not path.name.startswith(".")


def is_unchanged(state: StateDB, path: Path) -> bool:
    """True if `path` was fully handled before and hasn't changed since."""
    record = state.get_file(path)
    if record is None or record.status not in FINAL_STATUSES:
        return False
    if record.word_list_version != WORD_LIST_VERSION:
        return False
    st = path.stat()
    return record.size == st.st_size and record.mtime_ns == st.st_mtime_ns
