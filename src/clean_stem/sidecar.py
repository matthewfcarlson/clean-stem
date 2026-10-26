"""The .dirtystem sidecar that makes every modification reversible.

A sidecar is a zip archive next to the modified file (song.mp3.dirtystem):

    manifest.json   what was replaced, when, why, and hashes of both files
    original.<ext>  the original file's exact bytes

Storing the whole original costs space, but re-encoding a lossy file means
spliced-back snippets could never reproduce it exactly; this always can.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

SUFFIX = ".dirtystem"
FORMAT_VERSION = 1


def sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + SUFFIX)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sidecar(original: Path, modified: Path, details: dict) -> Path:
    """Record `original` (still untouched on disk) before it is replaced by `modified`."""
    manifest = {
        "format_version": FORMAT_VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file": original.name,
        "original_sha256": sha256(original),
        "modified_sha256": sha256(modified),
        **details,
    }
    dest = sidecar_path(original)
    tmp = dest.with_name(dest.name + ".tmp")
    with zipfile.ZipFile(tmp, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2), compress_type=zipfile.ZIP_DEFLATED)
        # Audio doesn't compress; store it as-is.
        zf.write(original, "original" + original.suffix.lower(), compress_type=zipfile.ZIP_STORED)
    os.replace(tmp, dest)
    return dest


def read_manifest(path: Path) -> dict:
    with zipfile.ZipFile(sidecar_path(path)) as zf:
        return json.loads(zf.read("manifest.json"))


class ReverseError(Exception):
    pass


def reverse(path: Path, force: bool = False) -> str:
    """Restore the original file from its sidecar and delete the sidecar.

    Returns a short description of what happened. Refuses (unless `force`) if
    the file has changed since it was cleaned, since restoring would discard
    those changes.
    """
    side = sidecar_path(path)
    if not side.exists():
        raise ReverseError("no sidecar")
    with zipfile.ZipFile(side) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        current = sha256(path) if path.exists() else None
        if current == manifest["original_sha256"]:
            # Interrupted before the modified file was put in place.
            side.unlink()
            return "already original; removed sidecar"
        if current != manifest["modified_sha256"] and not force:
            raise ReverseError("file changed since it was cleaned (use --force to restore anyway)")
        tmp = path.with_name(path.name + ".restore.tmp")
        with zf.open("original" + path.suffix.lower()) as src, tmp.open("wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
    if sha256(tmp) != manifest["original_sha256"]:
        tmp.unlink()
        raise ReverseError("sidecar is corrupt: restored bytes don't match the original hash")
    if path.exists():
        shutil.copymode(path, tmp)
    os.replace(tmp, path)
    side.unlink()
    return "restored"
