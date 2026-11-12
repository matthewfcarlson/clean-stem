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
    try:
        with tmp.open("wb") as f:
            with zipfile.ZipFile(f, "w") as zf:
                zf.writestr("manifest.json", json.dumps(manifest, indent=2), compress_type=zipfile.ZIP_DEFLATED)
                # Audio doesn't compress; store it as-is.
                zf.write(original, "original" + original.suffix.lower(), compress_type=zipfile.ZIP_STORED)
            f.flush()
            os.fsync(f.fileno())
        # Read it back before anyone relies on it: the original is about to go.
        if stored_sha256(tmp, original.suffix) != manifest["original_sha256"]:
            raise SidecarError(f"{tmp} doesn't read back as the original")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, dest)
    return dest


def stored_sha256(side: Path, suffix: str) -> str:
    """Hash of the original stored in a sidecar (zip CRCs checked on the way)."""
    h = hashlib.sha256()
    with zipfile.ZipFile(side) as zf, zf.open("original" + suffix.lower()) as src:
        for chunk in iter(lambda: src.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class SidecarError(Exception):
    pass


def read_manifest(path: Path) -> dict:
    with zipfile.ZipFile(sidecar_path(path)) as zf:
        return json.loads(zf.read("manifest.json"))


def restore_in_place(path: Path, times_ns: tuple[int, int] | None = None) -> None:
    """Write the sidecar's original back into `path` itself, without a rename, then drop the sidecar.

    For when renaming over `path` failed and left some other copy showing (an Unraid share with a
    stale copy of the file on a second disk does this). `times_ns` restores (atime, mtime). The
    sidecar is kept if the written bytes don't hash back to the original.
    """
    side = sidecar_path(path)
    with zipfile.ZipFile(side) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        with zf.open("original" + path.suffix.lower()) as src, path.open("wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
    if sha256(path) != manifest["original_sha256"]:
        raise SidecarError(f"{path} doesn't match the original after restoring it in place; it is still in {side}")
    if times_ns is not None:
        os.utime(path, ns=times_ns)
    side.unlink()


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
