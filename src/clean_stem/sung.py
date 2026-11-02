"""The "sung" fill: replace a sung swear with a clean word in the artist's voice.

Voice conversion runs in a separate Seed-VC environment (see
scripts/setup_seedvc.sh) through `sung_worker.py`, started once per run.
Swears that aren't sung (rap, short words), have no clean substitute, or fail
to convert keep the default fill.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from clean_stem.separate import Stems

# Clean words to sing in place of a swear, by lowercase spelling. A trailing
# "'s" is carried over ("shit's" -> "shoot's"). Swears not listed here (slurs,
# words without a natural substitute) always get the default fill.
SUBSTITUTES = {
    "fuck": "fudge",
    "fucks": "fudges",
    "fucked": "freaked",
    "fucking": "freaking",
    "fuckin": "freakin",
    "fuckin'": "freakin'",
    "fucker": "sucker",
    "fuckers": "suckers",
    "motherfucker": "mother trucker",
    "motherfuckers": "mother truckers",
    "motherfucking": "mother freaking",
    "shit": "shoot",
    "shits": "shoots",
    "shitty": "shoddy",
    "bullshit": "bull crap",
    "damn": "dang",
    "damned": "darned",
    "goddamn": "god dang",
    "goddamned": "god darned",
    "bitch": "witch",
    "bitches": "witches",
    "bastard": "rascal",
    "bastards": "rascals",
}

# Seconds of the singer's real vocals sent either side of the swear. The model
# gets context, and its edge artifacts land outside the word.
CONTEXT = 0.5
# Seconds of the artist's vocals used as the voice reference.
REFERENCE_LENGTH = 8.0
# Vocals within this far of a swear are left out of the reference.
REFERENCE_GUARD = 0.3
# Reference audio quieter than this (dBFS RMS over 50 ms) is left out too.
REFERENCE_SILENCE_DB = -40.0

WORKER = Path(__file__).with_name("sung_worker.py")
# The worker's (Seed-VC 44k model's) sample rate; the stems must match.
SAMPLE_RATE = 44100


def substitute(word: str) -> str | None:
    """The clean word to sing in place of `word`, or None."""
    w = word.lower().replace("’", "'")
    suffix = ""
    if w.endswith("'s") and w[:-2] in SUBSTITUTES:
        w, suffix = w[:-2], "'s"
    sub = SUBSTITUTES.get(w)
    return sub + suffix if sub else None


def build_reference(stems: list[Stems], swears: list[tuple[float, float]], sample_rate: int) -> np.ndarray:
    """Up to REFERENCE_LENGTH seconds of the artist's isolated vocals (mono),
    from the separated windows, skipping the swears and near-silence."""
    hop = int(0.05 * sample_rate)
    pieces, total = [], 0
    for s in sorted(stems, key=lambda s: s.start):
        mono = s.vocals.mean(axis=0)
        for lo in range(0, len(mono) - hop + 1, hop):
            t0 = s.start + lo / sample_rate
            t1 = t0 + hop / sample_rate
            if any(t0 < b + REFERENCE_GUARD and t1 > a - REFERENCE_GUARD for a, b in swears):
                continue
            chunk = mono[lo : lo + hop]
            if 20 * np.log10(np.sqrt(np.mean(np.square(chunk))) + 1e-12) < REFERENCE_SILENCE_DB:
                continue
            pieces.append(chunk)
            total += hop
            if total >= REFERENCE_LENGTH * sample_rate:
                return np.concatenate(pieces).astype(np.float32)
    return np.concatenate(pieces).astype(np.float32) if pieces else np.zeros(0, np.float32)


@dataclass(frozen=True)
class SungVocal:
    """A converted vocal: `audio` (mono, at the stems' rate) starts at `start`
    seconds into the song and covers its swear plus CONTEXT either side."""

    start: float
    audio: np.ndarray
    word: str


class SeedVCError(RuntimeError):
    pass


def default_seedvc_dir() -> Path | None:
    env = os.environ.get("CLEAN_STEM_SEEDVC")
    return Path(env) if env else None


class SungFill:
    """Talks to a long-running sung_worker.py in the Seed-VC environment.

    `seedvc_dir` is the Seed-VC checkout; its Python is `python` or else
    `$CLEAN_STEM_SEEDVC_PYTHON` or else `<seedvc_dir>/.venv/bin/python`.
    """

    def __init__(self, seedvc_dir: Path, python: Path | None = None):
        self.seedvc_dir = seedvc_dir
        env_python = os.environ.get("CLEAN_STEM_SEEDVC_PYTHON")
        self.python = python or (Path(env_python) if env_python else seedvc_dir / ".venv" / "bin" / "python")
        if not (seedvc_dir / "inference.py").exists():
            raise SeedVCError(f"{seedvc_dir} is not a Seed-VC checkout; see scripts/setup_seedvc.sh")
        if not self.python.exists():
            raise SeedVCError(f"no Seed-VC Python at {self.python}; see scripts/setup_seedvc.sh")
        self._proc: subprocess.Popen | None = None
        self._tmp = tempfile.TemporaryDirectory(prefix="clean-stem-sung-")
        self._refs = 0

    def _ask(self, request: dict) -> dict:
        if self._proc is None:
            self._proc = subprocess.Popen(
                [str(self.python), str(WORKER)],
                cwd=self.seedvc_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL if not os.environ.get("CLEAN_STEM_DEBUG") else None,
                text=True,
            )
            self._read()  # {"ready": true} once the models are loaded
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()
        reply = self._read()
        if "error" in reply:
            raise SeedVCError(reply["error"])
        return reply

    def _read(self) -> dict:
        line = self._proc.stdout.readline()
        if not line:
            raise SeedVCError(f"Seed-VC worker exited (code {self._proc.poll()})")
        return json.loads(line)

    def set_reference(self, audio: np.ndarray) -> str:
        """Send the artist's voice; returns an id for `convert`."""
        self._refs += 1
        ref_id = f"ref{self._refs}"
        path = Path(self._tmp.name) / f"{ref_id}.npy"
        np.save(path, audio)
        self._ask({"op": "reference", "id": ref_id, "path": str(path)})
        return ref_id

    def convert(self, ref_id: str, stems: Stems, start: float, end: float, word: str, sample_rate: int) -> SungVocal | str:
        """Sing `word` over the swear at [start, end] (song seconds) in the
        stems' window. Returns the converted vocal, or why it wasn't."""
        clip_start = max(stems.start, start - CONTEXT)
        clip_end = min(stems.end, end + CONTEXT)
        lo = int(round((clip_start - stems.start) * sample_rate))
        hi = int(round((clip_end - stems.start) * sample_rate))
        clip = stems.vocals[:, lo:hi].mean(axis=0).astype(np.float32)
        tmp = Path(self._tmp.name)
        np.save(tmp / "clip.npy", clip)
        reply = self._ask(
            {
                "op": "convert",
                "ref": ref_id,
                "path": str(tmp / "clip.npy"),
                "start": int(round((start - clip_start) * sample_rate)),
                "end": int(round((end - clip_start) * sample_rate)),
                "word": word,
                "out": str(tmp / "out.npy"),
            }
        )
        if not reply.get("sung"):
            return reply.get("reason", "not sung")
        return SungVocal(clip_start, np.load(tmp / "out.npy"), word)

    def close(self) -> None:
        if self._proc is not None:
            self._proc.stdin.close()
            self._proc.wait(timeout=30)
            self._proc = None
        self._tmp.cleanup()

