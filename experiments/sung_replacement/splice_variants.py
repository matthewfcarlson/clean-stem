"""Render comparison clips for one case directory.

    python splice_variants.py CASE_DIR OUTDIR [--borrow A B]

Variants: original, reversed (current default fill), instrumental, seedvc
(the Seed-VC conversion of source.wav), and with --borrow, the vocal stem over
[A, B] seconds of the window reshaped to the swear's length and melody, i.e.
the singer's own voice taken from elsewhere in the song.
"""

import argparse
import json
import subprocess
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from make_source import SR, reshape, trim

FADE = 0.03  # overridden by --fade


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def fit(x: np.ndarray, n: int) -> np.ndarray:
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("case", type=Path)
    p.add_argument("outdir", type=Path)
    p.add_argument("--borrow", type=float, nargs=2)
    p.add_argument("--fade", type=float, default=FADE)
    p.add_argument("--tag", default="")
    a = p.parse_args()

    meta = json.loads((a.case / "meta.json").read_text())
    mix, sr = sf.read(a.case / "window_mix.wav", dtype="float32")
    inst, _ = sf.read(a.case / "window_inst.wav", dtype="float32")
    vocals, _ = sf.read(a.case / "window_vocals.wav", dtype="float32")
    assert sr == SR
    lo = int(round((meta["start"] - meta["window_start"]) * sr))
    word = vocals[lo : lo + int(round((meta["end"] - meta["start"]) * sr))]
    n = len(word)
    target_rms = rms(word)

    def splice(new_vocal: np.ndarray | None) -> np.ndarray:
        """mix outside [lo, lo+n]; instrumental + new_vocal inside, crossfaded."""
        fill = inst[lo : lo + n].copy()
        if new_vocal is not None:
            fill += new_vocal
        f = int(a.fade * sr)
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, f))[:, None]
        out = mix.copy()
        seg = out[lo : lo + n]
        w = np.ones((n, 1), dtype=np.float32)
        w[:f] = ramp
        w[-f:] = ramp[::-1]
        out[lo : lo + n] = seg * (1 - w) + fill * w
        return out

    def stereo_like(mono: np.ndarray) -> np.ndarray:
        mono = fit(mono, n)
        mono = mono * (target_rms / rms(mono))
        return np.repeat(mono[:, None], 2, axis=1).astype(np.float32)

    variants = {
        "1-original": mix,
        "2-reversed-6dB": splice(word[::-1] * 0.5),
        "3-instrumental": splice(None),
    }
    vc_files = sorted((a.case / "vc").glob("*.wav"))
    if vc_files:
        vc, _ = librosa.load(vc_files[0], sr=SR)
        off = int(round(meta.get("source_offset", 0.0) * sr))
        variants["4-seedvc"] = splice(stereo_like(vc[off : off + n]))
    if a.borrow:
        b = vocals[int(a.borrow[0] * sr) : int(a.borrow[1] * sr)].mean(axis=1)
        variants["5-borrowed"] = splice(stereo_like(reshape(trim(b, 30), word.mean(axis=1))))

    a.outdir.mkdir(parents=True, exist_ok=True)
    for name, x in variants.items():
        wav = a.outdir / f"{a.case.name}-{name}{a.tag}.wav"
        sf.write(wav, np.clip(x, -1, 1), sr)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav), "-b:a", "192k", str(wav.with_suffix(".mp3"))], check=True)
        wav.unlink()
    print("wrote", ", ".join(variants))


if __name__ == "__main__":
    main()
