"""Speak the replacement word with macOS `say`, then reshape it with WORLD so it
has the original word's length and pitch contour. The result is the content
source for voice conversion (the voice itself comes from the reference).

    python make_source.py CASE_DIR WORD
"""

import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import pyworld as pw
import soundfile as sf

SR = 44100


def trim(x: np.ndarray, db: float = 35) -> np.ndarray:
    _, (lo, hi) = librosa.effects.trim(x, top_db=db)
    return x[lo:hi]


def reshape(word: np.ndarray, orig: np.ndarray) -> np.ndarray:
    """Resynthesize `word` with the length and pitch contour of `orig`."""
    word = word.astype(np.float64)
    orig = orig.astype(np.float64)

    # Original pitch contour (0 where unvoiced).
    f0_o, t_o = pw.harvest(orig, SR, f0_floor=70, f0_ceil=1000)
    f0_o = pw.stonemask(orig, f0_o, t_o, SR)

    # Analyse the new word, stretch its frames to the original's length.
    f0_t, t_t = pw.harvest(word, SR, f0_floor=60, f0_ceil=1000)
    sp = pw.cheaptrick(word, f0_t, t_t, SR)
    ap = pw.d4c(word, f0_t, t_t, SR)
    n = len(f0_o)
    idx = np.round(np.linspace(0, len(f0_t) - 1, n)).astype(int)
    sp_s, ap_s = sp[idx], ap[idx]
    voiced_t = f0_t[idx] > 0

    # Take the original melody where the new word is voiced; fill gaps in the
    # original contour by interpolation so vowels never lose their pitch.
    v = f0_o > 0
    if v.any():
        f0 = np.interp(np.arange(n), np.flatnonzero(v), f0_o[v])
    else:
        f0 = np.full(n, np.median(f0_t[f0_t > 0]))
    f0 = np.where(voiced_t, f0, 0.0)

    y = pw.synthesize(f0, np.ascontiguousarray(sp_s), np.ascontiguousarray(ap_s), SR)[: len(orig)]
    return (y / (np.abs(y).max() + 1e-9) * 0.5).astype(np.float32)


def main() -> None:
    case, word = Path(sys.argv[1]), sys.argv[2]
    aiff = case / "tts.aiff"
    subprocess.run(["say", "-o", str(aiff), word], check=True)
    tts, _ = librosa.load(aiff, sr=SR)
    orig, _ = librosa.load(case / "word.wav", sr=SR)  # mono vocals of the swear
    sf.write(case / "source.wav", reshape(trim(tts), orig), SR)


if __name__ == "__main__":
    main()
