"""Cut a window around a flagged word into mix / vocals / instrumental WAVs.

Run with the project venv:
    uv run python experiments/sung_replacement/prep.py SONG START END OUTDIR [--ref A B]

START/END are the word's span (seconds). Writes window_{mix,vocals,inst}.wav
covering START-8s..END+4s, word.wav (the word's vocals alone) and, with --ref,
ref.wav: the vocal stem over [A, B] to use as the voice-conversion reference.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from clean_stem.audio import decode_stereo
from clean_stem.separate import Separator

PRE, POST = 8.0, 4.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("song", type=Path)
    p.add_argument("start", type=float)
    p.add_argument("end", type=float)
    p.add_argument("outdir", type=Path)
    p.add_argument("--ref", type=float, nargs=2)
    a = p.parse_args()

    sep = Separator()
    sr = sep.sample_rate
    stereo = decode_stereo(a.song, sr)
    a.outdir.mkdir(parents=True, exist_ok=True)

    lo, hi = max(0.0, a.start - PRE), a.end + POST
    stems = sep.separate(stereo, sr, lo, hi)
    mix = stereo[:, int(lo * sr) : int(lo * sr) + stems.vocals.shape[1]]
    for name, x in (("mix", mix), ("vocals", stems.vocals), ("inst", stems.instrumental)):
        sf.write(a.outdir / f"window_{name}.wav", x.T, sr)
    w = stems.slice(a.start, a.end, sr)
    sf.write(a.outdir / "word.wav", w.vocals.T, sr)

    if a.ref:
        ref = sep.separate(stereo, sr, *a.ref)
        sf.write(a.outdir / "ref.wav", ref.vocals.T, sr)
    (a.outdir / "meta.json").write_text(json.dumps({"window_start": lo, "start": a.start, "end": a.end, "sr": sr}))


if __name__ == "__main__":
    main()
