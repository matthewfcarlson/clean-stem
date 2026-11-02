"""Seed-VC worker: sings a replacement word in the artist's voice.

This file is not imported by clean_stem. `clean_stem.sung` runs it with the
Python of a separate Seed-VC environment (see scripts/setup_seedvc.sh), with
the Seed-VC checkout as the working directory, so it must stay compatible with
that environment (Python 3.10, torch 2.4) and import nothing from clean_stem.

Protocol: one JSON request per line on stdin, one JSON reply per line on
stdout. Audio is exchanged as mono float32 .npy files at 44.1 kHz.

    {"op": "reference", "id": ..., "path": ...}
        Prepare a voice reference (the artist's isolated vocals).
    {"op": "convert", "ref": ..., "path": ..., "start": i, "end": j,
     "word": ..., "out": ...}
        `path` holds the vocals around a swear, which spans samples [i, j).
        Replies {"sung": false, "reason": ...} if the swear isn't sung;
        otherwise writes the clip with the swear replaced by `word` to `out`.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import librosa
import numpy as np
import pyworld as pw
import torch
import torchaudio

SR = 44100
DIFFUSION_STEPS = int(os.environ.get("CLEAN_STEM_SUNG_STEPS", "30"))
INFERENCE_CFG_RATE = 0.7
# Seconds of reference voice. Seed-VC allows up to 25, but the diffusion runs
# over the reference on every step, so it dominates the cost.
MAX_REFERENCE = float(os.environ.get("CLEAN_STEM_SUNG_REFERENCE", "8"))

# A swear counts as sung if it lasts at least this long, is mostly voiced, and
# its pitch holds notes: at least MIN_HELD of its 5 ms pitch frames move less
# than HELD_STEP semitones. Rapped words are short and glide like speech, and
# the conversion garbles them. Calibrated on the sample songs: sung swears in
# vampire and all-american-bitch hold 30-70%, rapped ones in squabble-up and
# without-me mostly 11-35%.
MIN_SUNG_LENGTH = 0.4
MIN_VOICED = 0.5
HELD_STEP = 0.15
MIN_HELD = 0.4


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def pitch(x):
    x = x.astype(np.float64)
    f0, t = pw.harvest(x, SR, f0_floor=70, f0_ceil=1000)
    return pw.stonemask(x, f0, t, SR)


def sung_reason(word):
    """Why `word` (the swear's vocals) doesn't count as sung, or None."""
    length = len(word) / SR
    if length < MIN_SUNG_LENGTH:
        return f"too short ({length:.2f}s)"
    f0 = pitch(word)
    voiced = f0 > 0
    if voiced.mean() < MIN_VOICED:
        return f"mostly unvoiced ({voiced.mean():.0%})"
    semitones = 12 * np.log2(np.where(voiced, f0, 1.0))
    steps = np.abs(np.diff(semitones))[voiced[1:] & voiced[:-1]]
    held = float((steps < HELD_STEP).mean()) if len(steps) else 0.0
    if held < MIN_HELD:
        return f"no held notes ({held:.0%} held)"
    return None


def speak(word):
    """The word spoken by a local text-to-speech engine, trimmed, at SR."""
    with tempfile.TemporaryDirectory() as tmp:
        if shutil.which("say"):
            path = os.path.join(tmp, "word.aiff")
            subprocess.run(["say", "-o", path, word], check=True)
        else:
            engine = shutil.which("espeak-ng") or shutil.which("espeak")
            if not engine:
                raise RuntimeError("no text-to-speech engine (say, espeak-ng or espeak)")
            path = os.path.join(tmp, "word.wav")
            subprocess.run([engine, "-w", path, word], check=True)
        audio, _ = librosa.load(path, sr=SR)
    _, (lo, hi) = librosa.effects.trim(audio, top_db=35)
    return audio[lo:hi]


def reshape(word, orig):
    """Resynthesize `word` with the length and pitch contour of `orig`."""
    word = word.astype(np.float64)
    f0_o = pitch(orig)

    # Analyse the new word, stretch its frames to the original's length.
    f0_t, t_t = pw.harvest(word, SR, f0_floor=60, f0_ceil=1000)
    sp = pw.cheaptrick(word, f0_t, t_t, SR)
    ap = pw.d4c(word, f0_t, t_t, SR)
    n = len(f0_o)
    idx = np.round(np.linspace(0, len(f0_t) - 1, n)).astype(int)
    voiced_t = f0_t[idx] > 0

    # Take the original melody where the new word is voiced; fill gaps in the
    # original contour by interpolation so vowels never lose their pitch.
    v = f0_o > 0
    f0 = np.interp(np.arange(n), np.flatnonzero(v), f0_o[v]) if v.any() else np.full(n, np.median(f0_t[f0_t > 0]))
    f0 = np.where(voiced_t, f0, 0.0)

    y = pw.synthesize(f0, np.ascontiguousarray(sp[idx]), np.ascontiguousarray(ap[idx]), SR)[: len(orig)]
    y = np.pad(y, (0, len(orig) - len(y)))
    return (y * (rms(orig) / rms(y))).astype(np.float32)


class SeedVC:
    def __init__(self):
        sys.path.insert(0, os.getcwd())
        import inference

        # Seed-VC picks MPS on Apple Silicon, which can't run its float64 pitch
        # code; clean-stem targets CPU anyway.
        inference.device = self.device = torch.device("cpu")
        # It also loads its Whisper encoder in float16, which is ~10x slower
        # than float32 on CPU.
        import transformers

        load_whisper = transformers.WhisperModel.from_pretrained
        transformers.WhisperModel.from_pretrained = lambda *a, **kw: load_whisper(*a, **{**kw, "torch_dtype": torch.float32})
        args = SimpleNamespace(f0_condition=True, checkpoint=None, config=None, fp16=False)
        (self.model, self.semantic_fn, self.f0_fn, self.vocoder_fn, self.campplus, self.mel_fn, _) = inference.load_models(args)
        self.refs = {}

    @torch.no_grad()
    def reference(self, ref_id, audio):
        audio = torch.tensor(audio[: int(SR * MAX_REFERENCE)]).unsqueeze(0).float()
        waves_16k = torchaudio.functional.resample(audio, SR, 16000)
        mel = self.mel_fn(audio)
        fbank = torchaudio.compliance.kaldi.fbank(waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        style = self.campplus((fbank - fbank.mean(dim=0, keepdim=True)).unsqueeze(0))
        f0 = torch.from_numpy(self.f0_fn(waves_16k[0], thred=0.03)).float()[None]
        prompt, *_ = self.model.length_regulator(
            self.semantic_fn(waves_16k), ylens=torch.LongTensor([mel.size(2)]), n_quantizers=3, f0=f0
        )
        self.refs[ref_id] = (mel, style, prompt)

    @torch.no_grad()
    def convert(self, ref_id, source):
        mel2, style, prompt = self.refs[ref_id]
        audio = torch.tensor(source).unsqueeze(0).float()
        waves_16k = torchaudio.functional.resample(audio, SR, 16000)
        mel = self.mel_fn(audio)
        f0 = torch.from_numpy(self.f0_fn(waves_16k[0], thred=0.03)).float()[None]
        cond, *_ = self.model.length_regulator(
            self.semantic_fn(waves_16k), ylens=torch.LongTensor([mel.size(2)]), n_quantizers=3, f0=f0
        )
        cat = torch.cat([prompt, cond], dim=1)
        torch.manual_seed(0)  # same input, same output
        target = self.model.cfm.inference(
            cat, torch.LongTensor([cat.size(1)]), mel2, style, None, DIFFUSION_STEPS, inference_cfg_rate=INFERENCE_CFG_RATE
        )[:, :, mel2.size(-1) :]
        out = self.vocoder_fn(target.float()).squeeze().numpy()
        return np.pad(out, (0, max(0, len(source) - len(out))))[: len(source)]


def handle(vc, req):
    if req["op"] == "reference":
        vc.reference(req["id"], np.load(req["path"]))
        return {}
    if req["op"] == "convert":
        clip = np.load(req["path"])
        i, j = req["start"], req["end"]
        reason = sung_reason(clip[i:j])
        if reason:
            return {"sung": False, "reason": reason}
        # Keep the singer's real vocals either side of the new word: the model
        # gets context, and its edge artifacts land outside the word.
        source = np.concatenate([clip[:i], reshape(speak(req["word"]), clip[i:j]), clip[j:]])
        out = vc.convert(req["ref"], source)
        out *= rms(clip[i:j]) / rms(out[i:j])
        np.save(req["out"], out.astype(np.float32))
        return {"sung": True}
    raise ValueError(f"unknown op {req['op']!r}")


def main():
    # Replies get the real stdout; anything else that prints (Seed-VC, torch,
    # C libraries) goes to stderr so it can't corrupt the protocol.
    replies = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def reply(obj):
        replies.write(json.dumps(obj) + "\n")
        replies.flush()

    started = time.time()
    vc = SeedVC()
    log(f"seed-vc loaded in {time.time() - started:.1f}s")
    reply({"ready": True})
    for line in sys.stdin:
        req = json.loads(line)
        started = time.time()
        try:
            result = handle(vc, req)
        except Exception as e:  # report and keep serving
            result = {"error": f"{type(e).__name__}: {e}"}
        reply({**result, "seconds": round(time.time() - started, 2)})


if __name__ == "__main__":
    main()
