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
     "max_lead": k, "word": ..., "out": ...}
        `path` holds the vocals around a swear, which spans samples [i, j).
        Replies {"sung": false, "reason": ...} if the swear isn't sung;
        otherwise writes the clip with the swear replaced by `word` to `out`
        and replies {"sung": true, "start": i2}: the new word may start up to
        `max_lead` samples early, in a gap before the swear, at i2.
    {"op": "score", "ref": ..., "paths": [...]}
        For the bench: how much each clip sounds like the reference voice
        (cosine similarity of speaker embeddings) and SQUIM's estimates of
        its quality. Replies {"scores": [{"speaker", "stoi", "pesq", "si_sdr"}]}.
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
# 15 sounded the same as Seed-VC's default 30 on vampire, at about half the cost.
DIFFUSION_STEPS = int(os.environ.get("CLEAN_STEM_SUNG_STEPS", "15"))
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

# A new word whose opening consonant is longer than the swear's ("s" in sucker
# vs "f" in fucker) may start up to this early, into a gap before the swear, so
# its vowel still lands on the beat. Singers do the same.
MAX_LEAD = 0.08
# "A gap" is vocals this far below the swear's peak, in 10 ms frames.
GAP_DB = 20


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


def frame_levels(x, times):
    """Log RMS of `x` in 10 ms windows centred on each frame time."""
    half = int(0.005 * SR)
    centres = (np.asarray(times) * SR).astype(int)
    return np.array([np.log(rms(x[max(0, c - half) : c + half]) + 1e-6) for c in centres])


def align_frames(f0_new, level_new, f0_orig, level_orig):
    """For each frame of the original word, the frame of the new word that
    plays there. Dynamic time warping on voicing and loudness lines up
    consonants with consonants and vowels with vowels, so a long "s" in
    "sucker" doesn't push its vowel late against the short "f" of "fucker"."""

    def features(f0, level):
        z = (level - level.mean()) / (level.std() + 1e-6)
        return np.stack([2.0 * (f0 > 0), z])

    _, path = librosa.sequence.dtw(features(f0_orig, level_orig), features(f0_new, level_new))
    idx = np.zeros(len(f0_orig))
    count = np.zeros(len(f0_orig))
    for i, j in path:
        idx[i] += j
        count[i] += 1
    return np.round(idx / count).astype(int)


# The vowel starts at the first voiced frame within this many dB of the word's
# loudest; quieter voiced frames before it are bleed or a voiced consonant.
VOWEL_DB = 12


def vowel_onset(f0, level):
    loud = level >= level.max() - VOWEL_DB * np.log(10) / 20
    frames = np.flatnonzero((f0 > 0) & loud)
    return int(frames[0]) if len(frames) else len(f0)


def reshape(word, orig):
    """Resynthesize `word` with the timing, loudness and pitch contour of `orig`.

    The new word's opening consonant keeps its own length and loudness and
    ends where the vowel of `orig` starts, so its vowel lands where the old one did;
    it may reach back into a gap at the start of `orig` (see lead_in). From
    the vowel on, frames are matched by dynamic time warping and follow the
    original's loudness.
    """
    word = word.astype(np.float64)
    orig = orig.astype(np.float64)
    f0_o, t_o = pw.harvest(orig, SR, f0_floor=70, f0_ceil=1000)
    f0_o = pw.stonemask(orig, f0_o, t_o, SR)
    f0_t, t_t = pw.harvest(word, SR, f0_floor=60, f0_ceil=1000)
    sp = pw.cheaptrick(word, f0_t, t_t, SR)
    ap = pw.d4c(word, f0_t, t_t, SR)
    level_o = frame_levels(orig, t_o)
    level_t = frame_levels(word, t_t)

    n = len(f0_o)
    v_o = min(vowel_onset(f0_o, level_o), n - 1)
    v_t = min(vowel_onset(f0_t, level_t), len(f0_t) - 1)
    consonant = min(v_t, v_o)  # frames
    onset = v_o - consonant
    idx = np.zeros(n, dtype=int)
    idx[onset:v_o] = np.round(np.linspace(v_t - consonant, v_t - 1, consonant)).astype(int) if consonant else []
    idx[v_o:] = v_t + align_frames(f0_t[v_t:], level_t[v_t:], f0_o[v_o:], level_o[v_o:])
    voiced_t = f0_t[idx] > 0

    # Take the original melody where the new word is voiced; fill gaps in the
    # original contour by interpolation so vowels never lose their pitch.
    v = f0_o > 0
    f0 = np.interp(np.arange(n), np.flatnonzero(v), f0_o[v]) if v.any() else np.full(n, np.median(f0_t[f0_t > 0]))
    f0 = np.where(voiced_t, f0, 0.0)

    y = pw.synthesize(f0, np.ascontiguousarray(sp[idx]), np.ascontiguousarray(ap[idx]), SR)[: len(orig)]
    y = np.pad(y, (0, len(orig) - len(y)))

    # From the vowel on, follow the original's loudness contour (smoothed over
    # ~30 ms, limited to +-12 dB) so the word swells and fades where the old
    # one did. The consonant keeps its level relative to the vowel, and
    # anything before it is silent.
    gain = np.exp(np.clip(level_o - frame_levels(y, t_o), np.log(0.25), np.log(4.0)))
    gain = np.convolve(gain, np.ones(6) / 6, mode="same")
    gain[:v_o] = gain[v_o]
    gain[:onset] = 0.0
    y *= np.interp(np.arange(len(y)) / SR, t_o, gain)
    voiced_part = y[int(t_o[v_o] * SR) :]
    return (y * (rms(orig[int(t_o[v_o] * SR) :]) / rms(voiced_part))).astype(np.float32)


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

    def speaker(self, waves_16k):
        """Speaker embedding (CAM++, as Seed-VC uses for style)."""
        fbank = torchaudio.compliance.kaldi.fbank(waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        return self.campplus((fbank - fbank.mean(dim=0, keepdim=True)).unsqueeze(0))

    @torch.no_grad()
    def score(self, ref_id, audio):
        waves_16k = torchaudio.functional.resample(torch.tensor(audio).unsqueeze(0).float(), SR, 16000)
        style = self.refs[ref_id][1]
        speaker = float(torch.nn.functional.cosine_similarity(self.speaker(waves_16k), style).item())
        if not hasattr(self, "squim"):
            self.squim = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model()
        stoi, pesq, si_sdr = (float(v.item()) for v in self.squim(waves_16k))
        return {"speaker": speaker, "stoi": stoi, "pesq": pesq, "si_sdr": si_sdr}

    @torch.no_grad()
    def reference(self, ref_id, audio):
        audio = torch.tensor(audio[: int(SR * MAX_REFERENCE)]).unsqueeze(0).float()
        waves_16k = torchaudio.functional.resample(audio, SR, 16000)
        mel = self.mel_fn(audio)
        style = self.speaker(waves_16k)
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


def lead_in(clip, i, j, max_lead):
    """The earliest start <= i such that clip[start:i] is a gap before the swear."""
    frame = int(0.01 * SR)
    peak = max(rms(clip[k : k + frame]) for k in range(i, max(i + 1, j - frame), frame))
    start = i
    while i - start + frame <= min(max_lead, int(MAX_LEAD * SR)) and start - frame >= 0:
        if rms(clip[start - frame : start]) > peak * 10 ** (-GAP_DB / 20):
            break
        start -= frame
    return start


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
        level = rms(clip[i:j])
        i = lead_in(clip, i, j, req.get("max_lead", 0))
        # Keep the singer's real vocals either side of the new word: the model
        # gets context, and its edge artifacts land outside the word.
        source = np.concatenate([clip[:i], reshape(speak(req["word"]), clip[i:j]), clip[j:]])
        out = vc.convert(req["ref"], source)
        out *= level / rms(out[req["start"] : j])
        np.save(req["out"], out.astype(np.float32))
        return {"sung": True, "start": int(i)}
    if req["op"] == "score":
        return {"scores": [vc.score(req["ref"], np.load(path)) for path in req["paths"]]}
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
