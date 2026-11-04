"""Test bench for the sung fill: score every sung swear in the sample songs.

    clean-stem bench corpus [SONGS...]   # once: cut test cases from the songs
    clean-stem bench run --label NAME    # sing, score, save clips + results
    clean-stem bench compare A B         # what changed between two runs

A case is one swear with a substitute: the mix, vocals and instrumental
around it, plus its song's voice reference. Cases and results live outside
the repo (the audio is copyrighted), in $CLEAN_STEM_BENCH or
~/.cache/clean-stem/bench.

Each run renders original / reversed / sung clips for listening and measures
what we've heard go wrong by ear:

    heard     what Whisper hears at the swear in the cleaned mix; it should
              be the substitute, never a swear
    onset_ms  how late the new word's vowel starts vs the swear's (the
              "jump at the start")
    dip_db    the deepest dip in loudness going into the vowel, beyond any
              the swear itself had (a gap between consonant and vowel)
    level_db  loudness vs the swear, whole word and attack (first 60 ms)
    click_db  a spike of high-frequency energy at the splice edges, beyond
              any the original had there
    speaker   how much the sung word sounds like the artist (speaker
              embedding similarity to the reference), next to the swear's own
    pesq      SQUIM's estimate of the word's quality (1-4.5), next to the swear's
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

SR = 44100
FRAME = 441  # 10 ms
# Seconds of audio kept around each swear.
BEFORE, AFTER = 4.0, 3.0
# The vowel starts at the first voiced frame within this many dB of the peak.
VOWEL_DB = 12
# Frames with fewer zero crossings per sample than this count as voiced:
# vowels (100-1000 Hz) sit far below it, sibilants (4-10 kHz) far above.
VOICED_ZCR = 0.1
ATTACK = 0.06

# A case passes if all of these hold.
LIMITS = {
    "onset_ms": 30,  # |vowel onset error|
    "dip_db": 6,  # extra dip into the vowel
    "level_db": 3,  # |whole-word loudness difference|
    "attack_db": 6,  # |attack loudness difference|
    "click_db": 6,  # extra edge energy
}


def bench_dir() -> Path:
    env = os.environ.get("CLEAN_STEM_BENCH")
    return Path(env) if env else Path.home() / ".cache" / "clean-stem" / "bench"


@dataclass(frozen=True)
class Case:
    name: str  # <song>-<index>
    song: str
    word: str
    substitute: str
    start: float  # swear span, seconds into the clip
    end: float
    max_lead: float


# --- corpus -----------------------------------------------------------------


def build_corpus(songs: list[Path], out: Path, models) -> list[Case]:
    """Cut a case for every swear with a substitute in `songs`."""
    from clean_stem.pipeline import locate, plan_sung
    from clean_stem.splice import Span, _covering_stems
    from clean_stem.sung import build_reference

    out.mkdir(parents=True, exist_ok=True)
    cases = []
    for song in songs:
        located = locate(song, models)
        if isinstance(located, str) or not located.words:
            continue
        sr = models.separator.sample_rate
        if sr != SR:
            raise ValueError(f"stems at {sr} Hz; the bench expects {SR}")
        from clean_stem.audio import decode_stereo

        stereo = decode_stereo(song, sr)
        np.save(out / f"{song.stem}-ref.npy", build_reference(located.stems, [(w.start, w.end) for w in located.words], sr))
        for k, (w, plan) in enumerate(zip(located.words, plan_sung(located.words))):
            if plan.substitute is None:
                continue
            stems = _covering_stems(Span(w.start, w.end), located.stems)
            lo = max(stems.start, w.start - BEFORE)
            hi = min(stems.end, w.end + AFTER)
            window = stems.slice(lo, hi, sr)
            a = int(round(lo * sr))
            mix = stereo[:, a : a + window.vocals.shape[1]]
            case = Case(f"{song.stem}-{k:02d}", song.stem, w.word, plan.substitute, w.start - lo, w.end - lo, min(plan.max_lead, w.start - lo))
            np.savez_compressed(out / f"{case.name}.npz", mix=mix, vocals=window.vocals, instrumental=window.instrumental)
            (out / f"{case.name}.json").write_text(json.dumps(asdict(case)))
            cases.append(case)
    return cases


def load_cases(corpus: Path) -> list[Case]:
    return [Case(**json.loads(p.read_text())) for p in sorted(corpus.glob("*.json"))]


# --- metrics ----------------------------------------------------------------


def levels(x: np.ndarray) -> np.ndarray:
    """10 ms RMS levels of mono `x`, in dB."""
    n = len(x) // FRAME
    frames = x[: n * FRAME].reshape(n, FRAME).astype(np.float64)
    return 20 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-9)


def voiced(x: np.ndarray) -> np.ndarray:
    n = len(x) // FRAME
    frames = x[: n * FRAME].reshape(n, FRAME)
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)
    return zcr < VOICED_ZCR


def vowel_onset(x: np.ndarray) -> int:
    """Frame where the vowel of mono word `x` starts."""
    lv = levels(x)
    frames = np.flatnonzero(voiced(x) & (lv >= lv.max() - VOWEL_DB))
    return int(frames[0]) if len(frames) else 0


def dip_into(lv: np.ndarray, until: int) -> float:
    """Deepest dip (dB) in `lv[:until]`: how far a frame falls below the
    quieter of the loudest frames either side of it."""
    worst = 0.0
    for k in range(1, max(1, until - 1)):
        worst = max(worst, min(lv[:k].max(), lv[k + 1 : until + 1].max()) - lv[k])
    return worst


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-9))


def edge_energy_db(mix: np.ndarray, at: int) -> float:
    """High-frequency (second difference) energy in 6 ms around sample `at`."""
    half = int(0.003 * SR)
    seg = mix[max(0, at - half) : at + half]
    return rms_db(np.diff(seg, n=2)) if len(seg) > 2 else -180.0


def spike_db(mix: np.ndarray, at: int) -> float:
    """How far high-frequency energy at `at` stands above the 60 ms around it:
    a click is a local spike, unlike a new word that simply sounds different."""
    around = [edge_energy_db(mix, at + int(d * SR)) for d in (-0.03, -0.02, -0.01, 0.01, 0.02, 0.03)]
    return edge_energy_db(mix, at) - float(np.median(around))


def signal_metrics(case: Case, original: dict, sung: dict, word_start: float) -> dict:
    """Timing, loudness and edge metrics; `original` / `sung` hold mono
    "mix" and "vocals" for the clip."""
    i, j = int(case.start * SR), int(case.end * SR)
    old_word, new_word = original["vocals"][i:j], sung["vocals"][i:j]
    onset_old, onset_new = vowel_onset(old_word), vowel_onset(new_word)
    # The dip window runs from wherever the new word starts to the vowel.
    lead = int(round((case.start - word_start) * SR))
    lv_old = levels(original["vocals"][i - lead : j])
    lv_new = levels(sung["vocals"][i - lead : j])
    until = lead // FRAME + onset_old + int(0.08 / 0.01)
    attack = slice(i, i + int(ATTACK * SR))
    edges = (i - lead, j)
    return {
        "onset_ms": int(onset_new - onset_old) * 10,
        "dip_db": round(float(dip_into(lv_new, until) - dip_into(lv_old, until)), 1),
        "level_db": round(float(rms_db(new_word) - rms_db(old_word)), 1),
        "attack_db": round(rms_db(sung["vocals"][attack]) - rms_db(original["vocals"][attack]), 1),
        "click_db": round(
            float(max(spike_db(sung["mix"], e) - spike_db(original["mix"], e) for e in edges)), 1
        ),
    }


def heard_at(transcriber, mix: np.ndarray, start: float, end: float) -> str:
    """What Whisper hears over [start, end] (s) of mono 44.1 kHz `mix`."""
    from clean_stem.audio import resample

    words = transcriber.words(resample(mix[None, :], SR, 16000)[0].astype(np.float32))
    return " ".join(w.text for w in words if w.start < end + 0.15 and w.end > start - 0.15)


def heard_metrics(heard: str, substitute: str) -> dict:
    from clean_stem.profanity import is_profane

    letters = lambda s: "".join(c for c in s.lower() if c.isalpha())  # noqa: E731
    target = letters(substitute)
    tokens = heard.split()
    # Best match of the substitute against any run of consecutive heard words.
    best = max(
        (SequenceMatcher(None, letters("".join(tokens[a:b])), target).ratio() for a in range(len(tokens)) for b in range(a + 1, len(tokens) + 1)),
        default=0.0,
    )
    return {"heard": heard, "heard_substitute": bool(best >= 0.75), "heard_swear": any(is_profane(t) for t in tokens)}


def passes(m: dict) -> bool:
    return bool(
        m["heard_substitute"]
        and not m["heard_swear"]
        and abs(m["onset_ms"]) <= LIMITS["onset_ms"]
        and m["dip_db"] <= LIMITS["dip_db"]
        and abs(m["level_db"]) <= LIMITS["level_db"]
        and abs(m["attack_db"]) <= LIMITS["attack_db"]
        and m["click_db"] <= LIMITS["click_db"]
    )


# --- run --------------------------------------------------------------------


def render(case: Case, data: dict, sung_vocal) -> dict:
    """Mono mix and vocals for each of original / reversed / sung."""
    from clean_stem.separate import Stems
    from clean_stem.splice import Span, replace_spans

    length = data["vocals"].shape[1] / SR
    stems = Stems(0.0, length, data["vocals"], data["instrumental"])
    zero = np.zeros_like(data["vocals"])
    vocal_only = Stems(0.0, length, data["vocals"], zero)
    out = {"original": {"mix": data["mix"].mean(0), "vocals": data["vocals"].mean(0)}}
    span = Span(case.start, case.end)
    out["reversed"] = {"mix": replace_spans(data["mix"], SR, [span], [stems], SR).mean(0)}
    if sung_vocal is not None:
        span = Span(min(case.start, sung_vocal.word_start), case.end, sung=sung_vocal)
        out["sung"] = {
            "mix": replace_spans(data["mix"], SR, [span], [stems], SR).mean(0),
            "vocals": replace_spans(data["vocals"], SR, [span], [vocal_only], SR).mean(0),
        }
    return out


def save_mp3(path: Path, mono: np.ndarray) -> None:
    pcm = (np.clip(mono, -1, 1) * 32767).astype("<i2").tobytes()
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-", "-b:a", "160k", str(path)]
    subprocess.run(cmd, input=pcm, check=True)


def run(label: str, corpus: Path, models, only: list[str] | None = None, log=print) -> dict:
    from clean_stem.sung import SungVocal

    cases = [c for c in load_cases(corpus) if not only or any(c.name.startswith(o) for o in only)]
    out = corpus / "results" / label
    out.mkdir(parents=True, exist_ok=True)
    sung, transcriber = models.sung, models.transcriber
    refs: dict[str, str] = {}
    results = {}
    for case in cases:
        data = dict(np.load(corpus / f"{case.name}.npz"))
        if case.song not in refs:
            refs[case.song] = sung.set_reference(np.load(corpus / f"{case.song}-ref.npy"))
        from clean_stem.separate import Stems

        length = data["vocals"].shape[1] / SR
        stems = Stems(0.0, length, data["vocals"], data["instrumental"])
        reply = sung.convert(refs[case.song], stems, case.start, case.end, case.substitute, SR, case.max_lead)
        vocal = reply if isinstance(reply, SungVocal) else None
        clips = render(case, data, vocal)
        for name, clip in clips.items():
            save_mp3(out / f"{case.name}-{name}.mp3", clip["mix"])
        result = {"word": case.word, "substitute": case.substitute}
        if vocal is None:
            result["skipped"] = reply
        else:
            i, j = int(case.start * SR), int(case.end * SR)
            result.update(heard_metrics(heard_at(transcriber, clips["sung"]["mix"], case.start, case.end), case.substitute))
            result.update(signal_metrics(case, clips["original"], clips["sung"], vocal.word_start))
            pad = int(0.1 * SR)
            new, old = sung.score(refs[case.song], [clips["sung"]["vocals"][i - pad : j + pad], clips["original"]["vocals"][i - pad : j + pad]])
            result.update({
                "speaker": round(new["speaker"], 3), "speaker_orig": round(old["speaker"], 3),
                "pesq": round(new["pesq"], 2), "pesq_orig": round(old["pesq"], 2),
            })
            result["pass"] = passes(result)
        results[case.name] = result
        log(format_row(case.name, result))
    (out / "results.json").write_text(json.dumps(results, indent=2))
    return results


# --- reporting --------------------------------------------------------------

HEADER = f"{'case':22} {'word':>10} -> {'substitute':<14} {'heard':<22} {'onset':>6} {'dip':>5} {'level':>6} {'attack':>6} {'click':>6} {'speaker':>11} {'pesq':>9}  ok"


def format_row(name: str, r: dict) -> str:
    if "skipped" in r:
        return f"{name:22} {r['word']:>10} -> {r['substitute']:<14} (default fill: {r['skipped']})"
    heard = r["heard"][:21] + ("!" if r["heard_swear"] else "")
    return (
        f"{name:22} {r['word']:>10} -> {r['substitute']:<14} {heard:<22} {r['onset_ms']:>5}ms {r['dip_db']:>5} {r['level_db']:>6} "
        f"{r['attack_db']:>6} {r['click_db']:>6} {r['speaker']:>5}/{r['speaker_orig']:<5} {r['pesq']:>4}/{r['pesq_orig']:<4}  {'yes' if r['pass'] else 'NO'}"
    )


def summary(results: dict) -> str:
    sung = [r for r in results.values() if "skipped" not in r]
    if not sung:
        return "no sung cases"
    mean = lambda k: np.mean([r[k] for r in sung])  # noqa: E731
    return (
        f"{sum(r['pass'] for r in sung)}/{len(sung)} pass ({len(results) - len(sung)} not sung); "
        f"substitute heard {sum(r['heard_substitute'] for r in sung)}/{len(sung)}, swear heard {sum(r['heard_swear'] for r in sung)}; "
        f"mean |onset| {np.mean([abs(r['onset_ms']) for r in sung]):.0f}ms, dip {mean('dip_db'):.1f}dB, "
        f"speaker {mean('speaker'):.2f} (orig {mean('speaker_orig'):.2f}), pesq {mean('pesq'):.2f} (orig {mean('pesq_orig'):.2f})"
    )


def compare(a: dict, b: dict) -> list[str]:
    lines = []
    for name in sorted(set(a) & set(b)):  # runs may cover different cases
        ra, rb = a[name], b[name]
        if "skipped" in ra and "skipped" in rb:
            continue  # default fill both times
        if "skipped" in ra or "skipped" in rb:
            lines.append(f"{name:22} sung in only one run")
            continue
        diffs = []
        for k in ("onset_ms", "dip_db", "level_db", "attack_db", "click_db", "speaker", "pesq"):
            if ra[k] != rb[k]:
                diffs.append(f"{k} {ra[k]}->{rb[k]}")
        if ra["heard"] != rb["heard"]:
            diffs.append(f"heard {ra['heard']!r}->{rb['heard']!r}")
        status = {(True, True): "", (False, True): "  FIXED", (True, False): "  BROKE", (False, False): ""}[(ra["pass"], rb["pass"])]
        lines.append(f"{name:22} {', '.join(diffs) or 'same'}{status}")
    return lines
