import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from clean_stem import sung
from clean_stem.align import FlaggedWord
from clean_stem.pipeline import Located, Models, sing_spans
from clean_stem.separate import Stems
from clean_stem.splice import Span, merge_spans, replace_spans
from clean_stem.sung import SeedVCError, SungFill, SungVocal, build_reference, substitute

RATE = 1000


def test_substitute():
    assert substitute("fucker") == "sucker"
    assert substitute("Fuck") == "fudge"
    assert substitute("shit's") == "shoot's"
    assert substitute("shit’s") == "shoot's"
    assert substitute("goddamn") == "god dang"
    assert substitute("nigga") is None  # slurs keep the default fill
    assert substitute("bitchy") is None
    assert substitute("famefucker") == "fame sucker"  # compounds
    assert substitute("abitch") is None  # too short a prefix to be a word


def test_build_reference_skips_swears_and_silence():
    vocals = np.full((2, 10 * RATE), 0.5, np.float32)
    vocals[:, : 2 * RATE] = 0.0  # silent first 2 s
    stems = [Stems(0.0, 10.0, vocals, np.zeros_like(vocals))]
    ref = build_reference(stems, [(5.0, 6.0)], RATE)
    # 10 s minus 2 s of silence minus the swear and its guard either side.
    assert len(ref) == pytest.approx((8 - 1 - 2 * sung.REFERENCE_GUARD) * RATE, abs=0.1 * RATE)
    assert np.all(ref == 0.5)


def test_build_reference_stops_at_reference_length():
    vocals = np.full((2, 30 * RATE), 0.5, np.float32)
    ref = build_reference([Stems(0.0, 30.0, vocals, vocals)], [], RATE)
    assert len(ref) == sung.REFERENCE_LENGTH * RATE


def test_replace_spans_uses_sung_vocal():
    audio = np.ones((2, 10 * RATE), dtype=np.float32)
    window = Stems(2.0, 8.0, vocals=np.full((2, 6 * RATE), 0.75, np.float32),
                   instrumental=np.full((2, 6 * RATE), 0.25, np.float32))
    vocal = SungVocal(3.5, np.full(2 * RATE, 0.5, np.float32), "sucker", 4.0)
    out = replace_spans(audio, RATE, [Span(4.0, 5.0, sung=vocal)], [window], RATE)
    assert np.allclose(out[:, 4 * RATE : 5 * RATE], 0.75)  # instrumental + sung vocal
    assert np.all(out[:, : int(3.9 * RATE)] == 1.0) and np.all(out[:, int(5.1 * RATE) :] == 1.0)


def test_merged_spans_lose_sung_vocal():
    vocal = SungVocal(0.0, np.zeros(10), "sucker", 1.0)
    merged = merge_spans([Span(1.0, 2.0, sung=vocal), Span(1.5, 3.0)])
    assert merged == [Span(1.0, 3.0)] and merged[0].sung is None


class FakeSung:
    def __init__(self, replies):
        self.replies = replies
        self.references = 0
        self.asked = []
        self.leads = []

    def set_reference(self, audio):
        self.references += 1
        return "ref"

    def convert(self, ref_id, stems, start, end, word, sample_rate, max_lead=0.0):
        self.asked.append(word)
        self.leads.append(round(max_lead, 3))
        reply = self.replies[word]
        if isinstance(reply, Exception):
            raise reply
        # Start the new word as early as allowed.
        return reply if isinstance(reply, str) else SungVocal(start - 0.5, np.zeros(10, np.float32), word, start - max_lead)


def located(words):
    vocals = np.full((2, 60 * sung.SAMPLE_RATE // 10), 0.5, np.float32)
    stems = [Stems(0.0, 6.0, vocals, vocals)]
    return Located(track=None, lyrics_source=None, words=[FlaggedWord(w, -1, s, e, "heard", "vocals") for w, s, e in words], stems=stems)


def test_sing_spans_falls_back_per_word():
    models = Models()
    models.sung = FakeSung({"sucker": True, "dang": "too short (0.30s)", "shoot": SeedVCError("boom")})
    loc = located([
        ("fucker", 0.5, 1.0),
        ("damn", 1.5, 1.8),
        ("nigga", 2.5, 2.9),
        ("shit", 3.5, 4.0),
        ("bitch", 5.0, 5.4),
        ("fuck", 5.42, 5.6),  # right after "bitch": spliced as one span
    ])
    spans, fills = sing_spans(loc, models, sung.SAMPLE_RATE)
    assert fills == [
        "sung:sucker",
        "reversed (too short (0.30s))",
        "reversed (no substitute)",
        "reversed (conversion failed: boom)",
        "reversed (next to another word)",
        "reversed (next to another word)",
    ]
    assert spans[0].sung is not None and all(s.sung is None for s in spans[1:])
    assert models.sung.references == 1
    assert models.sung.asked == ["sucker", "dang", "shoot"]
    # Each may start early, but stays clear of the word before (merge gap 0.06 + 0.01).
    assert models.sung.leads == [0.43, 0.43, 0.53]
    assert spans[0].start == pytest.approx(0.07)


def test_sing_spans_marks_compounds_with_lead():
    models = Models()
    models.sung = FakeSung({"fame sucker": True})
    spans, fills = sing_spans(located([("famefucker", 2.0, 2.6)]), models, sung.SAMPLE_RATE)
    assert fills == ["sung:fame sucker"] and spans[0].end == 2.6


def test_sing_spans_needs_44k_stems():
    models = Models()
    models.sung = FakeSung({})
    spans, fills = sing_spans(located([("fucker", 0.5, 1.0)]), models, 48000)
    assert fills == ["reversed (stems at 48000 Hz)"] and spans[0].sung is None


def test_sung_fill_checks_setup(tmp_path):
    with pytest.raises(SeedVCError, match="not a Seed-VC checkout"):
        SungFill(tmp_path, python=Path(sys.executable))
    (tmp_path / "inference.py").write_text("")
    with pytest.raises(SeedVCError, match="no Seed-VC Python"):
        SungFill(tmp_path, python=tmp_path / "missing-python")


def test_sung_fill_talks_to_worker(tmp_path, monkeypatch):
    """The JSON-lines protocol, against a stand-in worker."""
    (tmp_path / "inference.py").write_text("")
    worker = tmp_path / "worker.py"
    worker.write_text(textwrap.dedent("""
        import json, sys
        import numpy as np
        print(json.dumps({"ready": True}), flush=True)
        for line in sys.stdin:
            req = json.loads(line)
            if req["op"] == "reference":
                print(json.dumps({}), flush=True)
            elif req["word"] == "dang":
                print(json.dumps({"sung": False, "reason": "too short"}), flush=True)
            elif req["word"] == "shoot":
                print(json.dumps({"error": "boom"}), flush=True)
            else:
                clip = np.load(req["path"])
                np.save(req["out"], clip * 2)
                print(json.dumps({"sung": True}), flush=True)
    """))
    monkeypatch.setattr(sung, "WORKER", worker)
    fill = SungFill(tmp_path, python=Path(sys.executable))
    try:
        ref = fill.set_reference(np.zeros(10, np.float32))
        stems = Stems(10.0, 16.0, np.full((2, 6 * RATE), 0.25, np.float32), np.zeros((2, 6 * RATE), np.float32))
        vocal = fill.convert(ref, stems, 12.0, 12.5, "sucker", RATE)
        assert vocal.start == pytest.approx(12.0 - sung.CONTEXT)
        assert len(vocal.audio) == int((0.5 + 2 * sung.CONTEXT) * RATE)
        assert np.allclose(vocal.audio, 0.5)
        assert fill.convert(ref, stems, 12.0, 12.5, "dang", RATE) == "too short"
        with pytest.raises(SeedVCError, match="boom"):
            fill.convert(ref, stems, 12.0, 12.5, "shoot", RATE)
    finally:
        fill.close()
