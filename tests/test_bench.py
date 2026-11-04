import numpy as np

from clean_stem import bench
from clean_stem.bench import SR, compare, dip_into, heard_metrics, spike_db, vowel_onset


def tone(seconds, freq=220.0, level=0.5):
    t = np.arange(int(seconds * SR)) / SR
    return (level * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def hiss(seconds, level=0.1):
    return (level * np.random.default_rng(0).standard_normal(int(seconds * SR))).astype(np.float32)


def test_vowel_onset_skips_sibilant_and_quiet_lead():
    word = np.concatenate([np.zeros(int(0.05 * SR), np.float32), hiss(0.1), tone(0.3)])
    assert vowel_onset(word) == 15  # 10 ms frames: 50 ms silence + 100 ms "s"


def test_dip_into_finds_gap_between_consonant_and_vowel():
    lv = np.array([-20, -20, -35, -10, -10], float)
    assert dip_into(lv, 4) == 15.0
    assert dip_into(np.array([-20, -15, -10, -10], float), 3) == 0.0


def test_heard_metrics():
    assert heard_metrics("famed sucker bleeding", "sucker") == {"heard": "famed sucker bleeding", "heard_substitute": True, "heard_swear": False}
    m = heard_metrics("a god dang", "god dang")
    assert m["heard_substitute"] and not m["heard_swear"]
    m = heard_metrics("My bitch I'm", "witch")
    assert m["heard_swear"]
    assert not heard_metrics("a tank vampire", "god dang")["heard_substitute"]


def test_spike_db_flags_a_click_not_a_steady_sound():
    steady = tone(0.2)
    clicked = steady.copy()
    clicked[int(0.1 * SR)] += 0.8
    at = int(0.1 * SR)
    assert spike_db(clicked, at) > 10
    assert abs(spike_db(steady, at)) < 1


def test_compare_reports_fixes_and_breaks():
    row = {"heard": "x", "onset_ms": 0, "dip_db": 0, "level_db": 0, "attack_db": 0, "click_db": 0, "speaker": 0.5, "pesq": 2.0, "pass": False}
    a = {"c1": row, "c2": {**row, "pass": True}, "c3": {"skipped": "too short"}, "c4": row, "c5": {"skipped": "x"}}
    b = {"c1": {**row, "dip_db": 1.0, "pass": True}, "c2": {**row, "pass": False}, "c3": {"skipped": "too short"}, "c5": row}
    lines = compare(a, b)
    assert lines[0].endswith("dip_db 0->1.0  FIXED")
    assert lines[1].endswith("same  BROKE")
    assert lines[2].endswith("sung in only one run")
    assert len(lines) == 3  # c3 skipped both times, c4 only in one run


def test_passes_uses_limits():
    good = {"heard_substitute": True, "heard_swear": False, "onset_ms": 10, "dip_db": 1, "level_db": 0, "attack_db": 2, "click_db": 0}
    assert bench.passes(good)
    assert not bench.passes({**good, "dip_db": bench.LIMITS["dip_db"] + 1})
    assert not bench.passes({**good, "heard_swear": True})
