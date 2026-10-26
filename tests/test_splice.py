import numpy as np

from clean_stem.separate import Stems
from clean_stem.splice import CROSSFADE, Span, merge_spans, replace_spans

RATE = 1000  # low rate keeps the arithmetic readable


def test_merge_spans_joins_overlapping_and_near_spans():
    spans = [Span(5.0, 6.0), Span(1.0, 2.0), Span(1.5, 2.5), Span(2.55, 3.0)]
    assert merge_spans(spans) == [Span(1.0, 3.0), Span(5.0, 6.0)]


def test_replace_spans_swaps_in_instrumental_with_crossfades():
    audio = np.ones((2, 10 * RATE), dtype=np.float32)  # "vocals + music" = 1.0
    window = Stems(start=2.0, end=8.0, vocals=np.zeros((2, 6 * RATE), np.float32),
                   instrumental=np.full((2, 6 * RATE), 0.25, np.float32))
    out = replace_spans(audio, RATE, [Span(4.0, 5.0)], [window], RATE)

    fade = int(CROSSFADE * RATE)
    assert np.all(out[:, : 4 * RATE - fade] == 1.0)  # untouched before
    assert np.allclose(out[:, 4 * RATE : 5 * RATE], 0.25)  # word fully replaced
    assert np.all(out[:, 5 * RATE + fade :] == 1.0)  # untouched after
    ramp = out[0, 4 * RATE - fade : 4 * RATE]
    assert np.all(np.diff(ramp) <= 0) and ramp[0] > 0.9  # smooth fade down into the span
    assert np.all(audio == 1.0)  # input not modified


def test_replace_spans_resamples_stems_and_handles_mono():
    audio = np.ones((1, 10 * RATE), dtype=np.float32)
    window = Stems(start=0.0, end=10.0, vocals=np.zeros((2, 20 * RATE), np.float32),
                   instrumental=np.full((2, 20 * RATE), 0.5, np.float32))
    out = replace_spans(audio, RATE, [Span(3.0, 4.0)], [window], 2 * RATE)
    assert out.shape == (1, 10 * RATE)
    assert np.allclose(out[0, int(3.2 * RATE) : int(3.8 * RATE)], 0.5, atol=1e-3)
