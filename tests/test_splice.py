import numpy as np
import pytest
from mutagen.id3 import ID3, TIT2, TXXX

from clean_stem.separate import Stems
from clean_stem.splice import CROSSFADE, Span, copy_tags, merge_spans, replace_spans

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


def test_merge_windows_and_stems_slice():
    from clean_stem.separate import merge_windows

    assert merge_windows([(5.0, 9.0), (0.0, 3.0), (2.0, 4.0), (9.0, 10.0)]) == [(0.0, 4.0), (5.0, 10.0)]
    vocals = np.arange(10 * RATE, dtype=np.float32)[None].repeat(2, 0)
    stems = Stems(2.0, 12.0, vocals, -vocals)
    part = stems.slice(4.0, 5.0, RATE)
    assert (part.start, part.end, part.vocals.shape) == (4.0, 5.0, (2, RATE))
    assert part.vocals[0, 0] == 2 * RATE and part.instrumental[0, 0] == -2 * RATE


def test_flac_reencode_keeps_bit_depth_and_unmodified_samples(tmp_path):
    import subprocess

    import mutagen

    from clean_stem.audio import decode_stereo
    from clean_stem.splice import encode_like

    src = tmp_path / "src.flac"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anoisesrc=d=2:a=0.3", "-ac", "2", "-ar", "44100",
         "-sample_fmt", "s16", str(src)],
        check=True,
    )
    tags = mutagen.File(src)
    tags["title"] = "noise"
    tags.save()

    out = tmp_path / "out.flac"
    encode_like(src, decode_stereo(src, 44100), 44100, out)

    def pcm(path):
        return subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-"], capture_output=True, check=True).stdout

    assert mutagen.File(out).info.bits_per_sample == 16
    assert mutagen.File(out)["title"] == ["noise"]
    assert pcm(out) == pcm(src)


def test_replace_spans_mixes_in_reversed_vocal():
    audio = np.zeros((1, 10 * RATE), dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, 6 * RATE, dtype=np.float32)[None].repeat(2, 0)  # vocal rising over the window
    window = Stems(start=2.0, end=8.0, vocals=ramp, instrumental=np.zeros((2, 6 * RATE), np.float32))
    out = replace_spans(audio, RATE, [Span(4.0, 5.0)], [window], RATE, reversed_vocal_gain=0.5)
    inside = out[0, int(4.1 * RATE) : int(4.9 * RATE)]
    assert np.all(np.diff(inside) < 0)  # reversed: falling where the vocal rose
    assert 0.1 < inside.max() < 0.25  # at half level (vocal there is ~0.33-0.5)
    silent = replace_spans(audio, RATE, [Span(4.0, 5.0)], [window], RATE, reversed_vocal_gain=0.0)
    assert np.all(silent == 0.0)


@pytest.mark.parametrize("version", [3, 4])
def test_copy_tags_keeps_the_id3_version(tmp_path, version):
    original, dest = tmp_path / "a.mp3", tmp_path / "b.mp3"
    original.write_bytes(b"\xff\xfb\x90\x00" + bytes(413))  # one silent MPEG frame
    dest.write_bytes(original.read_bytes())
    tags = ID3()
    tags.add(TIT2(encoding=3, text="Title"))
    tags.add(TXXX(encoding=3, desc="BARCODE", text="123"))
    tags.save(original, v2_version=version)
    copy_tags(original, dest)
    copied = ID3(dest)
    assert copied.version == (2, version, 0)
    assert copied["TIT2"].text == ["Title"] and copied["TXXX:BARCODE"].text == ["123"]
