"""M4A (AAC and ALAC): re-encoding keeps codec, tags and timing."""

import subprocess

import numpy as np
import pytest
from mutagen.mp4 import MP4, MP4Cover, MP4FreeForm

from clean_stem.audio import decode_stereo
from clean_stem.lyrics import read_embedded_lyrics
from clean_stem.splice import ITUNSMPB, _gapless, encode_like

RATE = 44100
LYRICS = "Line one\r\nwith a damn word\r\n"
# What an iTunes encode carries (2112 priming). Wrong for an ffmpeg encode
# (1024), and ffmpeg honours it over the edit list, so copying it shifts audio.
ITUNES_SMPB = " 00000000 00000840 000001CA 0000000000015888" + " 00000000" * 8


def noise(seconds: float, seed: int = 0) -> np.ndarray:
    """A sweep (which AAC codes well) plus a little noise (so correlation has one clear peak)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    sweep = 0.3 * np.sin(2 * np.pi * (200 * t + 300 * t**2))
    return (sweep + 0.02 * rng.standard_normal((2, t.size))).astype(np.float32)


def make_m4a(path, codec_args, smpb=None):
    """A short m4a with the sort of tags a tagged library file has."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "f32le", "-ac", "2", "-ar", str(RATE), "-i", "-",
           *codec_args, "-f", "ipod", str(path)]  # fmt: skip
    subprocess.run(cmd, input=noise(2.0).T.tobytes(), check=True, capture_output=True)
    tags = MP4(path)
    tags["\xa9nam"] = ["Song"]
    tags["\xa9ART"] = ["Artist"]
    tags["\xa9too"] = ["iTunes 12.13"]
    tags["trkn"] = [(3, 12)]
    tags["cpil"] = False
    tags["\xa9lyr"] = [LYRICS]
    tags["covr"] = [MP4Cover(b"\x89PNG\r\n\x1a\nfake art", imageformat=MP4Cover.FORMAT_PNG)]
    tags["----:com.apple.iTunes:iTunNORM"] = [MP4FreeForm(b" 00000318 0000030C 00001E9C")]
    if smpb is True:  # correct gapless info, as an Apple encoder writes it
        smpb = " 00000000 {:08X} {:08X} {:016X}".format(*_gapless(path)) + " 00000000" * 8
    if smpb:
        tags[ITUNSMPB] = [MP4FreeForm(smpb.encode())]
    tags.save()
    return path


def lag(a: np.ndarray, b: np.ndarray, max_lag: int = 2500) -> int:
    """Offset of b relative to a, by cross-correlation."""
    n = min(a.shape[1], b.shape[1]) - 2 * max_lag
    ref = a[0, max_lag : max_lag + n]
    scores = [np.dot(ref, b[0, max_lag + k : max_lag + k + n]) for k in range(-max_lag, max_lag + 1)]
    return int(np.argmax(scores)) - max_lag


@pytest.mark.parametrize(
    "codec_args,smpb",
    [
        (["-c:a", "aac", "-b:a", "128k"], True),
        (["-c:a", "alac", "-sample_fmt", "s16p"], None),
        (["-c:a", "alac", "-sample_fmt", "s32p"], None),
    ],
    ids=["aac", "alac16", "alac24"],
)
def test_encode_like_m4a_keeps_codec_tags_and_timing(tmp_path, codec_args, smpb):
    original = make_m4a(tmp_path / "song.m4a", codec_args, smpb)
    audio = decode_stereo(original, RATE)
    dest = tmp_path / "out.m4a"
    encode_like(original, audio, RATE, dest)

    src, out = MP4(original), MP4(dest)
    assert out.info.codec == src.info.codec
    assert out.info.sample_rate == RATE and out.info.channels == 2
    if src.info.codec == "alac":
        assert out.info.bits_per_sample == src.info.bits_per_sample
    else:
        assert abs(out.info.bitrate - src.info.bitrate) / src.info.bitrate < 0.1

    # Every tag survives, byte for byte, except the gapless info.
    assert sorted(out.tags) == sorted(src.tags)
    for key in src.tags:
        if key != ITUNSMPB:
            assert out.tags[key] == src.tags[key], key
    assert out.tags["covr"][0].imageformat == MP4Cover.FORMAT_PNG
    assert read_embedded_lyrics(dest).text == "Line one\nwith a damn word"

    # Decodes to the same audio: no shift, no length change.
    decoded = decode_stereo(dest, RATE)
    assert lag(audio, decoded) == 0
    assert abs(decoded.shape[1] - audio.shape[1]) <= 0.005 * RATE
    if src.info.codec == "alac":
        assert np.array_equal(decoded, audio)
    else:
        # Apple players trim to the edit list's length; it must not grow by the old padding.
        assert abs(_gapless(dest)[2] - _gapless(original)[2]) <= 0.001 * RATE
        n = min(decoded.shape[1], audio.shape[1])
        assert np.corrcoef(audio[0, :n], decoded[0, :n])[0, 1] > 0.9


def test_itunsmpb_is_regenerated_for_the_new_encode(tmp_path):
    original = make_m4a(tmp_path / "song.m4a", ["-c:a", "aac", "-b:a", "128k"], ITUNES_SMPB)
    audio = noise(1.5, seed=1)
    dest = tmp_path / "out.m4a"
    encode_like(original, audio, RATE, dest)

    fields = bytes(MP4(dest).tags[ITUNSMPB][0]).decode().split()
    priming, padding, length = (int(f, 16) for f in fields[1:4])
    assert priming in (1024, 2048)  # ffmpeg aac / libfdk_aac; not the original's 2112
    assert (priming + padding + length) % 1024 == 0
    assert abs(length - audio.shape[1]) <= 0.001 * RATE  # edit list is in milliseconds
    assert lag(audio, decode_stereo(dest, RATE)) == 0


def test_batch_picks_up_m4a_but_not_movies_drm_or_raw_aac(tmp_path):
    from clean_stem.batch import _is_candidate

    assert _is_candidate(tmp_path / "Song.M4A")
    for name in ("movie.mp4", "protected.m4p", "raw.aac"):
        assert not _is_candidate(tmp_path / name)


def test_read_embedded_lyrics_from_m4a(tmp_path):
    song = make_m4a(tmp_path / "song.m4a", ["-c:a", "aac", "-b:a", "96k"])
    lyrics = read_embedded_lyrics(song)
    assert lyrics.text == "Line one\nwith a damn word"
    assert lyrics.source == "tag:\xa9lyr"

    tags = MP4(song)
    del tags["\xa9lyr"]
    tags.save()
    assert read_embedded_lyrics(song) is None
