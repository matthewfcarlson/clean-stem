"""Ogg Vorbis and Opus: re-encoding keeps codec, tags, art, length and timing."""

import base64
import json
import subprocess

import mutagen
import numpy as np
import pytest
from mutagen.flac import Picture

from clean_stem.audio import decode_stereo
from clean_stem.batch import iter_audio_files
from clean_stem.lyrics import read_embedded_lyrics
from clean_stem.splice import OPUS_SAMPLE_RATE, _ffmpeg_has_encoder, encode_like

# Homebrew's ffmpeg has no libvorbis; its built-in Vorbis encoder is stereo-only.
VORBIS = ["-c:a", "libvorbis"] if _ffmpeg_has_encoder("libvorbis") else ["-c:a", "vorbis", "-strict", "-2"]
needs_libvorbis = pytest.mark.skipif(not _ffmpeg_has_encoder("libvorbis"), reason="ffmpeg lacks libvorbis")

LYRICS = "First line\nSecond line"
COVER = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4  # any bytes will do


def make_song(path, rate, channels=2, seconds=3, lyrics_key="LYRICS"):
    """A short pink-noise clip (sharp correlation peak), tagged like a real song."""
    noise = f"anoisesrc=d={seconds}:c=pink:r={rate}:a=0.2:seed=7"
    codec = ["-c:a", "libopus", "-b:a", "96k"] if path.suffix == ".opus" else VORBIS
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", noise, "-ac", str(channels), *codec, str(path)],
        check=True,
    )
    picture = Picture()
    picture.type, picture.mime, picture.desc, picture.data = 3, "image/png", "Front", COVER
    picture.width = picture.height = 16
    song = mutagen.File(path)
    song.tags.clear()
    song.tags.extend([
        ("TITLE", "Test Song"),
        ("ARTIST", "First Artist"),
        ("Artist", "Second Artist"),  # repeated key, odd case: kept as is
        ("ALBUM", "Ünïcode Album"),
        (lyrics_key, LYRICS),
        ("METADATA_BLOCK_PICTURE", base64.b64encode(picture.write()).decode()),
        ("REPLAYGAIN_TRACK_GAIN", "-6.50 dB"),
    ])  # fmt: skip
    song.save()
    return path


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name,sample_rate,channels",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    return json.loads(out)["streams"][0]


def lag(a, b):
    """Samples by which `b` trails `a` (cross-correlation peak)."""
    n = len(a) + len(b)
    corr = np.fft.irfft(np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n)), n)
    k = int(np.argmax(corr))
    return k if k < n // 2 else k - n


@pytest.mark.parametrize(
    "name,rate,channels,codec",
    [
        ("song.ogg", 44100, 2, "vorbis"),
        pytest.param("song.ogg", 48000, 1, "vorbis", marks=needs_libvorbis),
        ("song.opus", 48000, 2, "opus"),
        ("song.opus", 48000, 1, "opus"),
        ("song.opus", 44100, 2, "opus"),  # Opus still decodes at 48 kHz
    ],
)
def test_encode_like_round_trip(tmp_path, name, rate, channels, codec):
    original = make_song(tmp_path / name, rate, channels)
    info = mutagen.File(original).info
    file_rate = getattr(info, "sample_rate", None) or OPUS_SAMPLE_RATE  # as pipeline.clean does
    assert file_rate == (OPUS_SAMPLE_RATE if codec == "opus" else rate)

    audio = decode_stereo(original, file_rate)[:channels]
    dest = tmp_path / f"out{original.suffix}"
    encode_like(original, audio, file_rate, dest)

    stream = probe(dest)
    assert stream["codec_name"] == codec
    assert int(stream["sample_rate"]) == file_rate
    assert stream["channels"] == channels

    # Tags identical, in order, including art and lyrics; no ENCODER tag added.
    src, out = mutagen.File(original), mutagen.File(dest)
    assert list(out.tags) == list(src.tags)
    picture = Picture(base64.b64decode(out.tags["metadata_block_picture"][0]))
    assert (picture.type, picture.mime, picture.data) == (3, "image/png", COVER)
    assert read_embedded_lyrics(dest).text == LYRICS
    assert abs(out.info.length - src.info.length) < 0.005

    # Decoded audio lines up with the original sample for sample.
    before, after = decode_stereo(original, file_rate)[0], decode_stereo(dest, file_rate)[0]
    assert abs(len(after) - len(before)) <= 0.005 * file_rate
    assert lag(before, after) == 0
    n = min(len(before), len(after))
    assert np.corrcoef(before[:n], after[:n])[0, 1] > 0.9

    if codec == "opus":  # bitrate matched to the original's average
        assert abs(out.info.bitrate - src.info.bitrate) / src.info.bitrate < 0.25


@pytest.mark.parametrize("name,key", [("song.ogg", "LYRICS"), ("song.opus", "UNSYNCEDLYRICS")])
def test_read_embedded_lyrics_from_vorbis_comments(tmp_path, name, key):
    song = make_song(tmp_path / name, 48000, seconds=1, lyrics_key=key)
    lyrics = read_embedded_lyrics(song)
    assert lyrics.text == LYRICS
    assert lyrics.source == f"tag:{key}"


def test_batch_finds_ogg_and_opus(tmp_path):
    for name in ("a.ogg", "b.opus", "c.OGG", "d.wav"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in iter_audio_files([tmp_path])] == ["a.ogg", "b.opus", "c.OGG"]
