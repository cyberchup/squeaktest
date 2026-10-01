from __future__ import annotations

import contextlib
import struct
from pathlib import Path

import numpy as np
import pytest
from conftest import TONE_HZ, TONE_SECONDS

from squeaktest.audio import (
    TARGET_SAMPLE_RATE,
    AudioError,
    DecodeTimeoutError,
    DurationTooLongError,
    EmptyAudioError,
    FFmpegNotFoundError,
    FileTooLargeError,
    UnsupportedFormatError,
    load_audio_bytes,
    load_audio_file,
    sniff_container,
)
from squeaktest.config import Settings

# An HLS playlist that points at a local file. If ffmpeg were allowed to guess the format,
# this is the kind of input that could make it read files it shouldn't.
HLS_PLAYLIST = (
    b"#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:0\n#EXTINF:10.0,\nfile:///etc/passwd\n#EXT-X-ENDLIST\n"
)


def _wav_header_then(body: bytes) -> bytes:
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body


# --- Sniffing (no ffmpeg needed) ---------------------------------------------------------


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (_wav_header_then(b"fmt "), "wav"),
        (b"fLaC\x00\x00\x00\x22", "flac"),
        (b"OggS\x00\x02", "ogg"),
        (b"ID3\x04\x00\x00", "mp3"),
        (b"\xff\xfb\x90\x64", "mp3"),  # MPEG-1 Layer III frame header
        (b"\x00\x00\x00\x20ftypM4A \x00\x00\x02\x00", "m4a"),
        (b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00", "m4a"),
    ],
)
def test_sniff_recognizes_allowed_containers(header: bytes, expected: str):
    assert sniff_container(header) == expected


@pytest.mark.parametrize(
    "header",
    [
        b"",
        b"\xff\xf1\x50\x80",  # ADTS AAC: sync word, but layer bits 00
        b"RIFF\x00\x00\x00\x00AVI LIST",  # RIFF, but a video container
        b"\x00\x00\x00\x20ftypqt  \x00\x00\x02\x00",  # QuickTime brand
        HLS_PLAYLIST,
        b"MZ\x90\x00",  # a Windows executable
        b"<html>",
    ],
)
def test_sniff_rejects_everything_else(header: bytes):
    assert sniff_container(header) is None


def test_rejection_happens_before_ffmpeg_is_needed():
    # Unrecognized content is rejected by the sniff alone, even with no ffmpeg available.
    settings = Settings(ffmpeg="no-such-ffmpeg", ffprobe="no-such-ffprobe")
    with pytest.raises(UnsupportedFormatError):
        load_audio_bytes(HLS_PLAYLIST, settings)


def test_size_limit_is_checked_first():
    settings = Settings(max_file_bytes=1024, ffmpeg="no-such-ffmpeg")
    with pytest.raises(FileTooLargeError, match="MB limit"):
        load_audio_bytes(b"\x00" * 1025, settings)


def test_size_limit_on_files(tmp_path: Path):
    big = tmp_path / "big.wav"
    big.write_bytes(_wav_header_then(b"\x00" * 2048))
    with pytest.raises(FileTooLargeError):
        load_audio_file(big, Settings(max_file_bytes=1024))


def test_directories_are_not_files(tmp_path: Path):
    with pytest.raises(UnsupportedFormatError, match="regular file"):
        load_audio_file(tmp_path, Settings())


def test_missing_ffmpeg_is_a_setup_error_not_an_input_error():
    settings = Settings(ffmpeg="no-such-ffmpeg", ffprobe="no-such-ffprobe")
    with pytest.raises(FFmpegNotFoundError):
        load_audio_bytes(_wav_header_then(b"fmt "), settings)


# --- Decoding (needs ffmpeg) -------------------------------------------------------------

ACCEPTED = [
    ("wav_pcm16_44k_stereo", "wav", "pcm_s16le"),
    ("wav_mulaw_8k", "wav", "pcm_mulaw"),
    ("flac", "flac", "flac"),
    ("ogg_vorbis", "ogg", "vorbis"),
    ("ogg_opus", "ogg", "opus"),
    ("mp3", "mp3", "mp3"),
    ("m4a_aac", "m4a", "aac"),
]


@pytest.mark.parametrize(("name", "container", "codec"), ACCEPTED)
def test_allowed_formats_decode_to_16k_mono_float32(
    audio_fixtures: dict[str, Path], name: str, container: str, codec: str
):
    audio = load_audio_file(audio_fixtures[name], Settings())
    assert audio.container == container
    assert audio.codec == codec
    assert audio.sample_rate == TARGET_SAMPLE_RATE
    assert audio.samples.dtype == np.float32
    assert audio.samples.ndim == 1
    # Lossy codecs add or trim a few milliseconds of padding.
    assert audio.duration_s == pytest.approx(TONE_SECONDS, abs=0.1)
    assert np.abs(audio.samples).max() > 0.1


@pytest.mark.parametrize("name", ["wav_pcm16_44k_stereo", "wav_mulaw_8k", "mp3"])
def test_resampling_keeps_the_pitch(audio_fixtures: dict[str, Path], name: str):
    audio = load_audio_file(audio_fixtures[name], Settings())
    spectrum = np.abs(np.fft.rfft(audio.samples))
    freqs = np.fft.rfftfreq(audio.samples.size, d=1 / audio.sample_rate)
    assert freqs[spectrum.argmax()] == pytest.approx(TONE_HZ, abs=2)


def test_extension_is_ignored(audio_fixtures: dict[str, Path], tmp_path: Path):
    disguised = tmp_path / "actually-flac.mp3"
    disguised.write_bytes(audio_fixtures["flac"].read_bytes())
    assert load_audio_file(disguised, Settings()).container == "flac"


@pytest.mark.parametrize(
    ("name", "match"),
    [
        ("adts_aac", "not a supported audio file"),
        ("wav_adpcm", "adpcm_ms codec is not supported"),
        ("mp4_video", "video files are not supported"),
    ],
)
def test_disallowed_formats_are_rejected(audio_fixtures: dict[str, Path], name: str, match: str):
    with pytest.raises(UnsupportedFormatError, match=match):
        load_audio_file(audio_fixtures[name], Settings())


@pytest.mark.usefixtures("audio_fixtures")
@pytest.mark.parametrize(
    "data",
    [
        b"fLaC" + bytes(range(256)) * 8,  # right magic, garbage body
        b"OggS" + b"\x00" * 512,
        _wav_header_then(b"junk" * 64),
    ],
)
def test_corrupt_files_are_rejected(data: bytes):
    with pytest.raises(AudioError):
        load_audio_bytes(data, Settings())


def test_header_duration_over_limit_is_rejected(audio_fixtures: dict[str, Path]):
    with pytest.raises(DurationTooLongError, match="1 s limit"):
        load_audio_file(audio_fixtures["flac"], Settings(max_duration_s=1.0))


def test_decoded_length_is_checked_even_if_the_header_hides_it(
    audio_fixtures: dict[str, Path], monkeypatch: pytest.MonkeyPatch
):
    # Simulate a file whose header gives no usable duration, so only the decode step can
    # catch that it's too long.
    monkeypatch.setattr("squeaktest.audio._parse_seconds", lambda _value: None)
    with pytest.raises(DurationTooLongError):
        load_audio_file(audio_fixtures["flac"], Settings(max_duration_s=1.0))


def test_empty_audio_is_rejected(audio_fixtures: dict[str, Path]):
    with pytest.raises(EmptyAudioError):
        load_audio_file(audio_fixtures["wav_empty"], Settings())


def test_decoder_timeout(audio_fixtures: dict[str, Path]):
    with pytest.raises(DecodeTimeoutError):
        load_audio_file(audio_fixtures["flac"], Settings(decode_timeout_s=0.001))


# --- No persistence, no leaks ------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "settings_kwargs"),
    [
        ("mp3", {}),  # success
        ("wav_adpcm", {}),  # rejected by the probe
        ("flac", {"max_duration_s": 1.0}),  # rejected for length
        ("flac", {"decode_timeout_s": 0.001}),  # decoder killed by the timeout
    ],
)
def test_temp_files_are_always_deleted(
    audio_fixtures: dict[str, Path], tmp_path: Path, name: str, settings_kwargs: dict
):
    work = tmp_path / "work"
    work.mkdir()
    settings = Settings(temp_dir=work, **settings_kwargs)
    with contextlib.suppress(AudioError):
        load_audio_file(audio_fixtures[name], settings)
    assert list(work.iterdir()) == []


@pytest.mark.usefixtures("audio_fixtures")
@pytest.mark.parametrize(
    "data",
    [
        b"\x00" * 2048,  # rejected by the sniff
        _wav_header_then(b"junk" * 64),  # rejected by ffprobe/ffmpeg
    ],
)
def test_errors_do_not_leak_the_filename(tmp_path: Path, data: bytes):
    path = tmp_path / "ceo-voicemail-secret.wav"
    path.write_bytes(data)
    with pytest.raises(AudioError) as excinfo:
        load_audio_file(path, Settings())
    assert "secret" not in str(excinfo.value)
    assert "ceo-voicemail" not in str(excinfo.value)
