"""Load untrusted audio safely: validate by content, enforce limits, decode in a subprocess.

Every input is treated as hostile. The pipeline is:

1. Size check, before reading the whole file.
2. Sniff the container from its magic bytes. The file extension is ignored.
3. Probe with ffprobe, forcing the sniffed demuxer, and check every stream's codec against an
   allowlist.
4. Decode with ffmpeg in a subprocess with a timeout, to 16 kHz mono float32. A separate
   process keeps a decoder crash or exploit out of this process, and can be killed. A crash
   in step 3 or 4 raises DecoderCrashedError, so callers can log it apart.
5. Check the decoded length (headers can lie) and that there is audio at all.

In steps 3 and 4, ffmpeg itself is told which decoders it may open (`-codec_whitelist`): only
the ones for the allowed codecs of the sniffed container. ffprobe opens a decoder for every
stream in the file while probing, so without this, a file could reach any of ffmpeg's decoders
(an image in cover art, an obscure codec in a second stream) before squeaktest's own allowlist
check ran (docs/decisions.md D26).

The bytes only ever exist in a private temporary directory that is deleted whether decoding
succeeds or fails. Error messages never contain filenames or file content, so they are safe
to show users and to log.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess  # nosec B404
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from squeaktest.config import MB, Settings

TARGET_SAMPLE_RATE = 16_000

# Sniffed container -> (ffmpeg demuxer, allowed codecs). Forcing the demuxer stops ffmpeg from
# guessing the format itself, which is how playlist formats (HLS, concat) that can pull in
# other local files would otherwise get through.
_CONTAINERS: dict[str, tuple[str, frozenset[str]]] = {
    "wav": (
        "wav",
        frozenset(
            {
                "pcm_u8",
                "pcm_s16le",
                "pcm_s24le",
                "pcm_s32le",
                "pcm_f32le",
                "pcm_f64le",
                "pcm_mulaw",  # G.711, common in telephony recordings
                "pcm_alaw",
            }
        ),
    ),
    "flac": ("flac", frozenset({"flac"})),
    "ogg": ("ogg", frozenset({"vorbis", "opus"})),
    "mp3": ("mp3", frozenset({"mp3"})),
    "m4a": ("mov", frozenset({"aac", "alac"})),
}

# Allowed codec -> the ffmpeg decoders that may handle it. ffmpeg's whitelist names decoders,
# not codecs: MP3, for one, is decoded by "mp3float". Only ffmpeg's own (native) decoders.
_DECODERS: dict[str, tuple[str, ...]] = {
    "mp3": ("mp3float", "mp3"),
    "aac": ("aac", "aac_fixed"),
    "alac": ("alac",),
    "flac": ("flac",),
    "vorbis": ("vorbis",),
    "opus": ("opus",),
    **{pcm: (pcm,) for pcm in _CONTAINERS["wav"][1]},
}

# ffmpeg's message when the whitelist stops it opening a decoder.
_REFUSED_DECODER = re.compile(rb"Codec \(([A-Za-z0-9_]+)\) not on whitelist")


def decoder_whitelist(container: str) -> str:
    """The `-codec_whitelist` value for a sniffed container: its allowed codecs' decoders."""
    codecs = _CONTAINERS[container][1]
    return ",".join(sorted({d for codec in codecs for d in _DECODERS[codec]}))


# ISO base media "ftyp" brands seen on M4A files. Generic brands (isom, mp42) are also used by
# MP4 video, which the probe step rejects.
_M4A_BRANDS = frozenset({b"M4A ", b"M4B ", b"mp41", b"mp42", b"isom", b"iso2", b"dash"})

ALLOWED_FORMATS = "wav, mp3, flac, ogg, m4a"


class AudioError(Exception):
    """The input was rejected. The message is safe to show: no filenames, no content."""


class FileTooLargeError(AudioError):
    pass


class UnsupportedFormatError(AudioError):
    pass


class DurationTooLongError(AudioError):
    pass


class DecodeError(AudioError):
    pass


class DecodeTimeoutError(DecodeError):
    pass


class DecoderCrashedError(DecodeError):
    """ffprobe or ffmpeg was killed instead of exiting, so it crashed rather than refusing the file.

    A crafted file trying to exploit a decoder looks like this, so the web app logs it apart
    from ordinary rejections. The message is the one any file that can't be read gets: the
    uploader learns nothing from it that would help tune an exploit.
    """

    def __init__(self, message: str, step: str, returncode: int) -> None:
        super().__init__(message)
        self.step = step
        self.returncode = returncode


class EmptyAudioError(AudioError):
    pass


class FFmpegNotFoundError(RuntimeError):
    """ffmpeg or ffprobe is missing. A setup problem, not a problem with the input."""


@dataclass(frozen=True, eq=False)
class Audio:
    """Decoded audio: mono float32 samples at TARGET_SAMPLE_RATE."""

    samples: np.ndarray
    sample_rate: int
    container: str
    codec: str

    @property
    def duration_s(self) -> float:
        return self.samples.size / self.sample_rate


def sniff_container(header: bytes) -> str | None:
    """Identify the container from the first bytes of a file, or return None."""
    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "wav"
    if header[:4] == b"fLaC":
        return "flac"
    if header[:4] == b"OggS":
        return "ogg"
    if header[4:8] == b"ftyp" and header[8:12] in _M4A_BRANDS:
        return "m4a"
    if header[:3] == b"ID3" or _is_mp3_frame_header(header):
        return "mp3"
    return None


def _is_mp3_frame_header(header: bytes) -> bool:
    # An 11-bit frame sync followed by layer bits 01 (Layer III). Raw ADTS AAC also starts
    # with a sync word but has layer bits 00, so it does not match.
    return (
        len(header) >= 2
        and header[0] == 0xFF
        and (header[1] & 0xE0) == 0xE0
        and (header[1] & 0x06) == 0x02
    )


def load_audio_file(path: str | os.PathLike[str], settings: Settings | None = None) -> Audio:
    """Validate and decode an audio file from disk."""
    settings = settings if settings is not None else Settings.from_env()
    file = Path(path)
    if not file.is_file():
        raise UnsupportedFormatError("input is not a regular file")
    size = file.stat().st_size
    if size > settings.max_file_bytes:
        raise FileTooLargeError(_too_large_message(settings))
    with file.open("rb") as f:
        # Read one byte more than the file's size: Python allocates the full requested length
        # up front, so asking for the 25 MB limit on every small file wastes memory. The extra
        # byte shows whether the file grew after the size check.
        data = f.read(size + 1)
    if len(data) > size:
        raise UnsupportedFormatError("the file changed while it was being read")
    return load_audio_bytes(data, settings)


def load_audio_bytes(data: bytes, settings: Settings | None = None) -> Audio:
    """Validate and decode audio held in memory, such as an upload."""
    settings = settings if settings is not None else Settings.from_env()
    if len(data) > settings.max_file_bytes:
        raise FileTooLargeError(_too_large_message(settings))
    container = sniff_container(data[:16])
    if container is None:
        raise UnsupportedFormatError(f"not a supported audio file (allowed: {ALLOWED_FORMATS})")
    ffmpeg, ffprobe = _find_tools(settings)
    demuxer, allowed_codecs = _CONTAINERS[container]

    # The input is written under a fixed name (never the user's filename) in a private
    # directory. The context manager deletes it on the way out, including on errors.
    with tempfile.TemporaryDirectory(prefix="squeaktest-", dir=settings.temp_dir) as tmp:
        source = Path(tmp) / "input"
        source.write_bytes(data)
        codec = _probe(ffprobe, source, container, demuxer, allowed_codecs, settings)
        samples = _decode(ffmpeg, source, container, demuxer, settings)
    return Audio(samples=samples, sample_rate=TARGET_SAMPLE_RATE, container=container, codec=codec)


def check_tools(settings: Settings) -> None:
    """Raise FFmpegNotFoundError now, rather than on the first file (used at server start)."""
    _find_tools(settings)


def _find_tools(settings: Settings) -> tuple[str, str]:
    ffmpeg = shutil.which(settings.ffmpeg)
    ffprobe = shutil.which(settings.ffprobe)
    if ffmpeg is None or ffprobe is None:
        raise FFmpegNotFoundError(
            "ffmpeg and ffprobe are required. Install them or point SQUEAKTEST_FFMPEG and "
            "SQUEAKTEST_FFPROBE at the executables."
        )
    return ffmpeg, ffprobe


def _probe(
    ffprobe: str,
    source: Path,
    container: str,
    demuxer: str,
    allowed_codecs: frozenset[str],
    settings: Settings,
) -> str:
    """Check every stream's codec, and reject early if the header says it's too long."""
    command = [
        ffprobe,
        "-v", "error",
        "-protocol_whitelist", "file",
        "-codec_whitelist", decoder_whitelist(container),
        "-f", demuxer,
        "-show_entries", "stream=codec_type,codec_name:format=duration",
        "-of", "json",
        str(source),
    ]  # fmt: skip
    result = _run(command, settings, step="probe")
    unparsable = f"file looks like {container} but could not be parsed"
    _check_not_crashed(result, "probe", unparsable)
    if result.returncode != 0:
        refused = _REFUSED_DECODER.search(result.stderr)
        if refused:
            # A stream needs a decoder that isn't allowed: an unsupported codec, cover art,
            # video, or an extra stream. ffmpeg refused to open it.
            raise UnsupportedFormatError(
                _codec_message(container, refused.group(1).decode("ascii"), allowed_codecs)
            )
        raise UnsupportedFormatError(unparsable)
    try:
        info = json.loads(result.stdout)
    except ValueError:
        raise UnsupportedFormatError(unparsable) from None

    streams = info.get("streams", [])
    if not any(s.get("codec_type") == "audio" for s in streams):
        raise UnsupportedFormatError("no audio stream found")
    # The whitelist already refused other decoders; this also catches streams ffmpeg has no
    # decoder for at all, which it reports without opening anything.
    for stream in streams:
        codec = str(stream.get("codec_name", ""))
        if stream.get("codec_type") != "audio" or codec not in allowed_codecs:
            raise UnsupportedFormatError(_codec_message(container, codec, allowed_codecs))
    codec = str(streams[0].get("codec_name"))

    duration = _parse_seconds(info.get("format", {}).get("duration"))
    if duration is not None and duration > settings.max_duration_s:
        raise DurationTooLongError(_too_long_message(settings))
    return codec


def _decode(
    ffmpeg: str, source: Path, container: str, demuxer: str, settings: Settings
) -> np.ndarray:
    """Decode the first audio stream to mono float32 at TARGET_SAMPLE_RATE."""
    # -t caps the output just past the limit. A file whose header understates its length
    # still can't make us decode, or hold in memory, more than that.
    cap_s = settings.max_duration_s + 1.0
    command = [
        ffmpeg,
        "-nostdin", "-hide_banner", "-loglevel", "error",
        "-threads", "1",
        "-protocol_whitelist", "file",
        "-codec_whitelist", decoder_whitelist(container),  # applies to this input's decoders
        "-f", demuxer,
        "-i", str(source),
        "-map", "0:a:0", "-vn", "-sn", "-dn",
        "-ac", "1",
        "-ar", str(TARGET_SAMPLE_RATE),
        "-t", f"{cap_s:.3f}",
        "-c:a", "pcm_f32le",
        "-f", "f32le",
        "pipe:1",
    ]  # fmt: skip
    result = _run(command, settings, step="decode")
    undecodable = "the audio stream could not be decoded"
    _check_not_crashed(result, "decode", undecodable)
    if result.returncode != 0:
        raise DecodeError(undecodable)

    raw = result.stdout[: len(result.stdout) - len(result.stdout) % 4]
    samples = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    if samples.size == 0:
        raise EmptyAudioError("the file contains no audio")
    if samples.size > settings.max_duration_s * TARGET_SAMPLE_RATE:
        raise DurationTooLongError(_too_long_message(settings))
    # Float WAV files can carry NaN or infinity, which would silently poison a model's output.
    if not np.isfinite(samples).all():
        raise DecodeError("the decoded audio contains invalid sample values")
    return samples


def _run(command: list[str], settings: Settings, step: str) -> subprocess.CompletedProcess[bytes]:
    # No shell, a fixed argument list, and a timeout. On timeout, subprocess.run kills the
    # child and waits for it, so the temp file is released before the directory is deleted.
    try:
        return subprocess.run(  # noqa: S603  # nosec B603
            command, capture_output=True, timeout=settings.decode_timeout_s, check=False
        )
    except subprocess.TimeoutExpired:
        raise DecodeTimeoutError(
            f"audio {step} took longer than the {settings.decode_timeout_s:g} s limit"
        ) from None


def _check_not_crashed(result: subprocess.CompletedProcess[bytes], step: str, message: str) -> None:
    # Linux reports a process killed by a signal as minus the signal number: -11 for SIGSEGV,
    # -6 for SIGABRT, -9 for SIGKILL (also what the out-of-memory killer sends). Windows
    # reports an unhandled exception as its NTSTATUS code, such as 0xC0000005 for an access
    # violation. ffmpeg's own errors exit with small positive codes.
    code = result.returncode
    if code < 0 or code >= 0xC0000000:
        raise DecoderCrashedError(message, step=step, returncode=code)


def _parse_seconds(value: object) -> float | None:
    if not isinstance(value, str | int | float):
        return None
    try:
        seconds = float(value)
    except ValueError:  # ffprobe reports "N/A" when it doesn't know
        return None
    return seconds if math.isfinite(seconds) else None


def _codec_message(container: str, codec: str, allowed: frozenset[str]) -> str:
    return (
        f"{container} with the {codec or 'unknown'} codec is not supported (only "
        f"{', '.join(sorted(allowed))} audio, with no embedded images or extra streams)"
    )


def _too_large_message(settings: Settings) -> str:
    return f"file is larger than the {settings.max_file_bytes / MB:g} MB limit"


def _too_long_message(settings: Settings) -> str:
    return f"audio is longer than the {settings.max_duration_s:g} s limit"
