"""Phone and compression conditions: re-encode a clip the way a SOC would receive it.

Each condition runs the clip through a real codec with ffmpeg. The result goes back through
squeaktest's normal loader, so the evaluation tests the same path users get.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Condition:
    name: str
    description: str
    encode: tuple[str, ...]  # ffmpeg output arguments
    suffix: str
    # For codecs the loader doesn't accept, decode back to WAV, as a call recorder would.
    back_to_wav: bool = False


CONDITIONS: dict[str, Condition] = {
    c.name: c
    for c in [
        Condition("clean", "original audio", (), ""),
        Condition(
            "g711",
            "landline call: G.711 mu-law, 8 kHz",
            ("-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw"),
            ".wav",
        ),
        Condition(
            "amr-nb",
            "mobile call: AMR-NB 12.2 kbit/s, 8 kHz",
            ("-ar", "8000", "-ac", "1", "-c:a", "libopencore_amrnb", "-b:a", "12.2k"),
            ".amr",
            back_to_wav=True,
        ),
        Condition(
            "opus",
            "VoIP call or voice note: Opus 12 kbit/s",
            ("-ac", "1", "-c:a", "libopus", "-b:a", "12k"),
            ".ogg",
        ),
        Condition(
            "mp3",
            "compressed recording: MP3 32 kbit/s",
            ("-ac", "1", "-c:a", "libmp3lame", "-b:a", "32k"),
            ".mp3",
        ),
    ]
}


def degrade(source: Path, condition: str, workdir: Path, ffmpeg: str = "ffmpeg") -> Path:
    """Return a path to `source` under `condition` (the source itself for "clean")."""
    cond = CONDITIONS[condition]
    if cond.name == "clean":
        return source
    encoded = workdir / f"encoded{cond.suffix}"
    _ffmpeg(ffmpeg, source, cond.encode, encoded)
    if not cond.back_to_wav:
        return encoded
    decoded = workdir / "decoded.wav"
    _ffmpeg(ffmpeg, encoded, ("-c:a", "pcm_s16le"), decoded)
    return decoded


def _ffmpeg(ffmpeg: str, source: Path, args: tuple[str, ...], target: Path) -> None:
    command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(source)]
    subprocess.run([*command, *args, "-y", str(target)], check=True, capture_output=True)
