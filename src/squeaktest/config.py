"""Runtime settings, read from environment variables with safe defaults."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MB = 1024 * 1024


@dataclass(frozen=True)
class Settings:
    """Limits and tool locations for handling untrusted audio.

    All limits are configurable because the hosted demo may run tighter limits than local use.
    """

    max_file_bytes: int = 25 * MB
    max_duration_s: float = 300.0
    decode_timeout_s: float = 30.0
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    temp_dir: Path | None = None

    def __post_init__(self) -> None:
        for name in ("max_file_bytes", "max_duration_s", "decode_timeout_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive number, got {value!r}")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Build settings from SQUEAKTEST_* environment variables, defaulting the rest."""
        env = os.environ if environ is None else environ
        kwargs: dict[str, Any] = {}
        for var, (field, convert) in _ENV_VARS.items():
            raw = env.get(var)
            if raw is None or raw == "":
                continue
            try:
                kwargs[field] = convert(raw)
            except ValueError:
                raise ValueError(f"{var} has an invalid value: {raw!r}") from None
        try:
            return cls(**kwargs)
        except ValueError as exc:
            raise ValueError(f"invalid squeaktest setting: {exc}") from None


def _megabytes(raw: str) -> int:
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(raw)
    return int(value * MB)


_ENV_VARS: dict[str, tuple[str, Callable[[str], Any]]] = {
    "SQUEAKTEST_MAX_FILE_MB": ("max_file_bytes", _megabytes),
    "SQUEAKTEST_MAX_DURATION_S": ("max_duration_s", float),
    "SQUEAKTEST_DECODE_TIMEOUT_S": ("decode_timeout_s", float),
    "SQUEAKTEST_FFMPEG": ("ffmpeg", str),
    "SQUEAKTEST_FFPROBE": ("ffprobe", str),
    "SQUEAKTEST_TEMP_DIR": ("temp_dir", Path),
}
