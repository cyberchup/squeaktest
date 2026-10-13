"""Runtime settings, read from environment variables with safe defaults."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MB = 1024 * 1024

DEFAULT_DATA_DIR = Path.home() / ".cache" / "squeaktest"


@dataclass(frozen=True)
class Settings:
    """Limits, tool locations and the data directory.

    All limits are configurable because the hosted demo may run tighter limits than local use.
    Model weights and datasets live under `data_dir`, never in the repository.
    """

    max_file_bytes: int = 25 * MB
    max_duration_s: float = 300.0
    decode_timeout_s: float = 30.0
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    temp_dir: Path | None = None
    data_dir: Path = DEFAULT_DATA_DIR

    def __post_init__(self) -> None:
        for name in ("max_file_bytes", "max_duration_s", "decode_timeout_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive number, got {value!r}")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Build settings from SQUEAKTEST_* environment variables, defaulting the rest."""
        return _from_env(cls, _ENV_VARS, environ)


@dataclass(frozen=True)
class ServerSettings:
    """Limits for the web app (`squeaktest serve`). The audio limits stay in Settings.

    The model scores one clip at a time: on a 2-vCPU host, two at once would only make both
    slower (docs/decisions.md D22). Up to `max_waiting` more requests wait their turn; beyond
    that the server answers "busy" straight away instead of holding uploads in memory.
    """

    rate_limit_per_minute: int = 6  # analyses per client, refilled evenly over the minute
    max_waiting: int = 4
    upload_timeout_s: float = 60.0  # a slow upload can't hold a connection open forever
    # How many reverse proxies sit in front of the app and append to X-Forwarded-For. 0 means
    # the client is whoever connects, which is right for local use. Behind one proxy that
    # appends the client's address (such as a Hugging Face Space), set it to 1. Setting it
    # higher than the real number lets clients choose their own address, and so dodge the
    # rate limit.
    proxy_hops: int = 0
    # Who may show the page in a frame (CSP frame-ancestors). Hugging Face shows a Space's page
    # inside huggingface.co, so a Space needs "https://huggingface.co".
    frame_ancestors: str = "'none'"

    def __post_init__(self) -> None:
        if self.rate_limit_per_minute < 1:
            raise ValueError("rate_limit_per_minute must be at least 1")
        if self.max_waiting < 0 or self.proxy_hops < 0:
            raise ValueError("max_waiting and proxy_hops must be 0 or more")
        if not math.isfinite(self.upload_timeout_s) or self.upload_timeout_s <= 0:
            raise ValueError("upload_timeout_s must be a positive number")
        # The value goes into a response header: refuse anything that could end the
        # directive or the header.
        if not self.frame_ancestors.strip() or any(c in self.frame_ancestors for c in ";,\r\n"):
            raise ValueError("frame_ancestors must be a CSP source list")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ServerSettings:
        return _from_env(cls, _SERVER_ENV_VARS, environ)


def _from_env[T](
    cls: Callable[..., T],
    table: Mapping[str, tuple[str, Callable[[str], Any]]],
    environ: Mapping[str, str] | None,
) -> T:
    env = os.environ if environ is None else environ
    kwargs: dict[str, Any] = {}
    for var, (field, convert) in table.items():
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
    "SQUEAKTEST_DATA_DIR": ("data_dir", Path),
}

_SERVER_ENV_VARS: dict[str, tuple[str, Callable[[str], Any]]] = {
    "SQUEAKTEST_RATE_LIMIT_PER_MINUTE": ("rate_limit_per_minute", int),
    "SQUEAKTEST_MAX_WAITING": ("max_waiting", int),
    "SQUEAKTEST_UPLOAD_TIMEOUT_S": ("upload_timeout_s", float),
    "SQUEAKTEST_PROXY_HOPS": ("proxy_hops", int),
    "SQUEAKTEST_FRAME_ANCESTORS": ("frame_ancestors", str),
}
