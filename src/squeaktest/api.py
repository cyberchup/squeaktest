"""The web app: one upload endpoint, the static page, and the controls around them.

    squeaktest serve        # then open http://127.0.0.1:8000

Uploads are untrusted, and they are sensitive (a voice is biometric data). So:

- The file is the raw request body (`application/octet-stream`), read into memory with a hard
  size cap and a time limit. There is no multipart parser, and the browser never sends a
  filename (docs/decisions.md D24).
- It then goes through the same checks as the CLI (`audio.load_audio_bytes`): content
  sniffing, a codec allowlist, the duration limit, and ffmpeg in a subprocess with a timeout,
  working in a private temporary directory that is deleted afterwards.
- One analysis runs at a time and a few more may wait; beyond that the server answers "busy"
  straight away. Each client is also rate limited (D25).
- Logs hold request metadata only: outcome, band, sizes and timings. Never audio, never a
  client's address, never a score.
- Only requests addressed to this machine's loopback names, or to a host name the operator
  lists (`SQUEAKTEST_ALLOWED_HOSTS`), are answered. That stops a website from using DNS
  rebinding to reach the app through a visitor's browser (docs/threat-model.md T3).
- Every response carries security headers, including a Content Security Policy that allows
  only squeaktest's own scripts, styles and requests. There is no CORS, so other sites can't
  read results, and the API docs pages are off (they load scripts from a third-party CDN).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from importlib import resources
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from squeaktest import __version__
from squeaktest.audio import (
    ALLOWED_FORMATS,
    AudioError,
    DecoderCrashedError,
    FileTooLargeError,
    check_tools,
    load_audio_bytes,
)
from squeaktest.config import LOOPBACK_HOSTS, MB, ServerSettings, Settings, normalize_host
from squeaktest.inference import WindowingConfig, analyze
from squeaktest.report import DISCLAIMER, SCORE_NOTE, build_report

if TYPE_CHECKING:
    from squeaktest.model import ModelSpec

log = logging.getLogger("squeaktest.api")

UPLOAD_TYPE = "application/octet-stream"

# Ceiling on open connections, passed to uvicorn (`squeaktest serve`). Uploads are held in
# memory while they arrive, so this also bounds memory: 32 uploads at the 25 MB cap is 800 MB.
MAX_CONNECTIONS = 32

# Clients the rate limiter remembers at once. Past this, the least recently seen are dropped.
MAX_TRACKED_CLIENTS = 10_000

# The static page: file name -> content type. Anything else under /static/ is a 404.
_ASSETS = {
    "index.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
    "favicon.svg": "image/svg+xml",
}


class Scorer(Protocol):
    """What the app needs from a detector (model.Detector, or a stand-in in tests)."""

    spec: ModelSpec

    def __call__(self, windows: Sequence[np.ndarray]) -> Sequence[float] | np.ndarray: ...


def create_app(
    detector: Scorer,
    settings: Settings | None = None,
    server: ServerSettings | None = None,
) -> FastAPI:
    """Build the app around an already loaded detector. Fails now if ffmpeg is missing."""
    settings = settings if settings is not None else Settings.from_env()
    server = server if server is not None else ServerSettings.from_env()
    check_tools(settings)
    bands = detector.spec.bands
    if bands is None:
        raise ValueError(f"{detector.spec.name} has no band thresholds; it is for evaluation only")
    config = WindowingConfig()
    assets = _load_assets()
    limiter = RateLimiter(server.rate_limit_per_minute)
    queue = AnalysisQueue(server.max_waiting)
    in_flight: set[str] = set()  # clients with an upload or analysis under way

    async def sweep_limiter() -> None:
        while True:
            await asyncio.sleep(60)
            limiter.forget()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        sweeper = asyncio.create_task(sweep_limiter())
        yield
        sweeper.cancel()
        queue.shutdown()

    app = FastAPI(
        title="squeaktest",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    # Added first, so it sits inside SecurityHeaders and its refusals get the headers too.
    app.add_middleware(HostCheck, allowed_hosts=server.allowed_hosts)
    app.add_middleware(SecurityHeaders, frame_ancestors=server.frame_ancestors)

    def score(data: bytes) -> dict[str, Any]:
        # Runs on the analysis thread: decoding and the model both block.
        audio = load_audio_bytes(data, settings)
        result = analyze(audio.samples, audio.sample_rate, detector, config)
        return build_report(result, audio, detector.spec, config)

    @app.post("/api/analyze")
    async def analyze_upload(request: Request) -> Response:
        started = time.monotonic()
        client = client_address(request, server.proxy_hops)
        wait_s = limiter.take(client)
        if wait_s > 0:
            retry = math.ceil(wait_s)
            _log_outcome("rate_limited", started)
            return _error(429, "rate_limited", f"Too many requests. Try again in {retry} s.", retry)
        if _media_type(request) != UPLOAD_TYPE:
            _log_outcome("wrong_content_type", started)
            return _error(415, "unsupported_media_type", f"Send the file as {UPLOAD_TYPE}.")
        if not queue.has_room():
            _log_outcome("busy", started)
            return _error(503, "busy", "The server is busy. Try again in a minute.", 60)
        # One analysis per client at a time, so one client can't fill the whole queue.
        if client in in_flight:
            _log_outcome("one_at_a_time", started)
            return _error(429, "one_at_a_time", "Wait for your current analysis to finish.", 10)
        in_flight.add(client)
        try:
            return await receive_and_score(request, started)
        finally:
            in_flight.discard(client)

    async def receive_and_score(request: Request, started: float) -> Response:
        try:
            data = await read_upload(request, settings.max_file_bytes, server.upload_timeout_s)
        except UploadTooLargeError:
            _log_outcome("too_large", started)
            return _error(413, "too_large", _too_large(settings))
        except TimeoutError:
            _log_outcome("upload_timeout", started)
            return _error(408, "upload_timeout", "The upload took too long.")

        try:
            report = await queue.run(score, data)
        except QueueFullError:
            _log_outcome("busy", started, size=len(data))
            return _error(503, "busy", "The server is busy. Try again in a minute.", 60)
        except FileTooLargeError:
            _log_outcome("too_large", started, size=len(data))
            return _error(413, "too_large", _too_large(settings))
        except DecoderCrashedError as exc:
            # ffmpeg was killed rather than refusing the file: what an exploit attempt looks
            # like, so it gets its own outcome at WARNING. The uploader gets the same answer as
            # for any file that can't be read (docs/decisions.md D29).
            _log_outcome(
                "decoder_crashed",
                started,
                level=logging.WARNING,
                size=len(data),
                step=exc.step,
                returncode=exc.returncode,
            )
            return _error(400, "rejected", f"Input rejected: {exc}.")
        except AudioError as exc:
            # AudioError messages never contain file names or content (see audio.py).
            _log_outcome("rejected", started, size=len(data), reason=type(exc).__name__)
            return _error(400, "rejected", f"Input rejected: {exc}.")
        except Exception as exc:
            # Only the exception's type is logged: its message could carry details of the input.
            _log_outcome("error", started, size=len(data), reason=type(exc).__name__)
            return _error(500, "internal_error", "Analysis failed. Nothing was kept.")

        _log_outcome(
            "ok",
            started,
            size=len(data),
            band=report["band"],
            audio_s=round(report["audio"]["duration_s"], 1),
            codec=report["audio"]["codec"],
        )
        return JSONResponse(report)

    @app.get("/api/info")
    async def info() -> dict[str, Any]:
        return {
            "version": __version__,
            "model": {
                "name": detector.spec.name,
                "revision": detector.spec.revision,
                "license": detector.spec.license,
            },
            "limits": {
                "max_file_mb": settings.max_file_bytes / MB,
                "max_duration_s": settings.max_duration_s,
                "formats": ALLOWED_FORMATS,
            },
            "bands": {
                "likely_synthetic_at": bands.likely_synthetic_at,
                "likely_genuine_below": bands.likely_genuine_below,
            },
            "disclaimer": DISCLAIMER,
            "score_note": SCORE_NOTE,
        }

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/")
    async def index() -> Response:
        return _asset(assets, "index.html")

    @app.get("/static/{name}")
    async def static(name: str) -> Response:
        if name not in assets or name == "index.html":
            return _error(404, "not_found", "Not found.")
        return _asset(assets, name)

    return app


class UploadTooLargeError(Exception):
    pass


async def read_upload(request: Request, limit: int, timeout_s: float) -> bytes:
    """Read the request body into memory, stopping as soon as it passes `limit` bytes."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise UploadTooLargeError  # refuse before reading a byte
    body = bytearray()
    async with asyncio.timeout(timeout_s):
        async for chunk in request.stream():
            body += chunk
            if len(body) > limit:
                raise UploadTooLargeError
    return bytes(body)


class QueueFullError(Exception):
    pass


class AnalysisQueue:
    """Runs analyses one at a time on a worker thread, with at most `max_waiting` waiting.

    The model on a 2-vCPU host is fastest one clip at a time (docs/decisions.md D22). The
    worker is a single thread, so analyses can't overlap even if a client disconnects
    mid-analysis. A request that disconnects while waiting is dropped from the queue.
    """

    def __init__(self, max_waiting: int) -> None:
        self.max_waiting = max_waiting
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="squeaktest-analysis")
        self._pending = 0  # running or waiting; only changed on the event loop's thread

    def has_room(self) -> bool:
        return self._pending < 1 + self.max_waiting

    async def run[T](self, fn: Callable[[bytes], T], data: bytes) -> T:
        if not self.has_room():
            raise QueueFullError
        loop = asyncio.get_running_loop()
        self._pending += 1
        future = self._executor.submit(fn, data)
        future.add_done_callback(lambda _: _call_soon(loop, self._finished))
        return await asyncio.wrap_future(future)

    def _finished(self) -> None:
        self._pending -= 1

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def _call_soon(loop: asyncio.AbstractEventLoop, callback: Callable[[], None]) -> None:
    # A RuntimeError means the loop has closed: the server is shutting down.
    with contextlib.suppress(RuntimeError):
        loop.call_soon_threadsafe(callback)


class RateLimiter:
    """A token bucket per client, in memory only. Client addresses are never logged.

    Each client may make `per_minute` requests in a burst, then one more every
    60 / `per_minute` seconds. A minute after its last request a client's bucket is full again,
    which is the same as never having been seen, so it is forgotten. The app sweeps once a
    minute, so an address is held for at most about two minutes after that client's last
    request. If more than `max_clients` are seen within a minute, the least recently seen are
    forgotten early, so memory stays bounded.
    """

    def __init__(
        self,
        per_minute: int,
        max_clients: int = MAX_TRACKED_CLIENTS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.capacity = float(per_minute)
        self.refill_per_s = per_minute / 60.0
        self.max_clients = max_clients
        self._clock = clock
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()

    def take(self, client: str) -> float:
        """Use one request. Returns 0 if allowed, else the seconds until one is available."""
        now = self._clock()
        tokens, last = self._buckets.pop(client, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - last) * self.refill_per_s)
        wait_s = 0.0
        if tokens >= 1.0:
            tokens -= 1.0
        else:
            wait_s = (1.0 - tokens) / self.refill_per_s
        self._buckets[client] = (tokens, now)
        self.forget(now)
        return wait_s

    def forget(self, now: float | None = None) -> None:
        """Drop clients whose bucket has refilled, and the oldest beyond `max_clients`."""
        now = self._clock() if now is None else now
        # Buckets are ordered from least to most recently seen.
        full_after_s = self.capacity / self.refill_per_s
        while self._buckets:
            oldest, (_, seen) = next(iter(self._buckets.items()))
            if len(self._buckets) <= self.max_clients and now - seen < full_after_s:
                break
            del self._buckets[oldest]

    def __len__(self) -> int:
        return len(self._buckets)


def client_address(request: Request, proxy_hops: int) -> str:
    """The client's address, as seen by the nearest proxy squeaktest trusts.

    With no trusted proxies, it's whoever connected. Behind `proxy_hops` proxies, each appends
    the address it saw to X-Forwarded-For, so the client is the entry `proxy_hops` from the
    end. Entries further left were written by the client and can't be trusted.
    """
    if proxy_hops > 0:
        forwarded = [
            part.strip()
            for header in request.headers.getlist("x-forwarded-for")
            for part in header.split(",")
            if part.strip()
        ]
        if len(forwarded) >= proxy_hops:
            return forwarded[-proxy_hops]
    return request.client.host if request.client else "unknown"


WRONG_HOST_MESSAGE = (
    "This server doesn't answer to that host name. If you run it, add the name to "
    "SQUEAKTEST_ALLOWED_HOSTS."
)


class HostCheck:
    """Answers only requests addressed to one of this server's own names (plain ASGI middleware).

    A website can point its own domain at this server's address (DNS rebinding). The visitor's
    browser then treats squeaktest as part of that website and lets its scripts upload files
    and read the results. But the browser still sends the website's name in the Host header,
    so only loopback names and the names the operator listed are answered.
    """

    def __init__(self, app: ASGIApp, allowed_hosts: Sequence[str]) -> None:
        self.app = app
        self.allowed = LOOPBACK_HOSTS | frozenset(allowed_hosts)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self._allowed(Headers(scope=scope).get("host", "")):
            await self.app(scope, receive, send)
            return
        # Metadata only, like every other line: the name itself is the caller's text. A run of
        # these is what a rebinding attempt looks like in the logs.
        log.info(json.dumps({"event": "request", "outcome": "wrong_host"}))
        await _error(400, "wrong_host", WRONG_HOST_MESSAGE)(scope, receive, send)

    def _allowed(self, header: str) -> bool:
        host = request_host(header)
        return host is not None and host in self.allowed


def request_host(header: str) -> str | None:
    """The host name in a Host header, normalized and without its port; None if malformed."""
    if header.startswith("["):  # an IPv6 address, as in [::1]:8000
        end = header.find("]")
        if end == -1:
            return None
        name, rest = header[1:end], header[end + 1 :]
    else:
        name, colon, port = header.partition(":")
        rest = colon + port
    if rest and not (rest.startswith(":") and rest[1:].isascii() and rest[1:].isdigit()):
        return None
    return normalize_host(name)


class SecurityHeaders:
    """Adds security headers to every response (plain ASGI middleware)."""

    def __init__(self, app: ASGIApp, frame_ancestors: str) -> None:
        self.app = app
        self.headers = [
            (k.encode(), v.encode()) for k, v in security_headers(frame_ancestors).items()
        ]
        self._names = {k for k, _ in self.headers}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                kept = [
                    (k, v) for k, v in message.get("headers", []) if k.lower() not in self._names
                ]
                message = {**message, "headers": kept + self.headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


def security_headers(frame_ancestors: str) -> dict[str, str]:
    csp = "; ".join(
        [
            "default-src 'none'",
            "script-src 'self'",
            "style-src 'self'",
            "img-src 'self'",
            "connect-src 'self'",
            "base-uri 'none'",
            "form-action 'none'",
            f"frame-ancestors {frame_ancestors}",
        ]
    )
    headers = {
        "content-security-policy": csp,
        "x-content-type-options": "nosniff",
        "referrer-policy": "no-referrer",
        "permissions-policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
        "cross-origin-opener-policy": "same-origin",
        "cross-origin-resource-policy": "same-origin",
        # Results describe someone's voice: no browser or proxy should keep a copy.
        "cache-control": "no-store",
    }
    if frame_ancestors == "'none'":
        headers["x-frame-options"] = "DENY"  # the same rule for browsers without CSP 3
    return headers


def _media_type(request: Request) -> str:
    return request.headers.get("content-type", "").split(";", 1)[0].strip().lower()


def _error(status: int, code: str, message: str, retry_after_s: int | None = None) -> Response:
    headers = {"retry-after": str(retry_after_s)} if retry_after_s is not None else None
    return JSONResponse({"error": code, "message": message}, status_code=status, headers=headers)


def _too_large(settings: Settings) -> str:
    return f"The file is larger than the {settings.max_file_bytes / MB:g} MB limit."


def _log_outcome(outcome: str, started: float, level: int = logging.INFO, **fields: object) -> None:
    # Metadata only. Never the audio, the client's address or the score.
    record = {"event": "analysis", "outcome": outcome, **fields}
    record["elapsed_s"] = round(time.monotonic() - started, 2)
    log.log(level, json.dumps(record))


def _load_assets() -> dict[str, tuple[bytes, str]]:
    # Read once at startup: requests never touch the filesystem for the page.
    root = resources.files("squeaktest") / "web"
    return {name: ((root / name).read_bytes(), ctype) for name, ctype in _ASSETS.items()}


def _asset(assets: dict[str, tuple[bytes, str]], name: str) -> Response:
    body, content_type = assets[name]
    return Response(body, media_type=content_type)
