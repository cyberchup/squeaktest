"""Tests for the web app (Phase 3).

A fake detector stands in for the model, so these run in CI. The security controls are tested
as behavior: what a browser or an attacker would see.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import logging
import re
import threading
from collections.abc import Iterator, Sequence
from importlib import resources
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

import squeaktest.model
from squeaktest.api import (
    MAX_CONNECTIONS,
    AnalysisQueue,
    QueueFullError,
    RateLimiter,
    UploadTooLargeError,
    create_app,
    read_upload,
)
from squeaktest.cli import EXIT_OK, main
from squeaktest.config import ServerSettings, Settings
from squeaktest.model import DEFAULT_MODEL, MODELS
from squeaktest.report import DISCLAIMER

SPEC = MODELS[DEFAULT_MODEL]
assert SPEC.bands is not None
SYNTHETIC = (SPEC.bands.likely_synthetic_at + 1.0) / 2
OCTET = {"content-type": "application/octet-stream"}


class FakeDetector:
    device = "cpu"

    def __init__(self, score: float = SYNTHETIC, spec=SPEC) -> None:
        self.score = score
        self.spec = spec

    def __call__(self, windows: Sequence[np.ndarray]) -> list[float]:
        return [self.score] * len(windows)


@pytest.fixture
def make_client(audio_fixtures: dict[str, Path]) -> Iterator:
    """Build a test client; the audio_fixtures dependency skips these tests without ffmpeg."""
    clients: list[TestClient] = []

    def make(settings: Settings | None = None, **server: object) -> TestClient:
        app = create_app(FakeDetector(), settings or Settings(), ServerSettings(**server))
        # TestClient's default host name, "testserver", isn't one the app answers to.
        client = TestClient(app, base_url="http://localhost").__enter__()
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


def post_file(client: TestClient, path: Path, headers: dict[str, str] | None = None):
    return client.post("/api/analyze", content=path.read_bytes(), headers=headers or OCTET)


# --- The endpoint --------------------------------------------------------------------------


def test_analyze_returns_the_same_report_as_the_cli(make_client, audio_fixtures):
    response = post_file(make_client(), audio_fixtures["flac"])
    assert response.status_code == 200
    report = response.json()
    assert report["band"] == "likely_synthetic"
    assert report["calibrated"] is False
    assert report["disclaimer"] == DISCLAIMER
    assert report["windows"]
    assert report["model"]["name"] == SPEC.name
    assert "tone.flac" not in response.text


def test_opus_uploads_carry_the_warning(make_client, audio_fixtures):
    report = post_file(make_client(), audio_fixtures["ogg_opus"]).json()
    assert [w["code"] for w in report["warnings"]] == ["opus_audio"]


@pytest.mark.parametrize("fixture", ["mp4_video", "adts_aac", "wav_empty"])
def test_rejected_files_get_a_400_with_a_safe_message(make_client, audio_fixtures, fixture):
    response = post_file(make_client(), audio_fixtures[fixture])
    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "rejected"
    assert audio_fixtures[fixture].name not in body["message"]


def test_random_bytes_are_rejected(make_client):
    response = make_client().post("/api/analyze", content=b"\x00" * 1000, headers=OCTET)
    assert response.status_code == 400
    assert "not a supported audio file" in response.json()["message"]


@pytest.mark.parametrize(
    "content_type",
    [None, "text/plain", "multipart/form-data; boundary=x", "audio/wav", "application/json"],
)
def test_only_raw_octet_stream_uploads_are_accepted(make_client, audio_fixtures, content_type):
    headers = {"content-type": content_type} if content_type else {}
    data = audio_fixtures["flac"].read_bytes()
    response = make_client().post("/api/analyze", content=data, headers=headers)
    assert response.status_code == 415


def test_declared_oversized_upload_is_refused(make_client):
    client = make_client(Settings(max_file_bytes=1000))
    response = client.post("/api/analyze", content=b"x" * 1001, headers=OCTET)
    assert response.status_code == 413
    assert response.json()["error"] == "too_large"


def test_oversized_upload_without_a_length_is_cut_off(make_client):
    # A generator body is sent chunked, with no Content-Length to check up front.
    client = make_client(Settings(max_file_bytes=1000))
    response = client.post("/api/analyze", content=iter([b"x" * 600, b"x" * 600]), headers=OCTET)
    assert response.status_code == 413


def test_info_reports_limits_and_model(make_client):
    client = make_client(Settings(max_duration_s=60))
    info = client.get("/api/info").json()
    assert info["limits"]["max_duration_s"] == 60
    assert info["limits"]["max_file_mb"] == 25
    assert info["model"]["license"] == SPEC.license
    assert info["bands"]["likely_synthetic_at"] == SPEC.bands.likely_synthetic_at


def test_healthz(make_client):
    assert make_client().get("/healthz").json() == {"status": "ok"}


def test_a_model_without_bands_is_refused(audio_fixtures):
    with pytest.raises(ValueError, match="no band thresholds"):
        create_app(FakeDetector(spec=MODELS["antideepfake-mms-300m"]), Settings(), ServerSettings())


# --- Security headers, CSP and CORS --------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/"),
        ("GET", "/static/app.js"),
        ("GET", "/api/info"),
        ("GET", "/healthz"),
        ("GET", "/no-such-page"),
        ("POST", "/api/analyze"),  # 415: no body, no content type
    ],
)
def test_every_response_carries_security_headers(make_client, method, path):
    response = make_client().request(method, path)
    headers = response.headers
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["referrer-policy"] == "no-referrer"
    assert headers["cache-control"] == "no-store"
    assert headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in headers["content-security-policy"]
    assert "microphone=()" in headers["permissions-policy"]
    assert "server" not in headers


def test_csp_allows_only_this_origin_and_no_inline_code(make_client):
    csp = make_client().get("/").headers["content-security-policy"]
    directives = dict(d.strip().split(" ", 1) for d in csp.split(";"))
    assert directives["default-src"] == "'none'"
    for name in ("script-src", "style-src", "img-src", "connect-src"):
        assert directives[name] == "'self'"
    assert "unsafe" not in csp
    assert "*" not in csp


def test_frame_ancestors_can_allow_hugging_face(make_client):
    headers = make_client(frame_ancestors="https://huggingface.co").get("/").headers
    assert "frame-ancestors https://huggingface.co" in headers["content-security-policy"]
    assert "x-frame-options" not in headers  # DENY would contradict the CSP


def test_there_is_no_cors(make_client, audio_fixtures):
    client = make_client()
    preflight = client.options(
        "/api/analyze",
        headers={
            "origin": "https://evil.example",
            "access-control-request-method": "POST",
            "access-control-request-headers": "content-type",
        },
    )
    assert "access-control-allow-origin" not in preflight.headers
    post = post_file(
        client, audio_fixtures["flac"], headers={**OCTET, "origin": "https://evil.example"}
    )
    assert "access-control-allow-origin" not in post.headers


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_api_docs_are_off(make_client, path):
    # FastAPI's docs pages load scripts from a third-party CDN.
    assert make_client().get(path).status_code == 404


@pytest.mark.parametrize("path", ["/static/index.html", "/static/nope.js", "/static/..%2fapi.py"])
def test_only_listed_static_files_are_served(make_client, path):
    assert make_client().get(path).status_code == 404


# --- Host names (DNS rebinding, docs/threat-model.md T3) -----------------------------------


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "localhost:8000",
        "LOCALHOST",
        "localhost.",
        "127.0.0.1:7860",  # the image's health check
        "[::1]:8000",
    ],
)
def test_loopback_names_are_always_answered(make_client, host):
    assert make_client().get("/healthz", headers={"host": host}).status_code == 200


@pytest.mark.parametrize(
    "host", ["squeaktest.lan", "squeaktest.lan:8000", "SqueakTest.LAN", "192.168.1.50:8000"]
)
def test_listed_names_are_answered(make_client, host):
    client = make_client(allowed_hosts=("squeaktest.lan", "192.168.1.50"))
    assert client.get("/healthz", headers={"host": host}).status_code == 200


@pytest.mark.parametrize(
    "host",
    [
        "evil.example",  # what a rebinding website's page sends
        "evil.example:8000",
        "localhost.evil.example",
        "squeaktest.lan.evil.example",
        "192.168.1.51",
        "localhost:http",
        "[::1",
        "",
    ],
)
def test_other_host_names_are_refused(make_client, host):
    client = make_client(allowed_hosts=("squeaktest.lan", "192.168.1.50"))
    response = client.get("/api/info", headers={"host": host})
    assert response.status_code == 400
    assert response.json()["error"] == "wrong_host"
    assert "SQUEAKTEST_ALLOWED_HOSTS" in response.json()["message"]
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_a_rebound_upload_is_refused_and_logged_without_the_name(
    make_client, audio_fixtures, caplog
):
    caplog.set_level(logging.INFO, logger="squeaktest.api")
    response = post_file(
        make_client(), audio_fixtures["flac"], headers={**OCTET, "host": "evil.example:8000"}
    )
    assert response.status_code == 400
    records = [json.loads(r.getMessage()) for r in caplog.records if r.name == "squeaktest.api"]
    assert records == [{"event": "request", "outcome": "wrong_host"}]  # never analyzed
    assert "evil.example" not in caplog.text


# --- The page complies with the CSP --------------------------------------------------------


def web_file(name: str) -> str:
    return (resources.files("squeaktest") / "web" / name).read_text(encoding="utf-8")


def test_page_has_no_inline_code():
    html = web_file("index.html")
    for script in re.findall(r"<script\b[^>]*>", html):
        assert 'src="/static/' in script
    assert not re.search(r"<script\b[^>]*>\s*[^<\s]", html), "inline script"
    assert "<style" not in html
    assert not re.search(r"\sstyle\s*=", html), "inline style attribute"
    assert not re.search(r"\son[a-z]+\s*=", html), "inline event handler"


def test_page_loads_nothing_from_other_origins():
    html = web_file("index.html")
    for tag in re.findall(r"<(?:script|link|img)\b[^>]*>", html):
        for url in re.findall(r'(?:src|href)="([^"]+)"', tag):
            assert url.startswith("/static/"), tag
    assert "url(" not in web_file("style.css")
    assert "@import" not in web_file("style.css")
    js = web_file("app.js")
    fetches = re.findall(r"fetch\(\s*\"([^\"]+)\"", js)
    assert fetches
    assert all(url.startswith("/api/") for url in fetches)


def test_script_never_parses_server_text_as_html():
    js = web_file("app.js")
    for pattern in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
        'setAttribute("style"',
    ):
        assert pattern not in js, pattern


# --- Rate limiting and client addresses ----------------------------------------------------


def test_clients_are_rate_limited(make_client):
    client = make_client(rate_limit_per_minute=2)
    for _ in range(2):
        assert client.post("/api/analyze", content=b"x", headers=OCTET).status_code == 400
    limited = client.post("/api/analyze", content=b"x", headers=OCTET)
    assert limited.status_code == 429
    assert int(limited.headers["retry-after"]) >= 1


def test_forwarded_addresses_are_ignored_without_trusted_proxies(make_client):
    client = make_client(rate_limit_per_minute=1)
    post = lambda ip: client.post(  # noqa: E731
        "/api/analyze", content=b"x", headers={**OCTET, "x-forwarded-for": ip}
    )
    assert post("1.1.1.1").status_code == 400
    assert post("2.2.2.2").status_code == 429  # same connection, same client


def test_behind_one_proxy_the_last_forwarded_address_is_the_client(make_client):
    client = make_client(rate_limit_per_minute=1, proxy_hops=1)
    post = lambda xff: client.post(  # noqa: E731
        "/api/analyze", content=b"x", headers={**OCTET, "x-forwarded-for": xff}
    )
    assert post("1.1.1.1").status_code == 400
    assert post("2.2.2.2").status_code == 400  # a different client
    # A client can't dodge the limit by writing its own entries in front of the proxy's.
    assert post("9.9.9.9, 1.1.1.1").status_code == 429


def test_rate_limiter_refills_over_time():
    now = [0.0]
    limiter = RateLimiter(per_minute=2, clock=lambda: now[0])
    assert limiter.take("a") == 0
    assert limiter.take("a") == 0
    assert limiter.take("a") == pytest.approx(30.0)
    now[0] = 30.0
    assert limiter.take("a") == 0
    assert limiter.take("b") == 0  # clients are independent


def test_rate_limiter_memory_is_bounded():
    limiter = RateLimiter(per_minute=1, max_clients=2)
    for client in ("a", "b", "c"):
        limiter.take(client)
    assert len(limiter) == 2
    assert limiter.take("a") == 0  # forgotten, so it starts afresh


def test_rate_limiter_forgets_clients_once_their_limit_has_reset():
    now = [0.0]
    limiter = RateLimiter(per_minute=6, clock=lambda: now[0])
    limiter.take("a")
    now[0] = 30.0
    limiter.take("b")
    limiter.forget()
    assert len(limiter) == 2
    now[0] = 60.0  # a's bucket is full again: same as never seen
    limiter.forget()
    assert len(limiter) == 1
    now[0] = 90.0
    limiter.forget()
    assert len(limiter) == 0


# --- The analysis queue and the upload reader ----------------------------------------------


def test_queue_runs_one_at_a_time_and_refuses_when_full():
    async def scenario() -> None:
        queue = AnalysisQueue(max_waiting=0)
        release = threading.Event()
        first = asyncio.create_task(queue.run(lambda _data: release.wait(5), b""))
        await asyncio.sleep(0.05)
        assert not queue.has_room()
        with pytest.raises(QueueFullError):
            await queue.run(lambda _data: None, b"")
        release.set()
        assert await first is True
        await asyncio.sleep(0.05)  # the done callback runs on the loop
        assert queue.has_room()
        assert await queue.run(len, b"abc") == 3
        queue.shutdown()

    asyncio.run(scenario())


class SlowRequest:
    headers = Headers({})

    async def stream(self):
        yield b"x"
        await asyncio.sleep(10)
        yield b"y"


def test_slow_uploads_time_out():
    with pytest.raises(TimeoutError):
        asyncio.run(read_upload(SlowRequest(), limit=100, timeout_s=0.1))


def test_upload_reader_stops_at_the_limit():
    class BigRequest:
        headers = Headers({})

        async def stream(self):
            for _ in range(100):
                yield b"x" * 10

    with pytest.raises(UploadTooLargeError):
        asyncio.run(read_upload(BigRequest(), limit=50, timeout_s=1))


# --- Logging and privacy -------------------------------------------------------------------


def test_logs_hold_metadata_only(make_client, audio_fixtures, caplog):
    caplog.set_level(logging.INFO, logger="squeaktest.api")
    response = post_file(make_client(), audio_fixtures["flac"])
    score = response.json()["score"]
    records = [json.loads(r.getMessage()) for r in caplog.records if r.name == "squeaktest.api"]
    assert len(records) == 1
    record = records[0]
    assert record["outcome"] == "ok"
    assert record["band"] == "likely_synthetic"
    assert set(record) == {"event", "outcome", "size", "band", "audio_s", "codec", "elapsed_s"}
    text = caplog.text
    assert "testclient" not in text  # the client's address
    assert str(score) not in text
    assert "tone.flac" not in text


def test_no_telemetry_sdk_is_installed():
    # FastAPI depends on opentelemetry-api, which does nothing on its own. An SDK, exporter or
    # distro package is what would actually send data somewhere.
    names = {d.metadata["Name"].lower() for d in importlib.metadata.distributions()}
    senders = {n for n in names if n.startswith("opentelemetry-") and n != "opentelemetry-api"}
    assert not senders
    assert "gradio" not in names


# --- `squeaktest serve` --------------------------------------------------------------------


def test_serve_runs_uvicorn_with_safe_options(audio_fixtures, monkeypatch: pytest.MonkeyPatch):
    import uvicorn

    calls: list[dict] = []
    monkeypatch.setattr(squeaktest.model, "load_detector", lambda **_kwargs: FakeDetector())
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append(kwargs))
    assert main(["serve"]) == EXIT_OK
    (kwargs,) = calls
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["access_log"] is False
    assert kwargs["proxy_headers"] is False
    assert kwargs["server_header"] is False
    assert kwargs["limit_concurrency"] == MAX_CONNECTIONS


def test_a_client_gets_one_analysis_at_a_time(audio_fixtures):
    import httpx2

    release = threading.Event()

    class BlockingDetector(FakeDetector):
        def __call__(self, windows: Sequence[np.ndarray]) -> list[float]:
            release.wait(5)
            return super().__call__(windows)

    app = create_app(BlockingDetector(), Settings(), ServerSettings())
    data = audio_fixtures["flac"].read_bytes()

    async def scenario() -> None:
        transport = httpx2.ASGITransport(app=app)
        async with httpx2.AsyncClient(transport=transport, base_url="http://localhost") as client:
            first = asyncio.create_task(client.post("/api/analyze", content=data, headers=OCTET))
            await asyncio.sleep(0.5)  # the first is now being analyzed
            second = await client.post("/api/analyze", content=data, headers=OCTET)
            assert second.status_code == 429
            assert second.json()["error"] == "one_at_a_time"
            release.set()
            assert (await first).status_code == 200
            third = await client.post("/api/analyze", content=data, headers=OCTET)
            assert third.status_code == 200  # free again once the first finished

    asyncio.run(scenario())
