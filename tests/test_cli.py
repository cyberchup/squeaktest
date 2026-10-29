"""Tests for the command line and the report it prints (step 7).

A fake detector stands in for the model so these run in CI. One end-to-end test uses the real
weights when they are downloaded.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

import squeaktest.model
from squeaktest.cli import EXIT_OK, EXIT_REJECTED, EXIT_SETUP, main
from squeaktest.inference import BANDS
from squeaktest.model import DEFAULT_MODEL, MODELS, ModelNotFoundError, weights_path
from squeaktest.report import DISCLAIMER, MAX_TIMELINE_ROWS, OPUS_WARNING

SPEC = MODELS[DEFAULT_MODEL]
assert SPEC.bands is not None
SYNTHETIC = (SPEC.bands.likely_synthetic_at + 1.0) / 2
UNCERTAIN = (SPEC.bands.likely_synthetic_at + SPEC.bands.likely_genuine_below) / 2
GENUINE = SPEC.bands.likely_genuine_below / 2


class FakeDetector:
    spec = SPEC

    def __init__(self, score: float) -> None:
        self.score = score

    def __call__(self, windows: Sequence[np.ndarray]) -> list[float]:
        return [self.score] * len(windows)


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch):
    """Make the CLI load a fake detector that gives every window the same score."""

    def use(score: float) -> None:
        monkeypatch.setattr(
            squeaktest.model, "load_detector", lambda **_kwargs: FakeDetector(score)
        )

    return use


def test_version(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert "squeaktest" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("score", "label"),
    [(SYNTHETIC, "Likely synthetic"), (UNCERTAIN, "Uncertain"), (GENUINE, "Likely genuine")],
)
def test_human_output_shows_band_score_and_disclaimer(
    audio_fixtures: dict[str, Path], fake_model, capsys, score: float, label: str
):
    fake_model(score)
    assert main(["analyze", str(audio_fixtures["flac"])]) == EXIT_OK
    out = capsys.readouterr().out
    assert f"Result:   {label}" in out
    assert f"score {score:.2f}, uncalibrated" in out
    assert "Timeline (4 s windows every 2 s)" in out
    assert DISCLAIMER in out
    assert "not a probability" in out
    assert "Warning:" not in out


def test_json_output_is_complete_and_parseable(audio_fixtures: dict[str, Path], fake_model, capsys):
    fake_model(SYNTHETIC)
    assert main(["analyze", str(audio_fixtures["m4a_aac"]), "--json"]) == EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["band"] == "likely_synthetic"
    assert report["score"] == pytest.approx(SYNTHETIC, abs=1e-4)
    assert report["calibrated"] is False
    assert report["warnings"] == []
    assert report["bands"] == {
        "likely_synthetic_at": SPEC.bands.likely_synthetic_at,
        "likely_genuine_below": SPEC.bands.likely_genuine_below,
    }
    assert report["audio"]["container"] == "m4a"
    assert report["audio"]["codec"] == "aac"
    assert report["windowing"] == {"window_s": 4.0, "hop_s": 2.0, "aggregation": "top_3_mean"}
    assert report["model"]["name"] == DEFAULT_MODEL
    assert report["model"]["revision"] == SPEC.revision
    assert report["model"]["license"] == "CC BY-NC-SA 4.0"
    assert report["windows"]
    assert all(w["score"] == pytest.approx(SYNTHETIC, abs=1e-4) for w in report["windows"])
    assert report["disclaimer"] == DISCLAIMER


def test_opus_audio_carries_a_warning(audio_fixtures: dict[str, Path], fake_model, capsys):
    fake_model(SYNTHETIC)
    assert main(["analyze", str(audio_fixtures["ogg_opus"]), "--json"]) == EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["warnings"] == [{"code": "opus_audio", "message": OPUS_WARNING}]
    assert main(["analyze", str(audio_fixtures["ogg_opus"])]) == EXIT_OK
    assert "Warning:  Opus audio" in capsys.readouterr().out


def test_vorbis_audio_in_the_same_container_has_no_warning(
    audio_fixtures: dict[str, Path], fake_model, capsys
):
    fake_model(SYNTHETIC)
    assert main(["analyze", str(audio_fixtures["ogg_vorbis"]), "--json"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["warnings"] == []


def test_models_without_bands_are_refused():
    from squeaktest.inference import ClipResult, WindowingConfig
    from squeaktest.model import MMS_300M
    from squeaktest.report import build_report

    assert MMS_300M.bands is None
    with pytest.raises(ValueError, match="evaluation only"):
        build_report(ClipResult(0.5, (), 3.0), None, MMS_300M, WindowingConfig())  # type: ignore[arg-type]


def test_published_numbers_match_the_evaluation_results():
    """The bands and the Opus warning quote eval results; fail if they drift apart."""
    import squeaktest.report as report

    path = Path(__file__).parents[1] / "eval" / "results" / DEFAULT_MODEL / "results.json"
    results = json.loads(path.read_text(encoding="utf-8"))
    bands = results["bands"]
    assert SPEC.bands.likely_synthetic_at == pytest.approx(
        bands["thresholds"]["likely_synthetic_at"], abs=5e-4
    )
    assert SPEC.bands.likely_genuine_below == pytest.approx(
        bands["thresholds"]["likely_genuine_below"], abs=5e-4
    )
    opus = results["conditions"]["opus"]
    measured = {
        "CLEAN_GENUINE_FLAGGED": bands["test"]["genuine"]["likely_synthetic"],
        "OPUS_GENUINE_FLAGGED": opus["bands"]["genuine"]["likely_synthetic"],
        "OPUS_ALERT_PRECISION_1_IN_100": opus["alerts"]["precision_at_prevalence"]["0.01"][
            "estimate"
        ],
    }
    for name, value in measured.items():
        assert getattr(report, name) == pytest.approx(value, abs=5e-4), name


def test_output_never_contains_the_filename(
    audio_fixtures: dict[str, Path], fake_model, capsys, tmp_path: Path
):
    secret = tmp_path / "ceo-voicemail-secret.flac"
    secret.write_bytes(audio_fixtures["flac"].read_bytes())
    fake_model(0.5)
    main(["analyze", str(secret)])
    main(["analyze", str(secret), "--json"])
    captured = capsys.readouterr()
    assert "secret" not in captured.out + captured.err
    assert "ceo-voicemail" not in captured.out + captured.err


def test_rejected_input_exits_1_without_loading_the_model(tmp_path: Path, monkeypatch, capsys):
    def must_not_load(**_kwargs):
        raise AssertionError("the model should not load for a rejected file")

    monkeypatch.setattr(squeaktest.model, "load_detector", must_not_load)
    bad = tmp_path / "invoice.wav"
    bad.write_text("not audio at all")
    assert main(["analyze", str(bad)]) == EXIT_REJECTED
    err = capsys.readouterr().err
    assert "input rejected" in err
    assert "invoice" not in err


def test_too_little_sound_is_not_assessed(fake_model, capsys, tmp_path: Path):
    import wave

    quiet = tmp_path / "quiet.wav"
    with wave.open(str(quiet), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(b"\x00\x00" * 16_000 * 3)  # 3 s of digital silence
    fake_model(0.99)
    assert main(["analyze", str(quiet), "--json"]) == EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["band"] == "not_assessed"
    assert report["score"] is None
    assert "not enough" in report["note"]


def test_steady_tones_are_left_out_with_a_warning(fake_model, capsys, tmp_path: Path):
    import wave

    t = np.arange(16_000 * 4) / 16_000
    dial_tone = 0.2 * (np.sin(2 * np.pi * 350 * t) + np.sin(2 * np.pi * 440 * t))
    path = tmp_path / "dial.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes((dial_tone * 32767).astype("<i2").tobytes())
    fake_model(0.99)  # what the model would say about a tone (K1)
    assert main(["analyze", str(path), "--json"]) == EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["band"] == "not_assessed"
    assert "steady tones" in report["note"]
    assert report["audio"]["tones_s"] == pytest.approx(4.0, abs=0.1)
    assert [w["code"] for w in report["warnings"]] == ["steady_tones"]


def test_missing_model_exits_2_and_says_how_to_fetch_it(
    audio_fixtures: dict[str, Path], monkeypatch, capsys
):
    def missing(**_kwargs):
        raise ModelNotFoundError("weights not found.")

    monkeypatch.setattr(squeaktest.model, "load_detector", missing)
    assert main(["analyze", str(audio_fixtures["flac"])]) == EXIT_SETUP
    assert "squeaktest fetch-model" in capsys.readouterr().err


def test_cuda_requested_without_a_gpu_exits_2(audio_fixtures: dict[str, Path], monkeypatch, capsys):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert main(["analyze", str(audio_fixtures["flac"]), "--device", "cuda"]) == EXIT_SETUP
    assert "no CUDA GPU" in capsys.readouterr().err


def test_long_clips_show_only_the_most_suspicious_windows(fake_model, capsys, tmp_path: Path):
    from conftest import FFMPEG, harmonic_tone

    if FFMPEG is None:
        pytest.skip("ffmpeg needed")
    import subprocess

    long_clip = tmp_path / "long.flac"
    subprocess.run(
        [FFMPEG, "-nostdin", "-loglevel", "error", "-f", "lavfi",
         "-i", harmonic_tone(440, 80), "-y", str(long_clip)],
        check=True,
    )  # fmt: skip
    fake_model(0.3)
    assert main(["analyze", str(long_clip)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "Most suspicious 10 of" in out
    assert "use --json for every window" in out
    assert out.count(" s  ") <= MAX_TIMELINE_ROWS


def test_fetch_model_asks_first_and_respects_no(monkeypatch, capsys, tmp_path: Path):
    monkeypatch.setenv("SQUEAKTEST_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    def must_not_download(*_args):
        raise AssertionError("declined, so nothing should download")

    monkeypatch.setattr(squeaktest.model, "fetch_weights", must_not_download)
    assert main(["fetch-model"]) == EXIT_REJECTED
    out = capsys.readouterr().out
    assert "1.27 GB" in out
    assert "non-commercial" in out


def test_fetch_model_with_yes_downloads_without_asking(monkeypatch, capsys, tmp_path: Path):
    monkeypatch.setenv("SQUEAKTEST_DATA_DIR", str(tmp_path))

    def no_prompt(_prompt):
        raise AssertionError("--yes should skip the prompt")

    calls = []
    monkeypatch.setattr("builtins.input", no_prompt)
    monkeypatch.setattr(
        squeaktest.model,
        "fetch_weights",
        lambda spec, data_dir: calls.append(data_dir) or weights_path(spec, data_dir),
    )
    assert main(["fetch-model", "--yes"]) == EXIT_OK
    assert calls == [tmp_path]
    assert "SHA-256 verified" in capsys.readouterr().out


# --- End to end with the real model ------------------------------------------------------


@pytest.mark.skipif(
    not weights_path(SPEC, squeaktest.config.Settings.from_env().data_dir).exists(),
    reason="default model's weights not downloaded (set SQUEAKTEST_DATA_DIR)",
)
def test_real_model_end_to_end(audio_fixtures: dict[str, Path], capsys):
    assert main(["analyze", str(audio_fixtures["mp3"]), "--json", "--device", "cpu"]) == EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["band"] in BANDS
    assert 0.0 <= report["score"] <= 1.0
