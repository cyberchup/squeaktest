"""Tests for the evaluation harness: speaker split, sampling, codec conditions and scoring."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from conftest import FFMPEG, HAVE_FFMPEG

from eval.datasets import (
    ITW_DEV_SPEAKERS,
    Clip,
    choose_dev_speakers,
    read_dev_speakers,
    split_clips,
    stratified_sample,
)
from eval.degrade import CONDITIONS, degrade
from eval.score import score_clips
from squeaktest.audio import load_audio_file
from squeaktest.config import Settings
from squeaktest.inference import WindowingConfig

needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg and ffprobe needed")


def make_clips(sizes: dict[str, tuple[int, int]]) -> list[Clip]:
    """sizes: speaker -> (genuine clips, synthetic clips)."""
    clips = []
    for speaker, (real, fake) in sizes.items():
        for i in range(real + fake):
            clips.append(Clip(f"{speaker}-{i}.wav", Path("x"), int(i >= real), speaker))
    return clips


# --- Split and sampling ------------------------------------------------------------------


def test_dev_speakers_are_representative_and_reproducible():
    sizes = {f"s{i:02d}": (20 + 7 * i, 10 + 3 * (i % 5)) for i in range(30)}
    clips = make_clips(sizes)
    dev = choose_dev_speakers(clips)
    assert dev == choose_dev_speakers(clips)  # same seed, same split
    assert dev == sorted(dev) and len(dev) == 6
    dev_clips = [c for c in clips if c.speaker in set(dev)]
    assert 0.15 <= len(dev_clips) / len(clips) <= 0.25
    overall = np.mean([c.label for c in clips])
    assert abs(np.mean([c.label for c in dev_clips]) - overall) <= 0.05


def test_impossible_constraints_raise():
    clips = make_clips({"a": (10, 0), "b": (0, 10)})
    with pytest.raises(RuntimeError):
        choose_dev_speakers(clips, attempts=50)


def test_splits_share_no_speakers_and_cover_everything():
    clips = make_clips({"a": (3, 2), "b": (4, 1), "c": (2, 2)})
    dev = split_clips(clips, "dev", {"b"})
    test = split_clips(clips, "test", {"b"})
    assert {c.speaker for c in dev} == {"b"}
    assert "b" not in {c.speaker for c in test}
    assert len(dev) + len(test) == len(clips)
    assert split_clips(clips, "all", {"b"}) == clips
    with pytest.raises(ValueError):
        split_clips(clips, "train", {"b"})


def test_committed_dev_split_is_readable():
    dev = read_dev_speakers(ITW_DEV_SPEAKERS)
    assert len(dev) == 11
    assert not any(name.startswith("#") for name in dev)


def test_stratified_sample_keeps_the_fake_ratio_and_is_reproducible():
    clips = make_clips({"a": (700, 300)})
    sample = stratified_sample(clips, 100, seed=1)
    assert len(sample) == 100
    assert sum(c.label for c in sample) == 30
    assert sample == stratified_sample(clips, 100, seed=1)
    assert sample != stratified_sample(clips, 100, seed=2)
    assert stratified_sample(clips, 5000) == clips


# --- Codec conditions --------------------------------------------------------------------


@pytest.fixture(scope="module")
def speechlike_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("degrade") / "source.wav"
    if HAVE_FFMPEG:
        subprocess.run(
            [FFMPEG, "-nostdin", "-loglevel", "error", "-f", "lavfi",
             "-i", "sine=frequency=300:sample_rate=16000:duration=3", "-y", str(path)],
            check=True,
        )  # fmt: skip
    return path


@needs_ffmpeg
@pytest.mark.parametrize(
    ("condition", "container", "codec"),
    [
        ("g711", "wav", "pcm_mulaw"),
        ("amr-nb", "wav", "pcm_s16le"),
        ("opus", "ogg", "opus"),
        ("mp3", "mp3", "mp3"),
    ],
)
def test_conditions_produce_files_the_product_accepts(
    speechlike_wav: Path, tmp_path: Path, condition: str, container: str, codec: str
):
    out = degrade(speechlike_wav, condition, tmp_path, FFMPEG)
    audio = load_audio_file(out, Settings())
    assert (audio.container, audio.codec) == (container, codec)
    assert audio.sample_rate == 16_000
    assert audio.duration_s == pytest.approx(3.0, abs=0.1)


def test_clean_condition_is_the_original_file(tmp_path: Path):
    source = tmp_path / "a.wav"
    assert degrade(source, "clean", tmp_path) == source


def test_every_condition_is_described():
    assert set(CONDITIONS) == {"clean", "g711", "amr-nb", "opus", "mp3"}
    assert all(c.description for c in CONDITIONS.values())


# --- Scoring runner ----------------------------------------------------------------------


class FakeLogitDetector:
    def __init__(self) -> None:
        self.calls = 0

    def logits(self, windows):
        self.calls += 1
        return np.array([[2.0, -1.0]] * len(windows))  # margin 3.0 for every window


@needs_ffmpeg
def test_runner_scores_every_window_records_errors_and_resumes(tmp_path: Path):
    clips = []
    for i, seconds in enumerate([6, 0.5]):
        path = tmp_path / f"tone{i}.wav"
        subprocess.run(
            [FFMPEG, "-nostdin", "-loglevel", "error", "-f", "lavfi",
             "-i", f"sine=frequency=300:duration={seconds}", "-y", str(path)],
            check=True,
        )  # fmt: skip
        clips.append(Clip(path.name, path, i % 2, "spk"))
    broken = tmp_path / "broken.wav"
    broken.write_text("not audio")
    clips.append(Clip("broken.wav", broken, 0, "spk"))

    out = tmp_path / "scores.jsonl"
    detector = FakeLogitDetector()
    config = WindowingConfig(min_speech_s=0.0, min_speech_fraction=0.0)
    added = score_clips(
        clips, "clean", out, detector, Settings(), config, workers=2, progress=False
    )
    assert added == 3
    records = {r["id"]: r for r in map(json.loads, out.read_text().splitlines())}

    six = records["tone0.wav"]
    assert six["error"] is None and six["duration_s"] == pytest.approx(6.0, abs=0.05)
    assert len(six["windows"]) == 2 and all(w[3] == pytest.approx(3.0) for w in six["windows"])
    # The half-second clip is still scored: the product's 1 s minimum is applied later.
    assert len(records["tone1.wav"]["windows"]) == 1
    assert records["broken.wav"]["error"].startswith("UnsupportedFormatError")

    # Running again adds nothing: finished clips are skipped.
    assert score_clips(clips, "clean", out, detector, Settings(), config, progress=False) == 0
