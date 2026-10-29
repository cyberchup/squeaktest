"""Tests for the K1 measurement: the generated non-speech signals, the mixes and the runner."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import HAVE_FFMPEG

from eval.k1 import CachedScorer, encode
from eval.nonspeech import MIXES, SIGNALS, SR, signal
from squeaktest.config import Settings
from squeaktest.vad import TONE_SHARE, tone_share

ALL_SIGNALS = [(split, s) for split, signals in SIGNALS.items() for s in signals]


@pytest.mark.parametrize(("split", "sig"), ALL_SIGNALS, ids=lambda v: getattr(v, "name", v))
def test_signals_have_the_asked_length_and_a_speech_level(split, sig):
    for seconds in (0.3, 5.0):
        x = sig.make(seconds)
        assert x.dtype == np.float32 and x.shape == (round(seconds * SR),)
        assert np.isfinite(x).all() and np.abs(x).max() <= 1.0
    assert np.sqrt(np.mean(np.square(sig.make(5.0)))) == pytest.approx(0.1, rel=0.01)


def test_signals_are_the_same_every_time():
    for _split, sig in ALL_SIGNALS:
        assert np.array_equal(sig.make(2.0), sig.make(2.0)), sig.name


def test_dev_and_test_signals_differ():
    # The gate is tuned on dev signals and reported on test ones, so no signal may be reused.
    for dev, test in zip(SIGNALS["dev"], SIGNALS["test"], strict=True):
        assert dev.family == test.family
        assert not np.array_equal(dev.make(2.0), test.make(2.0)), dev.name


def test_a_steady_test_tone_reads_as_a_tone():
    assert tone_share(signal("test", "tone").make(2.0), SR)[30:-30].min() >= TONE_SHARE


@pytest.mark.parametrize("mix", MIXES, ids=lambda m: m.name)
def test_mixes_keep_the_speech(mix):
    speech = (0.1 * np.sin(2 * np.pi * 200 * np.arange(SR * 3) / SR)).astype(np.float32)
    mixed = mix.make(speech, "dev")
    assert mixed.dtype == np.float32 and mixed.size >= speech.size
    if mix.name == "original":
        assert np.array_equal(mixed, speech)


def test_cached_scorer_scores_each_distinct_window_once():
    calls = []

    def detector(windows):
        calls.append(len(windows))
        return [float(w[0]) for w in windows]

    scorer = CachedScorer(detector)
    a, b = np.full(4, 0.25, np.float32), np.full(4, 0.75, np.float32)
    assert scorer([a, b, a]) == [0.25, 0.75, 0.25]
    assert scorer([b, a]) == [0.75, 0.25]
    assert calls == [2]


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg and ffprobe needed")
def test_encode_goes_through_the_codec_and_the_loader():
    tone = signal("dev", "tone").make(2.0)
    audio = encode(tone, "g711", Settings())
    assert audio.sample_rate == SR
    assert audio.duration_s == pytest.approx(2.0, abs=0.05)
