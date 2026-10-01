"""Tests for windowing and aggregation (step 4).

A toy scorer stands in for the model: 440 Hz tones play "real speech" and 880 Hz tones play
"synthetic speech", and the scorer reports the share of a window's energy at 880 Hz. That
makes the effects of the windowing and aggregation choices visible in plain numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import pytest

from squeaktest.inference import (
    WindowingConfig,
    aggregate,
    analyze,
    band_for,
    plan_windows,
)

SR = 16_000
REAL_HZ = 440.0
FAKE_HZ = 880.0


def tone(seconds: float, hz: float = REAL_HZ, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(round(seconds * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(round(seconds * SR), dtype=np.float32)


def clip(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts)


def _band_energy(spectrum: np.ndarray, freqs: np.ndarray, hz: float) -> float:
    return float(spectrum[(freqs > hz - 30) & (freqs < hz + 30)].sum())


def toy_scorer(windows: Sequence[np.ndarray]) -> list[float]:
    """Share of each window's tone energy that is at the 'fake' frequency."""
    scores = []
    for w in windows:
        spectrum = np.abs(np.fft.rfft(w)) ** 2
        freqs = np.fft.rfftfreq(w.size, d=1 / SR)
        fake = _band_energy(spectrum, freqs, FAKE_HZ)
        real = _band_energy(spectrum, freqs, REAL_HZ)
        scores.append(fake / (fake + real) if fake + real > 0 else 0.0)
    return scores


# --- Window planning ---------------------------------------------------------------------


def test_windows_cover_the_whole_speech_span_without_gaps():
    windows = plan_windows(tone(10.0), SR)
    window, hop = 4 * SR, 2 * SR
    assert windows[0].start == 0
    for prev, nxt in pairwise(windows):
        assert nxt.start - prev.start <= hop  # never a jump bigger than the hop
        assert nxt.start <= prev.end  # never a gap between windows
    assert all(w.end - w.start == window for w in windows)
    # The last window reaches the end of the detected speech (the final partial VAD frame
    # is dropped, so that's within one 25 ms frame of the clip's end).
    assert tone(10.0).size - windows[-1].end < 400


def test_an_end_aligned_window_covers_the_tail():
    # 10 s doesn't divide into 4 s windows every 2 s, so without an extra window aligned to
    # the end, the last seconds would never be scored.
    windows = plan_windows(tone(10.0), SR)
    on_grid = [w for w in windows if w.start % (2 * SR) == 0]
    assert on_grid[-1].end < 9 * SR  # the regular 2 s grid stops at 8 s...
    assert windows[-1].end > 9.9 * SR  # ...so an extra end-aligned window covers the rest


def test_a_clip_shorter_than_one_window_gets_a_single_window():
    windows = plan_windows(tone(3.0), SR)
    assert len(windows) == 1
    assert windows[0].start == 0
    assert windows[0].end == pytest.approx(3 * SR, abs=400)


def test_leading_and_trailing_silence_are_trimmed_before_windowing():
    windows = plan_windows(clip(silence(1.0), tone(6.0), silence(1.0)), SR)
    assert abs(windows[0].start - 1 * SR) <= 400
    assert abs(windows[-1].end - 7 * SR) <= 400


def test_silence_gives_no_windows():
    assert plan_windows(silence(5.0), SR) == []


def test_too_little_speech_gives_no_windows():
    assert plan_windows(clip(silence(2.0), tone(0.5), silence(2.0)), SR) == []


def test_windows_carry_their_speech_fraction():
    windows = plan_windows(clip(tone(4.0), silence(8.0), tone(4.0)), SR)
    fractions = [w.speech_fraction for w in windows]
    assert max(fractions) == pytest.approx(1.0)
    assert min(fractions) == pytest.approx(0.0, abs=0.01)  # a window inside the pause


# --- Aggregation -------------------------------------------------------------------------


def test_top_1_is_the_max_and_top_n_is_the_mean():
    scores = [0.1, 0.9, 0.3, 0.5]
    assert aggregate(scores, top_k=1) == pytest.approx(0.9)
    assert aggregate(scores, top_k=4) == pytest.approx(np.mean(scores))
    assert aggregate(scores, top_k=100) == pytest.approx(np.mean(scores))


def test_top_k_averages_the_highest_scores():
    assert aggregate([0.2, 0.9, 0.1, 0.8, 0.7], top_k=3) == pytest.approx(0.8)


def test_aggregation_ignores_order():
    assert aggregate([0.9, 0.1, 0.5], 2) == aggregate([0.1, 0.5, 0.9], 2)


def test_one_spike_does_not_flag_a_clip_with_top_3():
    # One window spikes (a cough, a codec glitch, music) in an otherwise clean call.
    # max would report 0.97; top-3 needs the suspicion to persist.
    scores = [0.05] * 10 + [0.97]
    assert aggregate(scores, top_k=1) > 0.9
    assert aggregate(scores, top_k=3) < 0.5


@pytest.mark.parametrize("bad", [[], np.array([])])
def test_aggregating_nothing_is_an_error(bad):
    with pytest.raises(ValueError):
        aggregate(bad, top_k=3)


def test_top_k_must_be_positive():
    with pytest.raises(ValueError):
        aggregate([0.5], top_k=0)


# --- End to end with the toy scorer ------------------------------------------------------


def test_partial_fake_is_caught_by_top_k_but_diluted_by_the_mean():
    # 10 s real, 4 s synthetic, 10 s real: a cloned sentence spliced into a real call.
    audio = clip(tone(10.0), tone(4.0, FAKE_HZ), tone(10.0))
    result_top3 = analyze(audio, SR, toy_scorer)
    result_mean = analyze(audio, SR, toy_scorer, WindowingConfig(top_k=1000))
    result_max = analyze(audio, SR, toy_scorer, WindowingConfig(top_k=1))
    assert result_mean.score is not None and result_mean.score < 0.25
    assert result_top3.score is not None and result_top3.score > 0.6
    assert result_max.score is not None and result_max.score > 0.95


def test_window_scores_locate_the_fake_on_the_timeline():
    audio = clip(tone(10.0), tone(4.0, FAKE_HZ), tone(10.0))
    result = analyze(audio, SR, toy_scorer)
    peak = max(result.windows, key=lambda w: w.score or 0.0)
    assert peak.start_s == pytest.approx(10.0)
    assert peak.end_s == pytest.approx(14.0)


def test_fully_synthetic_and_fully_real_clips():
    assert analyze(tone(8.0, FAKE_HZ), SR, toy_scorer).score == pytest.approx(1.0, abs=0.01)
    assert analyze(tone(8.0), SR, toy_scorer).score == pytest.approx(0.0, abs=0.01)


def test_mostly_silent_windows_are_skipped_not_scored():
    audio = clip(tone(4.0), silence(8.0), tone(4.0))
    calls: list[int] = []

    def scorer(windows: Sequence[np.ndarray]) -> list[float]:
        calls.append(len(windows))
        return [0.3] * len(windows)

    result = analyze(audio, SR, scorer)
    skipped = [w for w in result.windows if w.score is None]
    scored = [w for w in result.windows if w.score is not None]
    assert skipped and scored
    assert all(w.speech_fraction < 0.5 for w in skipped)
    assert calls == [len(scored)]  # one batch, containing only the windows with speech
    assert result.score == pytest.approx(0.3)


def test_too_little_speech_gives_no_score_and_says_why():
    result = analyze(clip(silence(2.0), tone(0.5), silence(2.0)), SR, toy_scorer)
    assert result.score is None
    assert result.windows == ()
    assert result.note is not None and "not enough" in result.note


def test_sparse_sound_gives_no_score_when_every_window_is_mostly_silent():
    # Short clicks spread out: enough total sound to pass the minimum, but no window is
    # more than about 10% active, so nothing gets scored.
    burst, gap = tone(0.1), silence(0.9)
    audio = clip(*([burst, gap] * 20))
    result = analyze(audio, SR, toy_scorer, WindowingConfig(min_speech_s=1.0))
    assert result.score is None
    assert result.windows and all(w.score is None for w in result.windows)
    assert result.note is not None and "mostly silence" in result.note


def test_speech_seconds_are_reported():
    result = analyze(clip(silence(1.0), tone(5.0), silence(1.0)), SR, toy_scorer)
    assert result.speech_s == pytest.approx(5.0, abs=0.05)


def test_timeline_is_in_clip_seconds_and_in_order():
    result = analyze(clip(silence(1.0), tone(9.0)), SR, toy_scorer)
    starts = [w.start_s for w in result.windows]
    assert starts == sorted(starts)
    assert starts[0] == pytest.approx(1.0, abs=0.03)


# --- Guarding against a misbehaving scorer -----------------------------------------------


@pytest.mark.parametrize(
    "bad_output",
    [
        lambda n: [0.5] * (n + 1),  # wrong number of scores
        lambda n: [float("nan")] * n,
        lambda n: [1.5] * n,
        lambda n: [-0.1] * n,
    ],
    ids=["count", "nan", "above-1", "below-0"],
)
def test_bad_scorer_output_is_rejected(bad_output):
    def scorer(windows: Sequence[np.ndarray]) -> list[float]:
        return bad_output(len(windows))

    with pytest.raises(ValueError, match="scorer returned"):
        analyze(tone(8.0), SR, scorer)


# --- Config ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_s": 0},
        {"hop_s": 0},
        {"hop_s": 5.0},  # longer than the 4 s window: gaps
        {"min_speech_fraction": 1.5},
        {"min_speech_s": -1},
        {"top_k": 0},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        WindowingConfig(**kwargs)


# --- Bands -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "band"),
    [
        (None, "not_assessed"),
        (0.0, "likely_genuine"),
        (0.4999, "likely_genuine"),
        (0.5, "uncertain"),
        (0.8999, "uncertain"),
        (0.9, "likely_synthetic"),
        (1.0, "likely_synthetic"),
    ],
)
def test_band_thresholds(score, band):
    assert band_for(score) == band
