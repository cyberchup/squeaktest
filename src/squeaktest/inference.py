"""Split a clip into overlapping windows, score each one, and combine them into a clip score.

The pipeline:

1. Trim leading and trailing silence (vad.speech_bounds). Silence length is a known shortcut
   for detectors, so it never reaches the model.
2. Refuse to score clips with less than `min_speech_s` of activity. A number computed from
   half a second of audio would look as confident as any other, and wouldn't deserve to.
3. Cut the trimmed span into windows of `window_s`, starting every `hop_s`. If the windows
   don't reach the end, add one more window aligned to the end, so no audio goes unscored.
4. Skip windows that are mostly silent (speech fraction below `min_speech_fraction`).
5. Score the remaining windows in one batch, then combine the scores with a top-k mean.

The reasons behind each default, and their effect on missed fakes, false positives and cost,
are in docs/decisions.md. Scores here are raw model outputs, not calibrated probabilities;
calibration comes in Phase 2.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from squeaktest.vad import speech_bounds, speech_fraction

# A window scorer takes a batch of mono sample arrays and returns one score per window, each
# in [0, 1], where higher means more likely synthetic. Batching lets a model run them together.
WindowScorer = Callable[[Sequence[np.ndarray]], Sequence[float] | np.ndarray]

# Provisional band thresholds on the raw, uncalibrated score (docs/decisions.md D14).
# Phase 2 replaces them with thresholds chosen on dev data at a target false-positive rate.
LIKELY_SYNTHETIC_AT = 0.90
LIKELY_GENUINE_BELOW = 0.50

BANDS = ("likely_synthetic", "uncertain", "likely_genuine", "not_assessed")


def band_for(score: float | None) -> str:
    """Map a clip score to a band. No score means "not assessed", never "likely genuine"."""
    if score is None:
        return "not_assessed"
    if score >= LIKELY_SYNTHETIC_AT:
        return "likely_synthetic"
    if score < LIKELY_GENUINE_BELOW:
        return "likely_genuine"
    return "uncertain"


@dataclass(frozen=True)
class WindowingConfig:
    window_s: float = 4.0
    hop_s: float = 2.0
    min_speech_fraction: float = 0.5
    min_speech_s: float = 1.0
    top_k: int = 3

    def __post_init__(self) -> None:
        if not (math.isfinite(self.window_s) and self.window_s > 0):
            raise ValueError("window_s must be positive")
        if not (math.isfinite(self.hop_s) and 0 < self.hop_s <= self.window_s):
            # A hop longer than the window would leave unscored gaps between windows.
            raise ValueError("hop_s must be positive and no longer than window_s")
        if not 0.0 <= self.min_speech_fraction <= 1.0:
            raise ValueError("min_speech_fraction must be between 0 and 1")
        if not (math.isfinite(self.min_speech_s) and self.min_speech_s >= 0):
            raise ValueError("min_speech_s must be zero or positive")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")


@dataclass(frozen=True)
class Window:
    """A window of the original clip, as sample indices [start, end)."""

    start: int
    end: int
    speech_fraction: float


@dataclass(frozen=True)
class WindowResult:
    start_s: float
    end_s: float
    speech_fraction: float
    score: float | None  # None when the window was skipped for being mostly silent


@dataclass(frozen=True)
class ClipResult:
    score: float | None  # None when the clip had too little speech to score
    windows: tuple[WindowResult, ...]
    speech_s: float  # seconds of detected activity
    note: str | None = None  # why there is no score, when there isn't one


def plan_windows(
    samples: np.ndarray, sample_rate: int, config: WindowingConfig | None = None
) -> list[Window]:
    """Return the windows to consider, in clip order. Empty if there's too little speech."""
    config = config or WindowingConfig()
    bounds = speech_bounds(samples, sample_rate)
    if bounds is None:
        return []
    span_start, span_end = bounds
    if _active_seconds(samples[span_start:span_end], sample_rate) < config.min_speech_s:
        return []

    window = round(config.window_s * sample_rate)
    hop = round(config.hop_s * sample_rate)
    span = span_end - span_start
    if span <= window:
        starts = [span_start]
    else:
        starts = list(range(span_start, span_end - window + 1, hop))
        if starts[-1] + window < span_end:
            starts.append(span_end - window)  # end-aligned window: no unscored tail

    windows = []
    for start in starts:
        end = min(start + window, span_end)
        windows.append(Window(start, end, speech_fraction(samples[start:end], sample_rate)))
    return windows


def aggregate(scores: Sequence[float] | np.ndarray, top_k: int) -> float:
    """Combine window scores into one: the mean of the `top_k` highest.

    top_k=1 is the max (most sensitive); top_k >= len(scores) is the plain mean (most
    forgiving). With 50% window overlap, every moment of audio lands in two windows, so
    top_k=3 needs suspicion that persists beyond a single moment before the clip score rises.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot aggregate an empty list of scores")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    k = min(top_k, values.size)
    return float(np.sort(values)[-k:].mean())


def analyze(
    samples: np.ndarray,
    sample_rate: int,
    scorer: WindowScorer,
    config: WindowingConfig | None = None,
) -> ClipResult:
    """Window the clip, score the windows that contain speech, and aggregate."""
    config = config or WindowingConfig()
    windows = plan_windows(samples, sample_rate, config)
    speech_s = _active_seconds(samples, sample_rate)
    if not windows:
        return ClipResult(
            score=None,
            windows=(),
            speech_s=speech_s,
            note=f"less than {config.min_speech_s:g} s with sound; not enough to score",
        )

    scoreable = [w for w in windows if w.speech_fraction >= config.min_speech_fraction]
    if not scoreable:
        return ClipResult(
            score=None,
            windows=tuple(_result(w, sample_rate, None) for w in windows),
            speech_s=speech_s,
            note="every window was mostly silence; not enough to score",
        )

    scores = _checked_scores(scorer([samples[w.start : w.end] for w in scoreable]), scoreable)
    score_of = dict(zip(scoreable, scores, strict=True))
    return ClipResult(
        score=aggregate(scores, config.top_k),
        windows=tuple(_result(w, sample_rate, score_of.get(w)) for w in windows),
        speech_s=speech_s,
    )


def _checked_scores(raw: Sequence[float] | np.ndarray, windows: list[Window]) -> list[float]:
    # The scorer is a model: check its output rather than trusting it, so a shape bug or a
    # NaN can't silently become a confident-looking clip score.
    scores = np.asarray(raw, dtype=np.float64).reshape(-1)
    if scores.size != len(windows):
        raise ValueError(f"scorer returned {scores.size} scores for {len(windows)} windows")
    if not (np.isfinite(scores).all() and (scores >= 0).all() and (scores <= 1).all()):
        raise ValueError("scorer returned scores outside [0, 1]")
    return [float(s) for s in scores]


def _result(window: Window, sample_rate: int, score: float | None) -> WindowResult:
    return WindowResult(
        start_s=window.start / sample_rate,
        end_s=window.end / sample_rate,
        speech_fraction=window.speech_fraction,
        score=score,
    )


def _active_seconds(samples: np.ndarray, sample_rate: int) -> float:
    return speech_fraction(samples, sample_rate) * samples.size / sample_rate
