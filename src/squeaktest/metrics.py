"""Evaluation metrics, in plain NumPy. Convention: label 1 = synthetic, 0 = genuine, and a
higher score means "more likely synthetic".

What each metric is for:
- EER (equal error rate): the error rate at the threshold where false positives and missed
  fakes are equally common. One number for comparing models; not an operating point.
- ROC-AUC: the probability that a random fake scores higher than a random genuine clip.
- Detection rate at a fixed false-positive rate: the SOC view. "If we accept flagging 1% of
  genuine calls, what share of fakes do we catch?"
- Calibration (reliability bins, ECE, Brier score): whether a score of 0.8 means 80%.
- Bootstrap confidence intervals: how much a number could move with different test clips.
  Resampling whole speakers, not single clips, because clips of one speaker aren't independent.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

ArrayLike = Sequence[float] | np.ndarray


def _validated(labels: ArrayLike, scores: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels).astype(int).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.size != s.size:
        raise ValueError("labels and scores must have the same length")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 (genuine) or 1 (synthetic)")
    if not (y == 1).any() or not (y == 0).any():
        raise ValueError("need at least one genuine and one synthetic example")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    return y, s


def roc_points(labels: ArrayLike, scores: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """False-positive rate, true-positive rate and threshold at every distinct score.

    "Flag as synthetic if score >= threshold". The first point is a threshold above every score
    (nothing flagged: FPR 0, TPR 0); thresholds then decrease. Tied scores move together, which
    matters with saturated models whose scores pile up at 0.9999.
    """
    y, s = _validated(labels, scores)
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    # Index of the last element of each run of equal scores.
    last_of_run = np.flatnonzero(np.diff(s_sorted, append=-np.inf) != 0)
    tp = np.cumsum(y_sorted)[last_of_run]
    fp = (last_of_run + 1) - tp
    tpr = np.concatenate([[0.0], tp / y.sum()])
    fpr = np.concatenate([[0.0], fp / (y.size - y.sum())])
    thresholds = np.concatenate([[np.inf], s_sorted[last_of_run]])
    return fpr, tpr, thresholds


def roc_auc(labels: ArrayLike, scores: ArrayLike) -> float:
    """Area under the ROC curve, computed from ranks (ties count half)."""
    y, s = _validated(labels, scores)
    ranks = _average_ranks(s)
    n_pos = y.sum()
    n_neg = y.size - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1)
    # Give tied values the mean of their ranks.
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    return (sums / counts)[inverse]


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    fpr: float  # share of genuine clips flagged
    tpr: float  # share of synthetic clips caught


def eer(labels: ArrayLike, scores: ArrayLike) -> tuple[float, float]:
    """Equal error rate and the threshold where it occurs.

    The EER lies where the false-positive rate (rising as the threshold drops) meets the
    missed-fake rate (falling). Between the two ROC points that bracket the crossing, the
    rates are interpolated linearly, as in ASVspoof's reference implementation.
    """
    fpr, tpr, thresholds = roc_points(labels, scores)
    fnr = 1.0 - tpr
    i = int(np.argmax(fpr >= fnr))  # first point where false positives catch up
    if i == 0:
        return 0.0, float(thresholds[0])
    d_fpr, d_fnr = fpr[i] - fpr[i - 1], fnr[i] - fnr[i - 1]
    alpha = (fnr[i - 1] - fpr[i - 1]) / (d_fpr - d_fnr) if d_fpr != d_fnr else 0.0
    return float(fpr[i - 1] + alpha * d_fpr), float(thresholds[i])


def at_fpr(labels: ArrayLike, scores: ArrayLike, max_fpr: float) -> OperatingPoint:
    """The operating point catching the most fakes while flagging at most `max_fpr` of genuine.

    No interpolation: this is a threshold you could actually deploy, so it never promises a
    false-positive rate the data didn't show.
    """
    if not 0.0 <= max_fpr <= 1.0:
        raise ValueError("max_fpr must be between 0 and 1")
    fpr, tpr, thresholds = roc_points(labels, scores)
    allowed = np.flatnonzero(fpr <= max_fpr)
    best = allowed[np.argmax(tpr[allowed])]
    return OperatingPoint(float(thresholds[best]), float(fpr[best]), float(tpr[best]))


@dataclass(frozen=True)
class ReliabilityBin:
    low: float
    high: float
    mean_predicted: float
    fraction_synthetic: float
    count: int


def reliability_bins(labels: ArrayLike, probs: ArrayLike, n_bins: int = 10) -> list[ReliabilityBin]:
    """Group predictions into equal-width bins and compare predicted vs observed rates.

    For a calibrated model, each bin's fraction of synthetic clips matches its mean prediction.
    This is the data behind a reliability diagram. Empty bins are left out.
    """
    y, p = _validated(labels, probs)
    if (p < 0).any() or (p > 1).any():
        raise ValueError("probabilities must be between 0 and 1")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = index == b
        if mask.any():
            bins.append(
                ReliabilityBin(
                    low=float(edges[b]),
                    high=float(edges[b + 1]),
                    mean_predicted=float(p[mask].mean()),
                    fraction_synthetic=float(y[mask].mean()),
                    count=int(mask.sum()),
                )
            )
    return bins


def expected_calibration_error(labels: ArrayLike, probs: ArrayLike, n_bins: int = 10) -> float:
    """Average gap between predicted and observed rates, weighted by how full each bin is."""
    bins = reliability_bins(labels, probs, n_bins)
    total = sum(b.count for b in bins)
    return float(sum(b.count * abs(b.mean_predicted - b.fraction_synthetic) for b in bins) / total)


def brier_score(labels: ArrayLike, probs: ArrayLike) -> float:
    """Mean squared error of probabilities. Rewards both accuracy and honest confidence."""
    y, p = _validated(labels, probs)
    return float(np.mean((p - y) ** 2))


def precision_at_prevalence(tpr: float, fpr: float, prevalence: float) -> float:
    """Share of alerts that are real fakes when only `prevalence` of all traffic is fake.

    Test sets are often a third or more fake; real voicemail is far less. At 1 fake in 1,000,
    even a 1% false-positive rate means most alerts are false.
    """
    if not 0.0 < prevalence < 1.0:
        raise ValueError("prevalence must be between 0 and 1, exclusive")
    caught = tpr * prevalence
    false_alarms = fpr * (1.0 - prevalence)
    return caught / (caught + false_alarms) if caught + false_alarms > 0 else 0.0


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


def bootstrap(
    metric: Callable[[np.ndarray, np.ndarray], float],
    labels: ArrayLike,
    scores: ArrayLike,
    groups: Sequence[str] | np.ndarray | None = None,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """A percentile bootstrap confidence interval for `metric(labels, scores)`.

    With `groups` (speaker names), whole groups are resampled, because clips from one speaker
    share voice, microphone and recording conditions and aren't independent. Ignoring that makes
    intervals look tighter than the evidence supports. Resamples that happen to contain only one
    class are skipped.
    """
    y, s = _validated(labels, scores)
    rng = np.random.default_rng(seed)
    if groups is None:
        members = [np.array([i]) for i in range(y.size)]
    else:
        g = np.asarray(groups)
        if g.size != y.size:
            raise ValueError("groups must have one entry per example")
        members = [np.flatnonzero(g == name) for name in np.unique(g)]
    values = []
    for _ in range(n_resamples):
        picked = rng.integers(0, len(members), size=len(members))
        idx = np.concatenate([members[k] for k in picked])
        if y[idx].min() == y[idx].max():
            continue
        values.append(metric(y[idx], s[idx]))
    if not values:
        raise ValueError("no usable resamples: each one contained only one class")
    tail = (1.0 - confidence) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return Interval(estimate=metric(y, s), low=float(low), high=float(high))
