"""Turn raw clip scores into calibrated probabilities (Platt scaling).

A raw score of 0.95 from this model doesn't mean "95% of clips like this are fake": NII's own
results put the EER threshold anywhere from 0.62 to 0.99 depending on the data (D9). Platt
scaling fits two numbers, a slope and an offset on the log-odds of the raw score:

    calibrated = sigmoid(a * logit(raw) + b)

Why Platt rather than a more flexible method such as isotonic regression: two parameters can't
overfit a dev set of a few thousand clips, and the mapping is smooth and strictly increasing,
so it never changes which clip ranks above which. Calibration changes what a score means, not
how well the model separates fakes; EER and AUC are identical before and after.

The fitted probabilities assume the share of fakes in the data they were fitted on. For other
base rates, `shift_prior` adjusts them (docs/decisions.md).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_EPS = 1e-7  # keeps logit finite for scores that saturate at exactly 0 or 1


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p) - np.log1p(-p)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class PlattCalibrator:
    a: float
    b: float
    fitted_fake_share: float  # share of fakes in the data the fit used

    def __call__(self, scores: Sequence[float] | np.ndarray) -> np.ndarray:
        raw = np.asarray(scores, dtype=np.float64)
        return _sigmoid(self.a * _logit(raw) + self.b)


def fit_platt(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    l2: float = 1e-4,
    max_iter: int = 100,
) -> PlattCalibrator:
    """Fit a and b by maximum likelihood (Newton's method), with a tiny L2 penalty on a.

    The penalty only matters when dev data is perfectly separable, where an unpenalized fit
    would push a toward infinity.
    """
    y = np.asarray(labels, dtype=np.float64)
    x = _logit(np.asarray(scores, dtype=np.float64))
    if y.size != x.size or y.size == 0:
        raise ValueError("labels and scores must be non-empty and the same length")
    if y.min() == y.max():
        raise ValueError("need both genuine and synthetic examples to calibrate")
    features = np.column_stack([x, np.ones_like(x)])
    w = np.zeros(2)
    penalty = np.diag([l2 * y.size, 0.0])
    for _ in range(max_iter):
        p = _sigmoid(features @ w)
        gradient = features.T @ (p - y) + penalty @ w
        hessian = features.T @ (features * (p * (1 - p))[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return PlattCalibrator(a=float(w[0]), b=float(w[1]), fitted_fake_share=float(y.mean()))


def shift_prior(
    probs: Sequence[float] | np.ndarray, from_share: float, to_share: float
) -> np.ndarray:
    """Re-express probabilities for a different share of fakes (Bayes' rule on the odds).

    A probability calibrated on data that is 40% fake overstates the chance a clip is fake when
    only 1 in 1,000 real-world recordings is. This keeps the evidence and swaps the prior.
    """
    for share in (from_share, to_share):
        if not 0.0 < share < 1.0:
            raise ValueError("shares must be between 0 and 1, exclusive")
    p = np.clip(np.asarray(probs, dtype=np.float64), _EPS, 1 - _EPS)
    odds = p / (1 - p) * (to_share / (1 - to_share)) / (from_share / (1 - from_share))
    return odds / (1 + odds)
