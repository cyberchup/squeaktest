"""Tests for the evaluation metrics, against hand-worked examples and known theory.

Two normal distributions one standard deviation apart (genuine ~ N(0, 1), synthetic ~ N(d, 1))
have a known EER of Phi(-d/2) and a known detection rate at false-positive rate f of
1 - Phi(z - d), where z = Phi^-1(1 - f). With many samples, the metrics must land on them.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pytest

from squeaktest.metrics import (
    at_fpr,
    bootstrap,
    brier_score,
    eer,
    expected_calibration_error,
    precision_at_prevalence,
    reliability_bins,
    roc_auc,
    roc_points,
)

PHI = NormalDist().cdf


def gaussians(d: float, n: int = 200_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scores = np.concatenate([rng.normal(0, 1, n), rng.normal(d, 1, n)])
    labels = np.concatenate([np.zeros(n), np.ones(n)])
    return labels, scores


# --- ROC and AUC -------------------------------------------------------------------------


def test_roc_points_on_a_small_example():
    #            genuine genuine synthetic genuine synthetic synthetic
    labels = [0, 0, 1, 0, 1, 1]
    scores = [0.1, 0.3, 0.35, 0.6, 0.8, 0.9]
    fpr, tpr, thresholds = roc_points(labels, scores)
    np.testing.assert_allclose(tpr, [0, 1 / 3, 2 / 3, 2 / 3, 1, 1, 1])
    np.testing.assert_allclose(fpr, [0, 0, 0, 1 / 3, 1 / 3, 2 / 3, 1])
    assert thresholds[0] == np.inf


def test_tied_scores_move_together():
    fpr, tpr, _ = roc_points([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.9])
    # One point for the 0.9, then one point for all three 0.5s at once
    np.testing.assert_allclose(tpr, [0, 0.5, 1.0])
    np.testing.assert_allclose(fpr, [0, 0.0, 1.0])


def test_auc_extremes_and_ties():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_auc_matches_counting_every_pair():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 300)
    scores = np.round(rng.random(300), 2)  # rounding creates ties
    pos, neg = scores[labels == 1], scores[labels == 0]
    pairs = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    assert roc_auc(labels, scores) == pytest.approx(pairs)


def test_auc_for_gaussians_matches_theory():
    labels, scores = gaussians(d=1.0)
    assert roc_auc(labels, scores) == pytest.approx(PHI(1.0 / math.sqrt(2)), abs=0.003)


# --- EER ---------------------------------------------------------------------------------


def test_eer_perfect_separation_is_zero():
    rate, _ = eer([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert rate == 0.0


def test_eer_of_identical_scores_is_one_half():
    rate, _ = eer([0, 1] * 50, [0.5] * 100)
    assert rate == pytest.approx(0.5)


@pytest.mark.parametrize("d", [0.5, 1.0, 2.0, 3.0])
def test_eer_for_gaussians_matches_theory(d: float):
    labels, scores = gaussians(d)
    rate, threshold = eer(labels, scores)
    assert rate == pytest.approx(PHI(-d / 2), abs=0.003)
    assert threshold == pytest.approx(d / 2, abs=0.02)  # the midpoint, by symmetry


def test_eer_ignores_how_scores_are_scaled():
    labels, scores = gaussians(1.5, n=20_000)
    squashed = 1 / (1 + np.exp(-scores))  # any increasing transform
    assert eer(labels, scores)[0] == pytest.approx(eer(labels, squashed)[0])


# --- Operating points --------------------------------------------------------------------


@pytest.mark.parametrize("max_fpr", [0.01, 0.05])
def test_detection_rate_at_fixed_fpr_matches_theory(max_fpr: float):
    d = 2.0
    labels, scores = gaussians(d)
    point = at_fpr(labels, scores, max_fpr)
    z = NormalDist().inv_cdf(1 - max_fpr)
    assert point.fpr <= max_fpr
    assert point.tpr == pytest.approx(1 - PHI(z - d), abs=0.005)


def test_operating_point_never_exceeds_the_fpr_budget():
    labels = [0, 0, 0, 0, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.9, 0.8, 0.95]
    point = at_fpr(labels, scores, max_fpr=0.2)
    assert point.fpr == 0.0  # flagging the 0.9 genuine clip would cost 25% FPR
    assert point.tpr == 0.5
    assert point.threshold == 0.95


def test_precision_at_realistic_prevalence():
    # docs/research.md section 4.1: 90% caught at 1% FPR, 1 fake in 1,000 -> 8.3% precision
    assert precision_at_prevalence(0.9, 0.01, 0.001) == pytest.approx(0.0826, abs=1e-4)
    assert precision_at_prevalence(0.9, 0.05, 0.001) == pytest.approx(0.0177, abs=1e-4)


# --- Calibration -------------------------------------------------------------------------


def test_perfectly_calibrated_predictions_have_near_zero_ece():
    rng = np.random.default_rng(2)
    probs = rng.random(200_000)
    labels = (rng.random(200_000) < probs).astype(int)  # outcomes drawn at the stated rate
    assert expected_calibration_error(labels, probs) < 0.005


def test_overconfident_predictions_have_large_ece():
    # Says 0.99 every time, but only half are synthetic
    labels = np.array([0, 1] * 500)
    probs = np.full(1000, 0.99)
    assert expected_calibration_error(labels, probs) == pytest.approx(0.49)


def test_reliability_bins_report_predicted_and_observed_rates():
    labels = [0, 0, 1, 1, 1]
    probs = [0.05, 0.15, 0.15, 0.95, 0.85]
    bins = reliability_bins(labels, probs, n_bins=10)
    by_low = {round(b.low, 1): b for b in bins}
    assert by_low[0.1].count == 2 and by_low[0.1].fraction_synthetic == 0.5
    assert by_low[0.8].fraction_synthetic == 1.0
    assert by_low[0.9].mean_predicted == pytest.approx(0.95)
    assert sum(b.count for b in bins) == 5


def test_probability_of_exactly_one_lands_in_the_top_bin():
    bins = reliability_bins([0, 1], [0.0, 1.0], n_bins=10)
    assert [b.low for b in bins] == [0.0, 0.9]


def test_brier_score():
    assert brier_score([0, 1], [0.0, 1.0]) == 0.0
    assert brier_score([0, 1], [0.5, 0.5]) == 0.25


# --- Bootstrap ---------------------------------------------------------------------------


def test_bootstrap_interval_contains_the_estimate_and_narrows_with_data():
    small_labels, small_scores = gaussians(1.0, n=200)
    large_labels, large_scores = gaussians(1.0, n=5_000)
    small = bootstrap(roc_auc, small_labels, small_scores, n_resamples=300)
    large = bootstrap(roc_auc, large_labels, large_scores, n_resamples=300)
    assert small.low <= small.estimate <= small.high
    assert (large.high - large.low) < (small.high - small.low)


def test_resampling_speakers_widens_intervals_when_clips_are_correlated():
    # 20 speakers, 50 clips each. Each speaker shifts all their scores together, so clips of
    # one speaker are correlated, and resampling single clips overstates the precision.
    rng = np.random.default_rng(3)
    labels, scores, groups = [], [], []
    for speaker in range(20):
        label = speaker % 2
        offset = rng.normal(0, 1.0)
        for _ in range(50):
            labels.append(label)
            scores.append(label * 1.0 + offset + rng.normal(0, 0.3))
            groups.append(f"speaker-{speaker}")
    by_clip = bootstrap(roc_auc, labels, scores, n_resamples=400)
    by_speaker = bootstrap(roc_auc, labels, scores, groups=groups, n_resamples=400)
    assert (by_speaker.high - by_speaker.low) > 2 * (by_clip.high - by_clip.low)


# --- Input checks ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("labels", "scores"),
    [
        ([0, 1], [0.5]),  # length mismatch
        ([0, 2], [0.1, 0.9]),  # bad label
        ([1, 1], [0.1, 0.9]),  # one class only
        ([0, 1], [0.1, float("nan")]),
    ],
)
def test_bad_inputs_are_rejected(labels, scores):
    with pytest.raises(ValueError):
        eer(labels, scores)
