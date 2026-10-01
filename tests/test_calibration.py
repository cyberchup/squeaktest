from __future__ import annotations

import numpy as np
import pytest

from squeaktest.calibration import PlattCalibrator, fit_platt, shift_prior
from squeaktest.metrics import eer, expected_calibration_error, roc_auc


def overconfident_scores(n: int = 20_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Scores whose true log-odds are half of what they claim: a typical overconfident model."""
    rng = np.random.default_rng(seed)
    true_logodds = rng.normal(0, 2, n)
    labels = (rng.random(n) < 1 / (1 + np.exp(-true_logodds))).astype(int)
    claimed = 1 / (1 + np.exp(-2 * true_logodds))  # twice as extreme as the truth
    return labels, claimed


def test_platt_recovers_the_true_slope():
    labels, scores = overconfident_scores()
    calibrator = fit_platt(labels, scores)
    assert calibrator.a == pytest.approx(0.5, abs=0.05)
    assert calibrator.b == pytest.approx(0.0, abs=0.05)


def test_calibration_fixes_overconfidence_without_changing_ranking():
    labels, scores = overconfident_scores()
    calibrated = fit_platt(labels, scores)(scores)
    assert (
        expected_calibration_error(labels, calibrated)
        < expected_calibration_error(labels, scores) / 3
    )
    assert roc_auc(labels, calibrated) == pytest.approx(roc_auc(labels, scores))
    assert eer(labels, calibrated)[0] == pytest.approx(eer(labels, scores)[0])


def test_separable_data_gives_a_finite_fit():
    calibrator = fit_platt([0, 0, 1, 1], [0.01, 0.02, 0.98, 0.99])
    assert np.isfinite([calibrator.a, calibrator.b]).all()
    out = calibrator([0.01, 0.99])
    assert out[0] < 0.5 < out[1]


def test_saturated_scores_stay_finite():
    calibrator = PlattCalibrator(a=0.5, b=0.0, fitted_fake_share=0.4)
    out = calibrator([0.0, 1.0])
    assert np.isfinite(out).all() and 0 < out[0] < out[1] < 1


def test_fit_records_the_share_of_fakes():
    assert fit_platt([0, 0, 0, 1], [0.1, 0.2, 0.3, 0.9]).fitted_fake_share == 0.25


@pytest.mark.parametrize("labels", [[1, 1], []])
def test_fit_needs_both_classes(labels):
    with pytest.raises(ValueError):
        fit_platt(labels, [0.5] * len(labels))


def test_shift_prior_to_a_rare_threat():
    # Calibrated at 40% fakes, 0.9 is strong evidence; at 1 in 1,000 it's far less likely fake.
    shifted = float(shift_prior([0.9], from_share=0.4, to_share=0.001)[0])
    expected_odds = 9 * (0.001 / 0.999) / (0.4 / 0.6)
    assert shifted == pytest.approx(expected_odds / (1 + expected_odds))
    assert shifted < 0.02


def test_shift_prior_is_identity_for_the_same_share():
    p = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(shift_prior(p, 0.3, 0.3), p)
