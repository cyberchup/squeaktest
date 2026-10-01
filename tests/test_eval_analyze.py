"""Tests for the offline analysis: applying the product's policy and choosing thresholds."""

from __future__ import annotations

import numpy as np
import pytest

from eval.analyze import Policy, band_table, choose_thresholds, clip_score, score_records


def record(label: int, margins: list[float | None], fractions=None, active_s=3.0, error=None):
    fractions = fractions or [1.0] * len(margins)
    windows = [
        [i * 2.0, i * 2.0 + 4.0, f, m]
        for i, (f, m) in enumerate(zip(fractions, margins, strict=True))
    ]
    return {"id": f"{label}-{margins}", "label": label, "speaker": "s", "duration_s": 5.0,
            "active_s": active_s, "windows": windows, "error": error}  # fmt: skip


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def test_clip_score_matches_the_product_rules():
    r = record(1, [0.0, 2.0, 4.0, 6.0])
    # top-3 mean of window probabilities
    expected = np.mean([sigmoid(2.0), sigmoid(4.0), sigmoid(6.0)])
    assert clip_score(r, Policy()) == pytest.approx(expected)
    assert clip_score(r, Policy(top_k=1)) == pytest.approx(sigmoid(6.0))


def test_clips_with_too_little_sound_are_not_assessed():
    assert clip_score(record(0, [1.0], active_s=0.6), Policy()) is None
    assert clip_score(record(0, [1.0], active_s=0.6), Policy(min_speech_s=0.5)) is not None


def test_mostly_silent_windows_are_ignored():
    r = record(0, [9.0, -3.0], fractions=[0.2, 0.9])
    assert clip_score(r, Policy()) == pytest.approx(sigmoid(-3.0))
    assert clip_score(record(0, [9.0], fractions=[0.2]), Policy()) is None


def test_errored_clips_are_not_assessed():
    assert clip_score(record(0, [], error="UnsupportedFormatError: x"), Policy()) is None


def test_coverage_is_counted_per_class():
    records = [record(0, [1.0]), record(0, [1.0], active_s=0.1), record(1, [5.0]), record(1, [5.0])]
    scored = score_records(records, Policy())
    assert scored.labels.tolist() == [0, 1, 1]
    assert scored.not_assessed == {0: 1, 1: 0}
    assert scored.total == {0: 2, 1: 2}


def test_thresholds_respect_the_dev_rules():
    rng = np.random.default_rng(0)
    labels = np.concatenate([np.zeros(5000), np.ones(5000)]).astype(int)
    probs = np.clip(np.concatenate([rng.beta(2, 8, 5000), rng.beta(8, 2, 5000)]), 0, 1)
    t = choose_thresholds(labels, probs)
    flagged_genuine = (probs[labels == 0] >= t["likely_synthetic_at"]).mean()
    fakes_called_genuine = (probs[labels == 1] < t["likely_genuine_below"]).mean()
    assert flagged_genuine <= 0.01
    assert fakes_called_genuine <= 0.05
    assert t["likely_genuine_below"] <= t["likely_synthetic_at"]


def test_band_table_rows_sum_to_one():
    labels = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.6, 0.6, 0.95])
    table = band_table(labels, probs, {"likely_synthetic_at": 0.9, "likely_genuine_below": 0.3})
    assert table["genuine"] == {"likely_synthetic": 0.0, "uncertain": 0.5, "likely_genuine": 0.5}
    assert table["synthetic"]["likely_synthetic"] == 0.5
    assert all(sum(row.values()) == pytest.approx(1.0) for row in table.values())
