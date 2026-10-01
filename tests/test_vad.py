"""Tests for voice activity detection (step 3). The spec is in src/squeaktest/vad.py."""

from __future__ import annotations

import numpy as np
import pytest

from squeaktest.vad import (
    SILENCE_DB,
    active_frames,
    frame_levels_db,
    speech_bounds,
    speech_fraction,
)

SR = 16_000


def tone(seconds: float, amplitude: float = 0.5, hz: float = 440.0, sr: int = SR) -> np.ndarray:
    t = np.arange(round(seconds * sr)) / sr
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def silence(seconds: float, sr: int = SR) -> np.ndarray:
    return np.zeros(round(seconds * sr), dtype=np.float32)


# --- Framing -----------------------------------------------------------------------------


def test_one_second_at_16k_gives_98_frames():
    # 400-sample frames every 160 samples: 1 + (16000 - 400) // 160 = 98
    assert frame_levels_db(tone(1.0), SR).shape == (98,)


def test_frame_lengths_follow_the_sample_rate():
    # At 8 kHz a frame is 200 samples and a hop is 80: 1 + (8000 - 200) // 80 = 98
    assert frame_levels_db(tone(1.0, sr=8000), 8000).shape == (98,)


def test_custom_frame_and_hop():
    # 50 ms frames (800 samples) every 25 ms (400): 1 + (16000 - 800) // 400 = 39
    assert frame_levels_db(tone(1.0), SR, frame_ms=50, hop_ms=25).shape == (39,)


def test_partial_last_frame_is_dropped():
    # 400 + 160 + 100 samples: two whole frames; the last 100 samples don't make a third
    assert frame_levels_db(np.ones(660, dtype=np.float32), SR).shape == (2,)


@pytest.mark.parametrize("length", [0, 1, 399])
def test_shorter_than_one_frame_gives_no_frames(length: int):
    assert frame_levels_db(np.ones(length, dtype=np.float32), SR).shape == (0,)


# --- Levels ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("signal", "expected_db"),
    [
        (np.ones(SR, dtype=np.float32), 0.0),
        (np.full(SR, 0.5, dtype=np.float32), -6.02),
        (tone(1.0, amplitude=1.0), -3.01),
        (tone(1.0, amplitude=0.1), -23.01),
    ],
    ids=["full-scale", "half-scale", "sine-1.0", "sine-0.1"],
)
def test_levels_in_dbfs(signal: np.ndarray, expected_db: float):
    levels = frame_levels_db(signal, SR)
    np.testing.assert_allclose(levels, expected_db, atol=0.05)


def test_digital_silence_is_clamped_not_minus_infinity():
    levels = frame_levels_db(silence(1.0), SR)
    np.testing.assert_allclose(levels, SILENCE_DB, atol=1e-6)


def test_levels_never_go_below_the_floor():
    tiny = np.full(SR, 1e-9, dtype=np.float32)  # about -180 dBFS
    np.testing.assert_allclose(frame_levels_db(tiny, SR), SILENCE_DB, atol=1e-6)


def test_rejects_multichannel_input():
    stereo = np.stack([tone(1.0), tone(1.0)])
    with pytest.raises(ValueError):
        frame_levels_db(stereo, SR)


def test_does_not_modify_the_input():
    samples = tone(1.0)
    before = samples.copy()
    frame_levels_db(samples, SR)
    np.testing.assert_array_equal(samples, before)


# --- Activity ----------------------------------------------------------------------------


def test_a_clear_tone_is_active():
    assert active_frames(tone(1.0, amplitude=0.1), SR).all()


def test_silence_is_inactive():
    assert not active_frames(silence(1.0), SR).any()


def test_quiet_noise_floor_is_inactive():
    # White noise around -60 dBFS: hiss on a quiet line, below the default -45 dB threshold
    rng = np.random.default_rng(seed=0)
    noise = rng.normal(0.0, 0.001, SR).astype(np.float32)
    assert not active_frames(noise, SR).any()


def test_threshold_is_adjustable():
    quiet = tone(1.0, amplitude=0.001)  # about -63 dBFS
    assert not active_frames(quiet, SR).any()
    assert active_frames(quiet, SR, threshold_db=-70).all()


def test_threshold_is_strictly_greater_than():
    signal = np.full(SR, 0.5, dtype=np.float32)  # -6.02 dBFS
    level = float(frame_levels_db(signal, SR)[0])
    assert not active_frames(signal, SR, threshold_db=level).any()


# --- Trimming ----------------------------------------------------------------------------


def test_speech_bounds_trim_leading_and_trailing_silence():
    # 0.5 s silence, 1 s tone, 0.5 s silence: activity spans samples 8000 to 24000.
    # Frames straddling an edge can count as active, so allow one frame of slack.
    clip = np.concatenate([silence(0.5), tone(1.0), silence(0.5)])
    bounds = speech_bounds(clip, SR)
    assert bounds is not None
    start, end = bounds
    assert abs(start - 8000) <= 400
    assert abs(end - 24000) <= 400


def test_speech_bounds_are_frame_aligned():
    clip = np.concatenate([silence(0.5), tone(1.0), silence(0.5)])
    bounds = speech_bounds(clip, SR)
    assert bounds is not None
    start, end = bounds
    assert start % 160 == 0  # starts where a frame starts
    assert (end - 400) % 160 == 0  # ends where a frame ends


def test_speech_bounds_for_all_activity_cover_every_whole_frame():
    # 1 s of tone: 98 frames, the last ending at 97 * 160 + 400 = 15920
    assert speech_bounds(tone(1.0), SR) == (0, 15_920)


def test_speech_bounds_none_when_silent():
    assert speech_bounds(silence(1.0), SR) is None


def test_speech_bounds_none_for_empty_input():
    assert speech_bounds(np.zeros(0, dtype=np.float32), SR) is None


# --- Fraction ----------------------------------------------------------------------------


def test_fraction_of_a_half_silent_clip():
    clip = np.concatenate([silence(0.5), tone(1.0), silence(0.5)])
    assert speech_fraction(clip, SR) == pytest.approx(0.5, abs=0.05)


def test_fraction_extremes():
    assert speech_fraction(tone(1.0), SR) == 1.0
    assert speech_fraction(silence(1.0), SR) == 0.0


def test_fraction_of_empty_input_is_zero():
    assert speech_fraction(np.zeros(0, dtype=np.float32), SR) == 0.0


def test_fraction_is_a_plain_float():
    assert type(speech_fraction(tone(1.0), SR)) is float
