"""Voice activity detection (VAD): find the parts of a clip that contain sound worth scoring.

Why this exists: detectors trained on ASVspoof learned to use silence as a shortcut (real
clips there have longer silences than fake ones), and Deepfake-Eval-2024 found that silent
stretches cut audio detectors' accuracy sharply. So squeaktest trims leading and trailing
silence, and skips windows that are mostly silent instead of scoring them.

This is energy-based VAD: a frame counts as "active" if it is loud enough. It can't tell
speech from music or a door slam, only sound from silence. That is enough to drop silence;
a model-based VAD could replace it later.

Concepts
--------
- samples: the audio as a 1-D float array, roughly in [-1, 1], at `sample_rate` samples per
  second (16,000 after loading).
- frame: a short slice of samples. 25 ms is standard for speech because speech is roughly
  stable over 20-30 ms. At 16 kHz, 25 ms is 400 samples.
- hop: how far the next frame starts after the previous one. A 10 ms hop with 25 ms frames
  means frames overlap. At 16 kHz, 10 ms is 160 samples.
- RMS (root mean square): sqrt(mean(x**2)), the "average loudness" of a frame.
- dBFS (decibels relative to full scale): 20 * log10(rms). 0 dBFS is the loudest a signal can
  be, every halving of amplitude is about -6 dB, and quieter is more negative. Decibels are
  logarithmic because hearing is.

Reference values that the tests check:
- constant signal of 1.0 -> RMS 1.0 -> 0 dBFS
- constant signal of 0.5 -> -6.02 dBFS
- sine wave with amplitude 1.0 -> RMS 1/sqrt(2) -> -3.01 dBFS
- sine wave with amplitude 0.1 -> -23.01 dBFS
- digital silence (all zeros) -> log10(0) is -infinity, so clamp to SILENCE_DB (-100 dBFS)
"""

from __future__ import annotations

import numpy as np

FRAME_MS = 25.0
HOP_MS = 10.0
DEFAULT_THRESHOLD_DB = -45.0
SILENCE_DB = -100.0

# The RMS value that corresponds to SILENCE_DB: 10 ** (-100 / 20) = 0.00001.
_SILENCE_RMS = 10 ** (SILENCE_DB / 20)


def frame_levels_db(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: float = FRAME_MS,
    hop_ms: float = HOP_MS,
) -> np.ndarray:
    """Return the RMS level of each frame in dBFS.

    - Frame length is round(sample_rate * frame_ms / 1000) samples; hop length likewise.
    - Frames start at 0, hop, 2*hop, ... and only whole frames count: a partial frame at the
      end is dropped. So the number of frames is 0 if there are fewer samples than one frame,
      otherwise 1 + (len(samples) - frame_length) // hop_length.
    - Levels are clamped so they are never below SILENCE_DB.
    - Raise ValueError if `samples` is not 1-D.
    - Don't modify `samples`.
    """
    samples = np.asarray(samples)
    if samples.ndim != 1:
        raise ValueError(f"samples must be 1-D (mono), got shape {samples.shape}")
    frame_length, hop_length = _frame_and_hop_lengths(sample_rate, frame_ms, hop_ms)
    if samples.size < frame_length:
        return np.zeros(0)

    # sliding_window_view gives every run of frame_length consecutive samples as a row,
    # without copying (it's a read-only view, so `samples` can't be modified). Taking every
    # hop_length-th row keeps the frames that start at 0, hop, 2*hop, ...
    frames = np.lib.stride_tricks.sliding_window_view(samples, frame_length)[::hop_length]

    # Square in float64: float32 loses precision when squaring very small values.
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))

    # Clamp before taking the log, so silence (RMS 0) becomes SILENCE_DB instead of -infinity.
    return 20 * np.log10(np.maximum(rms, _SILENCE_RMS))


def active_frames(
    samples: np.ndarray,
    sample_rate: int,
    threshold_db: float = DEFAULT_THRESHOLD_DB,
) -> np.ndarray:
    """Return a boolean array: True for each frame whose level is above `threshold_db`.

    Uses the default frame and hop lengths. "Above" means strictly greater than.
    """
    return frame_levels_db(samples, sample_rate) > threshold_db


def speech_bounds(
    samples: np.ndarray,
    sample_rate: int,
    threshold_db: float = DEFAULT_THRESHOLD_DB,
) -> tuple[int, int] | None:
    """Return (start, end) sample indices spanning all activity, or None if there is none.

    `start` is where the first active frame begins and `end` is where the last active frame
    ends, so `samples[start:end]` is the clip with leading and trailing silence trimmed.
    """
    # flatnonzero gives the indices of the True entries, i.e. which frames are active.
    active = np.flatnonzero(active_frames(samples, sample_rate, threshold_db))
    if active.size == 0:
        return None
    frame_length, hop_length = _frame_and_hop_lengths(sample_rate, FRAME_MS, HOP_MS)
    # Frame i covers samples [i * hop, i * hop + frame_length).
    start = int(active[0]) * hop_length
    end = int(active[-1]) * hop_length + frame_length
    return start, end


def speech_fraction(
    samples: np.ndarray,
    sample_rate: int,
    threshold_db: float = DEFAULT_THRESHOLD_DB,
) -> float:
    """Return the fraction of frames that are active, from 0.0 to 1.0.

    Step 4 uses this to skip windows that are mostly silent. Return 0.0 when there are no
    frames at all (for example, empty input).
    """
    active = active_frames(samples, sample_rate, threshold_db)
    if active.size == 0:
        return 0.0
    # The mean of a boolean array is the fraction of True values.
    return float(active.mean())


def _frame_and_hop_lengths(sample_rate: int, frame_ms: float, hop_ms: float) -> tuple[int, int]:
    frame_length = round(sample_rate * frame_ms / 1000)
    hop_length = round(sample_rate * hop_ms / 1000)
    if frame_length < 1 or hop_length < 1:
        raise ValueError("frame and hop must each be at least one sample long")
    return frame_length, hop_length
