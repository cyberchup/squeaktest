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

Your task: implement the four functions below so `uv run pytest tests/test_vad.py` passes.
Hint: numpy can do this without a Python loop. Look at
`numpy.lib.stride_tricks.sliding_window_view`, and read its docs on the `step` you need.
"""

from __future__ import annotations

import numpy as np

FRAME_MS = 25.0
HOP_MS = 10.0
DEFAULT_THRESHOLD_DB = -45.0
SILENCE_DB = -100.0


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
    raise NotImplementedError


def active_frames(
    samples: np.ndarray,
    sample_rate: int,
    threshold_db: float = DEFAULT_THRESHOLD_DB,
) -> np.ndarray:
    """Return a boolean array: True for each frame whose level is above `threshold_db`.

    Uses the default frame and hop lengths. "Above" means strictly greater than.
    """
    raise NotImplementedError


def speech_bounds(
    samples: np.ndarray,
    sample_rate: int,
    threshold_db: float = DEFAULT_THRESHOLD_DB,
) -> tuple[int, int] | None:
    """Return (start, end) sample indices spanning all activity, or None if there is none.

    `start` is where the first active frame begins and `end` is where the last active frame
    ends, so `samples[start:end]` is the clip with leading and trailing silence trimmed.
    """
    raise NotImplementedError


def speech_fraction(
    samples: np.ndarray,
    sample_rate: int,
    threshold_db: float = DEFAULT_THRESHOLD_DB,
) -> float:
    """Return the fraction of frames that are active, from 0.0 to 1.0.

    Step 4 uses this to skip windows that are mostly silent. Return 0.0 when there are no
    frames at all (for example, empty input).
    """
    raise NotImplementedError
