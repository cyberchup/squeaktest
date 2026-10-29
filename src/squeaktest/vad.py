"""Voice activity detection (VAD): find the parts of a clip that contain sound worth scoring.

Why this exists: detectors trained on ASVspoof learned to use silence as a shortcut (real
clips there have longer silences than fake ones), and Deepfake-Eval-2024 found that silent
stretches cut audio detectors' accuracy sharply. So squeaktest trims leading and trailing
silence, and skips windows that are mostly silent instead of scoring them.

This is energy-based VAD: a frame counts as "active" if it is loud enough. On its own it can't
tell speech from music or a door slam, only sound from silence. That is enough to drop silence;
a model-based VAD could replace it later.

One kind of non-speech is found as well: steady tones such as beeps and dial, ringing or busy
tones (tone_share and tone_mask, D31). The model scores them as if they were voices, so frames
in a tone don't count as active either. Music, noise and other sounds still do (K1).

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

# Steady tones (K1, D31): beeps, dial, ring and busy tones, fax tones. The model only learned
# about speech and gives these confident "synthetic" scores, so they are found and left out.
TONE_FRAME_MS = 256.0  # long frames: a steady tone stays one sharp line, speech's pitch smears
TONE_MIN_HZ = 200.0  # below this sits mains hum, which old genuine recordings are full of
TONE_PEAK_BINS = 10  # room for two tones (dual-tone signals) of about 5 bins each
TONE_SHARE = 0.99  # share of the energy in those bins that marks a frame as a tone


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
    exclude: np.ndarray | None = None,
) -> np.ndarray:
    """Return a boolean array: True for each frame whose level is above `threshold_db`.

    Uses the default frame and hop lengths. "Above" means strictly greater than. `exclude`,
    if given, holds one boolean per sample (from tone_mask): a frame whose centre sample is
    excluded is never active, however loud it is.
    """
    active = frame_levels_db(samples, sample_rate) > threshold_db
    if exclude is not None and active.size:
        exclude = np.asarray(exclude, dtype=bool)
        if exclude.shape != np.shape(samples):
            raise ValueError("exclude must hold one value per sample")
        frame_length, hop_length = _frame_and_hop_lengths(sample_rate, FRAME_MS, HOP_MS)
        active &= ~exclude[np.arange(active.size) * hop_length + frame_length // 2]
    return active


def speech_bounds(
    samples: np.ndarray,
    sample_rate: int,
    threshold_db: float = DEFAULT_THRESHOLD_DB,
    exclude: np.ndarray | None = None,
) -> tuple[int, int] | None:
    """Return (start, end) sample indices spanning all activity, or None if there is none.

    `start` is where the first active frame begins and `end` is where the last active frame
    ends, so `samples[start:end]` is the clip with leading and trailing silence trimmed.
    """
    # flatnonzero gives the indices of the True entries, i.e. which frames are active.
    active = np.flatnonzero(active_frames(samples, sample_rate, threshold_db, exclude))
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
    exclude: np.ndarray | None = None,
) -> float:
    """Return the fraction of frames that are active, from 0.0 to 1.0.

    Step 4 uses this to skip windows that are mostly silent. Return 0.0 when there are no
    frames at all (for example, empty input).
    """
    active = active_frames(samples, sample_rate, threshold_db, exclude)
    if active.size == 0:
        return 0.0
    # The mean of a boolean array is the fraction of True values.
    return float(active.mean())


def tone_share(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: float = TONE_FRAME_MS,
    peak_bins: int = TONE_PEAK_BINS,
    min_hz: float = TONE_MIN_HZ,
) -> np.ndarray:
    """Return, for each frame, how much of the sound around it is one or two steady tones.

    One value per frame of frame_levels_db. The long (`frame_ms`) Hann-windowed frame centred
    on that frame is Fourier-transformed, and the value is the share of its energy above
    `min_hz` that sits in its `peak_bins` strongest frequency bins. A steady tone puts nearly
    all of it in one or two sharp lines (close to 1.0). Speech spreads it over many harmonics
    whose frequencies move with the voice's pitch. Frames with (almost) nothing above `min_hz`,
    such as silence or a low hum with no harmonics, get 0.0: there is nothing there to judge.
    """
    samples = np.asarray(samples)
    frame_length, hop_length = _frame_and_hop_lengths(sample_rate, FRAME_MS, HOP_MS)
    n_frames = 0 if samples.size < frame_length else 1 + (samples.size - frame_length) // hop_length
    n_fft = round(sample_rate * frame_ms / 1000)
    lowest_bin = int(np.ceil(min_hz * n_fft / sample_rate))
    window = np.hanning(n_fft)
    # Long frames centred where the short frames are centred, with zeros beyond the clip.
    padded = np.pad(samples.astype(np.float64), n_fft // 2)
    starts = np.arange(n_frames) * hop_length + frame_length // 2
    shares = np.zeros(n_frames)
    for first in range(0, n_frames, _TONE_BLOCK):  # in blocks, to keep memory small
        rows = starts[first : first + _TONE_BLOCK, None] + np.arange(n_fft)[None, :]
        spectrum = np.square(np.abs(np.fft.rfft(padded[rows] * window, axis=1)))
        power = spectrum[:, lowest_bin:]
        total = power.sum(axis=1)
        peaks = np.partition(power, -peak_bins, axis=1)[:, -peak_bins:].sum(axis=1)
        # Below 1/10,000 of the frame's energy (-40 dB), what's above min_hz is only the
        # window's leakage from lower frequencies, whose shape says nothing about the sound.
        judged = total > np.maximum(1e-4 * spectrum.sum(axis=1), 1e-12)
        block = np.divide(peaks, total, out=np.zeros_like(total), where=judged)
        shares[first : first + _TONE_BLOCK] = block
    return shares


def tone_mask(
    samples: np.ndarray,
    sample_rate: int,
    threshold: float = TONE_SHARE,
    frame_ms: float = TONE_FRAME_MS,
    **share_options: float,
) -> np.ndarray:
    """Return one boolean per sample: True where the sample is part of a steady tone.

    A frame is a tone when its tone_share reaches `threshold`. Each run of tone frames is then
    widened by half a long frame on both sides: a long frame that straddles a tone's start or
    end also holds silence or speech, so it reads as less pure than the tone really is.
    """
    samples = np.asarray(samples)
    tones = tone_share(samples, sample_rate, frame_ms, **share_options) >= threshold
    frame_length, hop_length = _frame_and_hop_lengths(sample_rate, FRAME_MS, HOP_MS)
    reach = round(frame_ms / 2 / HOP_MS)  # half a long frame, counted in short frames
    if reach and tones.any():
        # Frame i is a tone if any frame within `reach` of it is: count tone frames in
        # [i - reach, i + reach] with a running total.
        counts = np.concatenate([[0], np.cumsum(tones)])
        index = np.arange(tones.size)
        lo = np.clip(index - reach, 0, tones.size)
        hi = np.clip(index + reach + 1, 0, tones.size)
        tones = counts[hi] - counts[lo] > 0
    # Each frame owns the hop of samples around its centre, so a frame's centre sample
    # (what active_frames looks up) carries that frame's own verdict.
    mask = np.zeros(samples.size, dtype=bool)
    first = frame_length // 2 - hop_length // 2
    owned = np.repeat(tones, hop_length)[: max(samples.size - first, 0)]
    mask[first : first + owned.size] = owned
    return mask


_TONE_BLOCK = 500  # long frames transformed at once: 500 x 4096 float64 is 16 MB


def _frame_and_hop_lengths(sample_rate: int, frame_ms: float, hop_ms: float) -> tuple[int, int]:
    frame_length = round(sample_rate * frame_ms / 1000)
    hop_length = round(sample_rate * hop_ms / 1000)
    if frame_length < 1 or hop_length < 1:
        raise ValueError("frame and hop must each be at least one sample long")
    return frame_length, hop_length
