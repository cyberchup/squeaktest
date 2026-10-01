"""Deterministic synthetic signals for model parity tests.

No random number generator: NumPy doesn't promise identical random streams across versions,
and these signals must come out the same on Windows and in the WSL reference environment.
Noise comes from a fixed linear congruential generator instead. No real voices.
"""

from __future__ import annotations

import numpy as np

SR = 16_000


def _t(seconds: float) -> np.ndarray:
    return np.arange(round(seconds * SR), dtype=np.float64) / SR


def _lcg_noise(n: int, seed: int = 12345) -> np.ndarray:
    # Numerical Recipes LCG: exact integer arithmetic, identical on every platform.
    values = np.empty(n, dtype=np.float64)
    x = seed
    for i in range(n):
        x = (1664525 * x + 1013904223) % 2**32
        values[i] = x / 2**31 - 1.0
    return values


def parity_signals() -> dict[str, np.ndarray]:
    t4 = _t(4.0)
    t45 = _t(4.5)
    buzz = sum(np.sin(2 * np.pi * 150 * h * t45) / h for h in range(1, 20))
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 4 * t45))  # syllable-rate loudness changes
    signals = {
        "tone_440hz_4s": 0.5 * np.sin(2 * np.pi * 440 * t4),
        "tone_880hz_1s": 0.5 * np.sin(2 * np.pi * 880 * _t(1.0)),
        "chirp_100_4000hz_4s": 0.5 * np.sin(2 * np.pi * (100 * t4 + (3900 / 8) * t4**2)),
        "buzz_150hz_am_4_5s": 0.2 * buzz * envelope,
        "lcg_noise_2s": 0.3 * _lcg_noise(round(2.0 * SR)),
    }
    return {name: s.astype(np.float32) for name, s in signals.items()}
