"""Synthetic non-speech audio for measuring K1: tones, telephone signals, noise and "music".

Everything here is generated, so nothing is downloaded and no recording is ever committed.
Each signal comes in two variants with different parameters (frequencies, cadences, seeds,
keys): "dev" for choosing the tone gate's settings and "test" for reporting, so the gate is
never tuned on the signals it is scored on.

The telephone signals follow published call-progress tones (for example ITU-T E.180 and the
North American precise tone plan). The "music" is a few synthesized instruments playing chords
and melodies. It is a stand-in for hold music, not a substitute for real recordings: it says
nothing about how the model or the gate behave on real music.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

SR = 16_000
LEVEL_RMS = 0.1  # about -20 dBFS: a normal speech level, well above the -45 dBFS VAD threshold
DTMF_LOW = {"1": 697, "2": 697, "3": 697, "4": 770, "5": 770, "6": 770}
DTMF_LOW |= {"7": 852, "8": 852, "9": 852, "*": 941, "0": 941, "#": 941}
DTMF_HIGH = {"1": 1209, "4": 1209, "7": 1209, "*": 1209, "2": 1336, "5": 1336, "8": 1336}
DTMF_HIGH |= {"0": 1336, "3": 1477, "6": 1477, "9": 1477, "#": 1477}


@dataclass(frozen=True)
class Signal:
    name: str
    family: str  # what kind of sound, for grouping results
    make: Callable[[float], np.ndarray]  # duration in seconds -> float32 samples at SR


def _t(seconds: float) -> np.ndarray:
    return np.arange(round(seconds * SR)) / SR


def _level(x: np.ndarray, rms: float = LEVEL_RMS) -> np.ndarray:
    current = np.sqrt(np.mean(np.square(x)))
    return (x * (rms / current)).astype(np.float32) if current > 0 else x.astype(np.float32)


def _tones(freqs: tuple[float, ...], seconds: float) -> np.ndarray:
    t = _t(seconds)
    return sum(np.sin(2 * np.pi * f * t) for f in freqs)


def _cadence(x: np.ndarray, pattern: tuple[float, ...]) -> np.ndarray:
    """Switch `x` on and off: pattern is (on, off, on, off, ...) in seconds, repeated."""
    gate = np.zeros(x.size, dtype=bool)
    period = sum(pattern)
    t = (np.arange(x.size) / SR) % period
    edge = 0.0
    for i, length in enumerate(pattern):
        if i % 2 == 0:
            gate |= (t >= edge) & (t < edge + length)
        edge += length
    return x * gate


def _steady(freqs: tuple[float, ...], pattern: tuple[float, ...] | None = None):
    def make(seconds: float) -> np.ndarray:
        x = _tones(freqs, seconds)
        return _level(_cadence(x, pattern) if pattern else x)

    return make


def _dtmf(digits: str, on_s: float, off_s: float, pause_s: float):
    def make(seconds: float) -> np.ndarray:
        out = np.zeros(round(seconds * SR))
        pos = 0
        while pos < out.size:
            for digit in digits:
                tone = _tones((DTMF_LOW[digit], DTMF_HIGH[digit]), on_s)
                end = min(pos + tone.size, out.size)
                out[pos:end] = tone[: end - pos]
                pos += round((on_s + off_s) * SR)
                if pos >= out.size:
                    break
            pos += round(pause_s * SR)
        return _level(out)

    return make


def _sit(freqs: tuple[float, float, float]):
    # Special information tone ("the number you have dialed..."): three rising tones, then a gap.
    def make(seconds: float) -> np.ndarray:
        segments = [_tones((f,), d) for f, d in zip(freqs, (0.274, 0.274, 0.380), strict=True)]
        cycle = np.concatenate([*segments, np.zeros(round(2.0 * SR))])
        return _level(np.resize(cycle, round(seconds * SR)))

    return make


def _sweep(f0: float, f1: float, period_s: float):
    def make(seconds: float) -> np.ndarray:
        t = _t(seconds) % period_s
        k = (f1 - f0) / period_s
        return _level(np.sin(2 * np.pi * (f0 * t + 0.5 * k * t**2)))

    return make


def _noise(color: str, seed: int):
    def make(seconds: float) -> np.ndarray:
        white = np.random.default_rng(seed).standard_normal(round(seconds * SR))
        if color == "white":
            return _level(white)
        # Pink noise: shape white noise's spectrum by 1/sqrt(f), so power falls 3 dB per octave.
        spectrum = np.fft.rfft(white)
        f = np.fft.rfftfreq(white.size, 1 / SR)
        spectrum[1:] /= np.sqrt(f[1:])
        spectrum[0] = 0
        return _level(np.fft.irfft(spectrum, white.size))

    return make


def _hum(mains_hz: float):
    def make(seconds: float) -> np.ndarray:
        t = _t(seconds)
        return _level(sum(np.sin(2 * np.pi * mains_hz * h * t) / h for h in range(1, 6)))

    return make


def _note(freq: float, seconds: float, partials: int, decay: float) -> np.ndarray:
    t = _t(seconds)
    tone = sum(np.sin(2 * np.pi * freq * h * t) / h**1.5 for h in range(1, partials + 1))
    attack = np.minimum(t / 0.01, 1.0)
    return tone * attack * np.exp(-decay * t)


def _midi(note: int) -> float:
    return 440.0 * 2 ** ((note - 69) / 12)


def _add(out: np.ndarray, at: int, sound: np.ndarray) -> None:
    """Mix `sound` into `out` starting at sample `at`, cutting off whatever runs past the end."""
    end = min(at + sound.size, out.size)
    if end > at:
        out[at:end] += sound[: end - at]


def _music(progression: list[tuple[int, int, int]], beat_s: float, partials: int, decay: float):
    """Chords held for four beats, with a melody of arpeggiated chord tones on every beat."""

    def make(seconds: float) -> np.ndarray:
        out = np.zeros(round(seconds * SR))
        bar_s = 4 * beat_s
        for bar in range(int(np.ceil(seconds / bar_s))):
            chord = progression[bar % len(progression)]
            start = round(bar * bar_s * SR)
            for note in chord:
                _add(out, start, 0.3 * _note(_midi(note), bar_s, partials, decay / 4))
            for beat in range(4):
                lead = _note(_midi(chord[beat % 3] + 12), beat_s, partials, decay)
                _add(out, start + round(beat * beat_s * SR), lead)
        return _level(out)

    return make


# C major I-V-vi-IV, the same in D major, and A minor i-VI-III-VII, as MIDI note numbers.
_POP = [(60, 64, 67), (55, 59, 62), (57, 60, 64), (53, 57, 60)]
_POP_D = [(n + 2, m + 2, k + 2) for n, m, k in _POP]
_MINOR = [(57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)]

SIGNALS: dict[str, list[Signal]] = {
    "dev": [
        Signal("tone_1000hz", "tone", _steady((1000,))),
        Signal("tone_440hz", "tone", _steady((440,))),
        Signal("beeps_1000hz", "beeps", _steady((1000,), (0.25, 0.75))),
        Signal("dial_us", "call_progress", _steady((350, 440))),
        Signal("ringback_us", "call_progress", _steady((440, 480), (2.0, 4.0))),
        Signal("busy_us", "call_progress", _steady((480, 620), (0.5, 0.5))),
        Signal("reorder_us", "call_progress", _steady((480, 620), (0.25, 0.25))),
        Signal("sit_low", "call_progress", _sit((913.8, 1370.6, 1776.7))),
        Signal("dtmf_fast", "dtmf", _dtmf("8005551234", 0.10, 0.10, 1.0)),
        Signal("dtmf_slow", "dtmf", _dtmf("2468#", 0.25, 0.30, 1.5)),
        Signal("fax_cng", "fax", _steady((1100,), (0.5, 3.0))),
        Signal("sweep", "sweep", _sweep(300, 3000, 2.0)),
        Signal("noise_white", "noise", _noise("white", 1)),
        Signal("noise_pink", "noise", _noise("pink", 2)),
        Signal("hum_60hz", "hum", _hum(60)),
        Signal("music_pop_piano", "music", _music(_POP, 0.5, 8, 3.0)),
        Signal("music_minor_organ", "music", _music(_MINOR, 0.75, 4, 0.2)),
    ],
    "test": [
        Signal("tone_1400hz", "tone", _steady((1400,))),
        Signal("tone_600hz", "tone", _steady((600,))),
        Signal("beeps_850hz", "beeps", _steady((850,), (0.4, 0.6))),
        Signal("dial_uk", "call_progress", _steady((350, 450))),
        Signal("ringback_uk", "call_progress", _steady((400, 450), (0.4, 0.2, 0.4, 2.0))),
        Signal("busy_eu", "call_progress", _steady((425,), (0.5, 0.5))),
        Signal("reorder_uk", "call_progress", _steady((400,), (0.4, 0.35, 0.225, 0.525))),
        Signal("sit_high", "call_progress", _sit((985.2, 1428.5, 1776.7))),
        Signal("dtmf_fast", "dtmf", _dtmf("2125550199", 0.08, 0.08, 1.2)),
        Signal("dtmf_slow", "dtmf", _dtmf("13579*", 0.20, 0.35, 1.0)),
        Signal("fax_ced", "fax", _steady((2100,))),
        Signal("sweep", "sweep", _sweep(500, 2500, 3.0)),
        Signal("noise_white", "noise", _noise("white", 11)),
        Signal("noise_pink", "noise", _noise("pink", 12)),
        Signal("hum_50hz", "hum", _hum(50)),
        Signal("music_pop_bell", "music", _music(_POP_D, 0.4, 6, 5.0)),
        Signal("music_minor_strings", "music", _music(_MINOR, 1.0, 10, 0.1)),
    ],
}


def signal(split: str, prefix: str) -> Signal:
    """The first signal in a split whose name starts with `prefix` (used by the mixes below)."""
    return next(s for s in SIGNALS[split] if s.name.startswith(prefix))


@dataclass(frozen=True)
class Mix:
    name: str
    describe: str
    make: Callable[[np.ndarray, str], np.ndarray]  # (speech samples, split) -> mixed samples


def _before(prefix: str, seconds: float, gap_s: float = 0.5):
    def make(speech: np.ndarray, split: str) -> np.ndarray:
        sound = signal(split, prefix).make(seconds)
        return np.concatenate([sound, np.zeros(round(gap_s * SR), np.float32), speech])

    return make


def _after(prefix: str, seconds: float, gap_s: float = 0.5):
    def make(speech: np.ndarray, split: str) -> np.ndarray:
        sound = signal(split, prefix).make(seconds)
        return np.concatenate([speech, np.zeros(round(gap_s * SR), np.float32), sound])

    return make


def _under(prefix: str, below_db: float):
    """Add a sound under the whole clip, `below_db` quieter than the speech's active level."""

    def make(speech: np.ndarray, split: str) -> np.ndarray:
        sound = signal(split, prefix).make(speech.size / SR)[: speech.size]
        loud = np.abs(speech) > 10 ** (-45 / 20)
        level = np.sqrt(np.mean(np.square(speech[loud]))) if loud.any() else LEVEL_RMS
        return (speech + sound * (level / LEVEL_RMS) * 10 ** (-below_db / 20)).astype(np.float32)

    return make


def _beep_before(speech: np.ndarray, split: str) -> np.ndarray:
    # A voicemail system's single beep, then the caller starts talking.
    beep = signal(split, "tone").make(0.5)
    return np.concatenate([beep, np.zeros(round(0.3 * SR), np.float32), speech])


MIXES: list[Mix] = [
    Mix("original", "the clip as it is", lambda speech, split: speech),
    Mix("beep_before", "0.5 s voicemail beep, then speech", _beep_before),
    Mix("ringback_before", "8 s of ringing before speech", _before("ringback", 8.0)),
    Mix("dtmf_before", "4 s of key presses before speech", _before("dtmf_fast", 4.0)),
    Mix("busy_after", "speech, then 6 s of busy tone", _after("busy", 6.0)),
    Mix("music_before", "20 s of synthetic hold music, then speech", _before("music_pop", 20.0)),
    Mix("tone_under", "a steady tone 20 dB under the speech", _under("tone", 20.0)),
    Mix("hum_under", "mains hum 20 dB under the speech", _under("hum", 20.0)),
    Mix("music_under", "synthetic music 15 dB under the speech", _under("music", 15.0)),
]
