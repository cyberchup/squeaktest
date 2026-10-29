"""Turn an analysis into a report: a JSON-ready dict, and a human-readable text version.

The report never includes the input's filename or path, so it is safe to log or share.
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Any

from squeaktest import __version__
from squeaktest.inference import ClipResult, WindowingConfig, band_for

if TYPE_CHECKING:
    from squeaktest.audio import Audio
    from squeaktest.model import ModelSpec

DISCLAIMER = "This score is an estimate. It must not be the sole basis for any decision."
SCORE_NOTE = (
    "The score is the model's raw output, not a probability. The bands are thresholds chosen "
    "on development data, kept apart from the test data (docs/decisions.md D16 and D18)."
)

# Measured for the default model and its bands on In-the-Wild test speakers, clean and
# re-encoded as Opus at 12 kbit/s. Source: eval/results/antideepfake-w2v-large/results.json;
# tests/test_cli.py checks that these still match it.
CLEAN_GENUINE_FLAGGED = 0.0007  # share of genuine clips rated likely synthetic
OPUS_GENUINE_FLAGGED = 0.0861
OPUS_ALERT_PRECISION_1_IN_100 = 0.101  # share of Opus alerts that are fakes, if 1% are fake
OPUS_WARNING = (
    "Opus audio (common in voice notes and VoIP calls) is a known weak spot. In testing at "
    f"12 kbit/s, {OPUS_GENUINE_FLAGGED:.1%} of genuine clips were rated likely synthetic, "
    f"against {CLEAN_GENUINE_FLAGGED:.1%} for uncompressed audio. If 1 recording in 100 were "
    f"fake, about {1 - OPUS_ALERT_PRECISION_1_IN_100:.0%} of likely-synthetic results on Opus "
    "audio would be false alarms. Higher bitrates were not tested. See docs/decisions.md K2."
)


def tones_warning(tones_s: float) -> str:
    return (
        f"{tones_s:.1f} s of steady tones (such as beeps, or dial, ringing or busy tones) were "
        "left out of scoring, because the model gives tones meaningless scores. Music and other "
        "non-speech sounds are not filtered and can still move the score. See docs/decisions.md "
        "K1."
    )


BAND_LABELS = {
    "likely_synthetic": "Likely synthetic",
    "uncertain": "Uncertain",
    "likely_genuine": "Likely genuine",
    "not_assessed": "Not assessed",
}

# Terminal timelines stay readable up to this many windows; longer clips show the top ones.
MAX_TIMELINE_ROWS = 30
TOP_ROWS_FOR_LONG_CLIPS = 10


def build_report(
    result: ClipResult, audio: Audio, spec: ModelSpec, config: WindowingConfig
) -> dict[str, Any]:
    if spec.bands is None:
        raise ValueError(f"{spec.name} has no band thresholds; it is for evaluation only")
    warnings = []
    if audio.codec == "opus":
        warnings.append({"code": "opus_audio", "message": OPUS_WARNING})
    if result.tones_s > 0:
        warnings.append({"code": "steady_tones", "message": tones_warning(result.tones_s)})
    return {
        "band": band_for(result.score, spec.bands),
        "score": _round(result.score),
        "calibrated": False,
        "note": result.note,
        "warnings": warnings,
        "bands": {
            "likely_synthetic_at": spec.bands.likely_synthetic_at,
            "likely_genuine_below": spec.bands.likely_genuine_below,
        },
        "audio": {
            "duration_s": round(audio.duration_s, 3),
            # With sound, not counting steady tones: the VAD can't tell speech from music.
            "active_s": round(result.speech_s, 3),
            "tones_s": round(result.tones_s, 3),  # steady tones, left out of scoring
            "container": audio.container,
            "codec": audio.codec,
        },
        "windowing": {
            "window_s": config.window_s,
            "hop_s": config.hop_s,
            "aggregation": f"top_{config.top_k}_mean",
        },
        "windows": [
            {
                "start_s": round(w.start_s, 3),
                "end_s": round(w.end_s, 3),
                "speech_fraction": round(w.speech_fraction, 3),
                "score": _round(w.score),
            }
            for w in result.windows
        ],
        "model": {"name": spec.name, "revision": spec.revision, "license": spec.license},
        "squeaktest_version": __version__,
        "disclaimer": DISCLAIMER,
    }


def format_report(report: dict[str, Any]) -> str:
    """Plain-ASCII text for a terminal (no box-drawing characters, so redirects work anywhere)."""
    lines = [f"Result:   {BAND_LABELS[report['band']]}"]
    if report["score"] is None:
        lines[0] += f"  ({report['note']})"
    else:
        lines[0] += f"  (score {report['score']:.2f}, uncalibrated)"

    audio = report["audio"]
    lines.append(
        f"Audio:    {audio['duration_s']:.1f} s, {audio['container']}/{audio['codec']}, "
        f"{audio['active_s']:.1f} s with sound"
        + (f", {audio['tones_s']:.1f} s of tones left out" if audio["tones_s"] else "")
    )
    for warning in report["warnings"]:
        wrapped = textwrap.wrap(warning["message"], width=80)
        lines += [f"Warning:  {wrapped[0]}", *(f"          {line}" for line in wrapped[1:])]
    windows = report["windows"]
    if windows:
        scored = sum(w["score"] is not None for w in windows)
        skipped = len(windows) - scored
        lines.append(f"Windows:  {scored} scored, {skipped} skipped (mostly silence or tones)")
        lines += ["", *_timeline(windows, report["windowing"])]

    model = report["model"]
    lines += [
        "",
        f"Model:    {model['name']} @ {model['revision'][:7]} ({model['license']})",
        "",
        DISCLAIMER,
        *textwrap.wrap(SCORE_NOTE, width=90),
    ]
    return "\n".join(lines)


def _timeline(windows: list[dict[str, Any]], windowing: dict[str, Any]) -> list[str]:
    header = f"Timeline ({windowing['window_s']:g} s windows every {windowing['hop_s']:g} s):"
    shown = windows
    footer = []
    if len(windows) > MAX_TIMELINE_ROWS:
        ranked = sorted(windows, key=lambda w: w["score"] or -1.0, reverse=True)
        top = {id(w) for w in ranked[:TOP_ROWS_FOR_LONG_CLIPS]}
        shown = [w for w in windows if id(w) in top]  # keep clip order
        header = f"Most suspicious {len(shown)} of {len(windows)} windows:"
        footer = ["  (use --json for every window)"]

    scored = [w for w in windows if w["score"] is not None]
    peak = max(scored, key=lambda w: w["score"]) if scored else None
    rows = [header]
    for w in shown:
        span = f"{w['start_s']:6.1f} - {w['end_s']:6.1f} s"
        if w["score"] is None:
            rows.append(f"  {span}  {'':20}  (mostly silence or tones, not scored)")
            continue
        bar = "#" * round(w["score"] * 20)
        marker = "  <- most suspicious" if w is peak else ""
        rows.append(f"  {span}  {bar:.<20}  {w['score']:.2f}{marker}")
    return rows + footer


def _round(score: float | None) -> float | None:
    return None if score is None else round(score, 4)
