"""Turn an analysis into a report: a JSON-ready dict, and a human-readable text version.

The report never includes the input's filename or path, so it is safe to log or share.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from squeaktest import __version__
from squeaktest.inference import ClipResult, WindowingConfig, band_for

if TYPE_CHECKING:
    from squeaktest.audio import Audio
    from squeaktest.model import ModelSpec

DISCLAIMER = "This score is an estimate. It must not be the sole basis for any decision."
UNCALIBRATED = (
    "Scores are not calibrated yet, so the bands are provisional (docs/decisions.md D9 and D14)."
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
    return {
        "band": band_for(result.score),
        "score": _round(result.score),
        "calibrated": False,
        "note": result.note,
        "audio": {
            "duration_s": round(audio.duration_s, 3),
            "active_s": round(result.speech_s, 3),  # with sound: VAD cannot tell speech apart
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
    )
    windows = report["windows"]
    if windows:
        scored = sum(w["score"] is not None for w in windows)
        lines.append(f"Windows:  {scored} scored, {len(windows) - scored} skipped (mostly silence)")
        lines += ["", *_timeline(windows, report["windowing"])]

    model = report["model"]
    lines += [
        "",
        f"Model:    {model['name']} @ {model['revision'][:7]} ({model['license']})",
        "",
        DISCLAIMER,
        UNCALIBRATED,
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
            rows.append(f"  {span}  {'':20}  (mostly silence, not scored)")
            continue
        bar = "#" * round(w["score"] * 20)
        marker = "  <- most suspicious" if w is peak else ""
        rows.append(f"  {span}  {bar:.<20}  {w['score']:.2f}{marker}")
    return rows + footer


def _round(score: float | None) -> float | None:
    return None if score is None else round(score, 4)
