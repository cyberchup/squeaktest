"""Score a dataset under one condition and cache every window's raw model output.

    uv run python -m eval.score --condition clean --split all
    uv run python -m eval.score --condition g711 --split test --sample 3000

Each clip goes through squeaktest's own pipeline: (degrade) -> load_audio_file -> plan_windows
-> detector. For every window, the logit margin (fake logit minus real logit) is stored, not a
rounded probability, because this model's scores pile up near 1.0 where rounding would erase
the ranking. Aggregation, calibration and thresholds are then computed offline by eval.analyze,
so trying a different aggregation never means re-running the model. Every window is
scored, even mostly-silent ones and clips under the product's 1 s minimum; the product's
skip rules are applied offline too.

Output: SQUEAKTEST_DATA_DIR/eval/scores/<model>/<dataset>/<condition>-<split>[-n<sample>].jsonl,
one line per clip, plus a .meta.json describing the run. Re-running resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from eval.datasets import (
    Clip,
    load_in_the_wild,
    read_dev_speakers,
    split_clips,
    stratified_sample,
)
from eval.degrade import CONDITIONS, degrade
from squeaktest import __version__
from squeaktest.audio import AudioError, load_audio_file
from squeaktest.config import Settings
from squeaktest.inference import WindowingConfig, plan_windows
from squeaktest.vad import speech_fraction


@dataclass
class Prepared:
    clip: Clip
    duration_s: float = 0.0
    active_s: float = 0.0
    windows: list[tuple[float, float, float, bool]] = field(default_factory=list)
    audio: list[np.ndarray] = field(default_factory=list)  # samples of the scoreable windows
    error: str | None = None


def prepare(clip: Clip, condition: str, settings: Settings, config: WindowingConfig) -> Prepared:
    """Degrade, load and window one clip (runs in a worker thread)."""
    result = Prepared(clip)
    try:
        with tempfile.TemporaryDirectory(prefix="squeaktest-eval-") as tmp:
            source = degrade(clip.path, condition, Path(tmp), settings.ffmpeg)
            audio = load_audio_file(source, settings)
    except (AudioError, OSError) as exc:  # a clip the product would also reject
        result.error = f"{type(exc).__name__}: {exc}"
        return result
    sr = audio.sample_rate
    result.duration_s = audio.duration_s
    result.active_s = speech_fraction(audio.samples, sr) * audio.samples.size / sr
    for w in plan_windows(audio.samples, sr, config):
        scoreable = w.speech_fraction >= config.min_speech_fraction
        result.windows.append((w.start / sr, w.end / sr, w.speech_fraction, scoreable))
        if scoreable:
            result.audio.append(audio.samples[w.start : w.end])
    return result


def to_record(p: Prepared, margins: Sequence[float]) -> dict:
    margin_iter = iter(margins)
    windows = [
        [round(start, 3), round(end, 3), round(frac, 4), float(next(margin_iter)) if ok else None]
        for start, end, frac, ok in p.windows
    ]
    return {
        "id": p.clip.id,
        "label": p.clip.label,
        "speaker": p.clip.speaker,
        "duration_s": round(p.duration_s, 3),
        "active_s": round(p.active_s, 3),
        "windows": windows,  # [start_s, end_s, speech_fraction, margin or null if skipped]
        "error": p.error,
    }


def score_clips(
    clips: Sequence[Clip],
    condition: str,
    out_path: Path,
    detector,
    settings: Settings,
    config: WindowingConfig,
    workers: int = 6,
    chunk_size: int = 64,
    progress: bool = True,
) -> int:
    """Score clips not yet in out_path and append their records. Returns how many were added."""
    done = _done_ids(out_path)
    todo = [c for c in clips if c.id not in done]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    added = 0
    with (
        ThreadPoolExecutor(max_workers=workers) as pool,
        out_path.open("a", encoding="utf-8") as out,
    ):
        for chunk in _chunks(todo, chunk_size):
            prepared = list(pool.map(lambda c: prepare(c, condition, settings, config), chunk))
            window_audio = [a for p in prepared for a in p.audio]
            logits = detector.logits(window_audio) if window_audio else np.empty((0, 2))
            margins = (logits[:, 0] - logits[:, 1]).tolist()
            offset = 0
            for p in prepared:
                n = len(p.audio)
                out.write(json.dumps(to_record(p, margins[offset : offset + n])) + "\n")
                offset += n
            out.flush()
            added += len(prepared)
            if progress:
                rate = added / (time.perf_counter() - start)
                left = (len(todo) - added) / rate if rate else 0
                print(
                    f"  {len(done) + added}/{len(clips)} clips, {rate:.1f}/s,"
                    f" ~{left / 60:.0f} min left",
                    flush=True,
                )
    return added


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def _chunks(items: Sequence[Clip], size: int) -> Iterator[Sequence[Clip]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def output_path(
    data_dir: Path, model: str, dataset: str, condition: str, split: str, sample: int | None
) -> Path:
    name = f"{condition}-{split}" + (f"-n{sample}" if sample else "") + ".jsonl"
    return Path(data_dir) / "eval" / "scores" / model / dataset / name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="in-the-wild", choices=["in-the-wild"])
    parser.add_argument("--condition", default="clean", choices=sorted(CONDITIONS))
    parser.add_argument("--split", default="all", choices=["all", "dev", "test"])
    parser.add_argument("--sample", type=int, help="score a stratified random subset of this size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="antideepfake-mms-300m")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)

    from squeaktest.model import MODELS, load_detector

    settings = Settings.from_env()
    # Score every window, including ones the product would skip, and apply the product's
    # minimum-speech rules later in eval.analyze. Then those rules can be tuned on dev data
    # without re-running the model.
    config = WindowingConfig(min_speech_s=0.0, min_speech_fraction=0.0)
    clips = split_clips(load_in_the_wild(settings.data_dir), args.split, read_dev_speakers())
    if args.sample:
        clips = stratified_sample(clips, args.sample, args.seed)
    out = output_path(
        settings.data_dir, args.model, args.dataset, args.condition, args.split, args.sample
    )
    spec = MODELS[args.model]
    meta = {
        "dataset": args.dataset,
        "condition": args.condition,
        "condition_description": CONDITIONS[args.condition].description,
        "split": args.split,
        "sample": args.sample,
        "seed": args.seed,
        "clips": len(clips),
        "model": spec.name,
        "model_revision": spec.revision,
        "windowing": asdict(config),
        "squeaktest_version": __version__,
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Scoring {len(clips)} clips ({args.condition}, {args.split}) -> {out}")
    detector = load_detector(args.model, data_dir=settings.data_dir, device=args.device)
    added = score_clips(
        clips, args.condition, out, detector, settings, config, workers=args.workers
    )
    print(f"Done: {added} new clips scored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
