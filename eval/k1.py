"""Measure K1: what squeaktest makes of non-speech audio, with the tone gate (D31) off and on.

    uv run python -m eval.k1 synthetic --split dev
    uv run python -m eval.k1 mixed --split dev --clips 300
    uv run python -m eval.k1 speech --split dev
    uv run python -m eval.k1 report

Three measurements:
- synthetic: the generated signals of eval.nonspeech on their own, under every condition. None
  of them is speech, so the right answer is "Not assessed"; "Likely synthetic" is the false
  alarm K1 is about.
- mixed: In-the-Wild clips with a beep, ringing, key presses, a busy tone, hum or synthetic
  music added before, after or under the speech (eval.nonspeech.MIXES). On genuine clips, a
  move to "Likely synthetic" is a false alarm the sound caused. On fakes, a move away from it
  is a fake that the sound, or the gate, hid.
- speech: In-the-Wild clips as they are, to measure what the gate costs on real speech. It
  reuses the window scores cached by eval.score and runs the model only on windows the gate
  moved.

Every item goes through squeaktest's own pipeline (inference.analyze), once per gate setting.
Raw results go to SQUEAKTEST_DATA_DIR/eval/k1/<model>/; `report` writes
eval/results/<model>/k1.json and k1.md. Choices are made on --split dev; test is for reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import wave
from collections import defaultdict
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from eval.analyze import RESULTS_DIR, Policy, load_records, score_records
from eval.datasets import (
    Clip,
    load_in_the_wild,
    read_dev_speakers,
    split_clips,
    stratified_sample,
)
from eval.degrade import CONDITIONS, degrade
from eval.nonspeech import MIXES, SIGNALS, SR
from eval.score import prepare, to_record
from squeaktest import __version__
from squeaktest.audio import Audio, AudioError, load_audio_file
from squeaktest.config import Settings
from squeaktest.inference import BandThresholds, WindowingConfig, analyze, band_for
from squeaktest.metrics import at_fpr, eer
from squeaktest.model import DEFAULT_MODEL, MODELS

GATES = {"off": WindowingConfig(skip_tones=False), "on": WindowingConfig(skip_tones=True)}
MIX_DESCRIPTIONS = {m.name: m.describe[0].upper() + m.describe[1:] for m in MIXES}
SYNTHETIC_SECONDS = 12.0
# The cached Phase 2 window scores that `speech` builds on, per split and condition.
CACHED_SCORES = {
    "dev": {"clean": "clean-all", **{c: f"{c}-dev" for c in CONDITIONS if c != "clean"}},
    "test": {"clean": "clean-all", **{c: f"{c}-test-n3000" for c in CONDITIONS if c != "clean"}},
}


def k1_dir(data_dir: Path, model: str) -> Path:
    return Path(data_dir) / "eval" / "k1" / model


# --- synthetic and mixed ------------------------------------------------------------------


class CachedScorer:
    """Score each distinct window once: the two gate settings mostly share windows."""

    def __init__(self, detector) -> None:
        self.detector = detector
        self.cache: dict[bytes, float] = {}

    def __call__(self, windows: Sequence[np.ndarray]) -> list[float]:
        keys = [hashlib.blake2b(w.tobytes(), digest_size=16).digest() for w in windows]
        missing = {k: w for k, w in zip(keys, windows, strict=True) if k not in self.cache}
        if missing:
            scores = self.detector(list(missing.values()))
            self.cache.update(zip(missing, scores, strict=True))
        return [self.cache[k] for k in keys]


def write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = np.clip(np.round(samples * 32767), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def encode(samples: np.ndarray, condition: str, settings: Settings) -> Audio | str:
    """Send generated audio through a condition and squeaktest's loader, as an upload would."""
    with tempfile.TemporaryDirectory(prefix="squeaktest-k1-") as tmp:
        source = Path(tmp) / "item.wav"
        write_wav(source, samples)
        try:
            return load_audio_file(degrade(source, condition, Path(tmp), settings.ffmpeg), settings)
        except (AudioError, OSError, subprocess.CalledProcessError) as exc:
            return f"{type(exc).__name__}: {exc}"


def outcomes(audio: Audio, scorer: CachedScorer, bands: BandThresholds) -> dict:
    out = {}
    for gate, config in GATES.items():
        result = analyze(audio.samples, audio.sample_rate, scorer, config)
        out[gate] = {
            "band": band_for(result.score, bands),
            "score": None if result.score is None else round(result.score, 5),
            "speech_s": round(result.speech_s, 3),
            "tones_s": round(result.tones_s, 3),
            "scored_windows": sum(w.score is not None for w in result.windows),
        }
    return out


def synthetic_items(split: str) -> Iterator[tuple[dict, np.ndarray]]:
    for s in SIGNALS[split]:
        yield {"signal": s.name, "family": s.family}, s.make(SYNTHETIC_SECONDS)


def mixed_items(split: str, clips: list[Clip], settings: Settings):
    """One group per clip: the clip under every mix (the clip is loaded once)."""
    for clip in clips:
        try:
            speech = load_audio_file(clip.path, settings).samples
        except AudioError:
            continue  # the product would reject it too; eval.score records these
        base = {"clip": clip.id, "label": clip.label, "speaker": clip.speaker}
        yield [({**base, "mix": m.name}, m.make(speech, split)) for m in MIXES]


def run_items(
    groups, kind: str, split: str, conditions: list[str], out: Path, detector, settings: Settings
) -> int:
    """Analyse every item of every group under every condition; append records to `out`."""
    bands = detector.spec.bands
    done = set()
    if out.exists():
        done = {(r["condition"], r["item"]) for r in load_records(out)}
    added = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool, out.open("a", encoding="utf-8") as f:
        for group in groups:
            tasks = [
                (key, samples, condition)
                for key, samples in group
                for condition in conditions
                if (condition, json.dumps(key, sort_keys=True)) not in done
            ]
            encoded = pool.map(lambda t: encode(t[1], t[2], settings), tasks)
            scorer = CachedScorer(detector)  # windows repeat within a group, not across
            for (key, samples, condition), audio in zip(tasks, encoded, strict=True):
                record = {
                    "kind": kind,
                    "split": split,
                    "condition": condition,
                    "item": json.dumps(key, sort_keys=True),
                    **key,
                    "duration_s": round(samples.size / SR, 3),
                }
                if isinstance(audio, str):
                    record["error"] = audio
                else:
                    record |= outcomes(audio, scorer, bands)
                f.write(json.dumps(record) + "\n")
                added += 1
            f.flush()
            print(f"  {kind} {split}: {added} items analysed", flush=True)
    return added


# --- speech -------------------------------------------------------------------------------


def gated_speech(
    split: str, condition: str, detector, settings: Settings, model: str, chunk_size: int = 64
) -> Path:
    """Re-plan cached In-the-Wild clips with the gate on, reusing every unchanged window score.

    Writes records in eval.score's format, plus `tones_s` (seconds the gate removed) and
    `rescored` (whether any window moved, so the model ran again).
    """
    cached_path = (
        Path(settings.data_dir)
        / "eval"
        / "scores"
        / model
        / "in-the-wild"
        / f"{CACHED_SCORES[split][condition]}.jsonl"
    )
    dev = read_dev_speakers()
    cached = [r for r in load_records(cached_path) if (r["speaker"] in dev) == (split == "dev")]
    clips = {c.id: c for c in load_in_the_wild(settings.data_dir)}
    out = k1_dir(settings.data_dir, model) / "speech" / f"{condition}-{split}.jsonl"
    done = {r["id"] for r in load_records(out)} if out.exists() else set()
    todo = [r for r in cached if r["id"] not in done]
    # As eval.score does: plan and score every window, and apply the product's rules later.
    config = WindowingConfig(min_speech_s=0.0, min_speech_fraction=0.0, skip_tones=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool, out.open("a", encoding="utf-8") as f:
        for first in range(0, len(todo), chunk_size):
            chunk = todo[first : first + chunk_size]
            prepared = list(
                pool.map(lambda r: prepare(clips[r["id"]], condition, settings, config), chunk)
            )
            moved = []
            for old, p in zip(chunk, prepared, strict=True):
                spans = [[round(s, 3), round(e, 3)] for s, e, _, _ in p.windows]
                moved.append(spans != [w[:2] for w in old["windows"]])
            margins = [
                [w[3] for w in old["windows"]] if not m else []
                for old, m in zip(chunk, moved, strict=True)
            ]
            redo = [i for i, m in enumerate(moved) if m and prepared[i].audio]
            if redo:
                logits = detector.logits([a for i in redo for a in prepared[i].audio])
                fresh = iter((logits[:, 0] - logits[:, 1]).tolist())
                for i in redo:
                    margins[i] = [next(fresh) for _ in prepared[i].audio]
            for old, p, m, was_moved in zip(chunk, prepared, margins, moved, strict=True):
                record = to_record(p, m)
                record["tones_s"] = round(max(old["active_s"] - record["active_s"], 0.0), 3)
                record["rescored"] = was_moved
                f.write(json.dumps(record) + "\n")
            f.flush()
            print(f"  speech {condition} {split}: {first + len(chunk)}/{len(todo)}", flush=True)
    return out


# --- report -------------------------------------------------------------------------------


BAND_NAMES = ("likely_synthetic", "uncertain", "likely_genuine", "not_assessed")
LABELS = {0: "genuine", 1: "synthetic", None: "non-speech"}


def _band_counts(rows: list[dict], gate: str) -> dict[str, int]:
    bands = [r[gate]["band"] for r in rows if "error" not in r]
    return {b: bands.count(b) for b in BAND_NAMES if b in bands} | {"n": len(bands)}


def summarize_items(records: list[dict], by: str) -> dict:
    """Band counts per `by` (signal or mix), condition ("all" pooled) and label, per gate."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        label = LABELS[r.get("label")]
        groups[(r[by], r["condition"], label)].append(r)
        groups[(r[by], "all", label)].append(r)
    out: dict = {}
    for (name, condition, label), rows in groups.items():
        cell = {g: _band_counts(rows, g) for g in GATES}
        out.setdefault(name, {}).setdefault(condition, {})[label] = cell
    return out


def speech_effect(old: list[dict], new: list[dict], bands: BandThresholds) -> dict:
    """What the gate does to real speech: coverage, EER, detection and the D18 bands."""
    policy = Policy()
    out = {}
    for gate, records in (("off", old), ("on", new)):
        d = score_records(records, policy)
        flagged = d.scores >= bands.likely_synthetic_at
        out[gate] = {
            "clips": sum(d.total.values()),
            "not_assessed": {"genuine": d.not_assessed[0], "synthetic": d.not_assessed[1]},
            "eer": eer(d.labels, d.scores)[0],
            "tpr_at_1pct_fpr": at_fpr(d.labels, d.scores, 0.01).tpr,
            "genuine_likely_synthetic": float(flagged[d.labels == 0].mean()),
            "synthetic_likely_synthetic": float(flagged[d.labels == 1].mean()),
        }
    active = sum(r["active_s"] for r in old)
    removed = [r["tones_s"] for r in new]
    out["tones"] = {
        "share_of_activity_removed": sum(removed) / active if active else 0.0,
        "clips_with_any_removed": sum(t > 0 for t in removed),
        "clips_rescored": sum(r["rescored"] for r in new),
        "largest_removed_s": max(removed, default=0.0),
    }
    return out


def _unique(records: list[dict], key: str) -> list[dict]:
    """The first record per key: a resumed or repeated run may have appended a second one."""
    first: dict = {}
    for r in records:
        first.setdefault(r[key], r)
    return list(first.values())


def report(data_dir: Path, model: str) -> dict:
    root = k1_dir(data_dir, model)
    bands = MODELS[model].bands
    results: dict = {"model": model, "generated": datetime.now(UTC).isoformat(timespec="seconds")}
    for split in ("dev", "test"):
        part: dict = {}
        for kind, by in (("synthetic", "signal"), ("mixed", "mix")):
            path = root / f"{kind}-{split}.jsonl"
            if path.exists():
                records = load_records(path)
                for r in records:
                    r["key"] = (r["condition"], r["item"])
                part[kind] = summarize_items(_unique(records, "key"), by)
        speech = {}
        for condition in CONDITIONS:
            new_path = root / "speech" / f"{condition}-{split}.jsonl"
            if not new_path.exists():
                continue
            new = _unique(load_records(new_path), "id")
            ids = {r["id"] for r in new}
            cached = (
                Path(data_dir)
                / "eval"
                / "scores"
                / model
                / "in-the-wild"
                / f"{CACHED_SCORES[split][condition]}.jsonl"
            )
            old = [r for r in load_records(cached) if r["id"] in ids]
            speech[condition] = speech_effect(old, new, bands)
        if speech:
            part["speech"] = speech
        if part:
            results[split] = part
    out_dir = RESULTS_DIR / model
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "k1.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (out_dir / "k1.md").write_text(render(results), encoding="utf-8")
    return results


SHORT = {
    "likely_synthetic": "synthetic",
    "uncertain": "uncertain",
    "likely_genuine": "genuine",
    "not_assessed": "not assessed",
}
CONDITION_NAMES = {
    "clean": "Clean",
    "g711": "G.711",
    "amr-nb": "AMR-NB",
    "opus": "Opus",
    "mp3": "MP3",
}


def _bands(counts: dict) -> str:
    """Band counts as text, most common first: "4 synthetic, 1 uncertain"."""
    parts = sorted(((counts[b], b) for b in BAND_NAMES if b in counts), key=lambda p: -p[0])
    return ", ".join(f"{n} {SHORT[b]}" for n, b in parts) or "-"


def _share(counts: dict, band: str) -> str:
    return f"{counts.get(band, 0) / counts['n']:.1%}" if counts["n"] else "-"


def _change(cell: dict, band: str) -> str:
    return f"{_share(cell['off'], band)} -> {_share(cell['on'], band)}"


def render(results: dict) -> str:
    lines = [
        "# K1: non-speech audio, with the tone gate off and on",
        "",
        f"Generated by `python -m eval.k1 report` on {results['generated'][:10]}, model"
        f" `{results['model']}`. *Off* is the Phase 2 pipeline; *on* adds the tone gate (D31,"
        " `docs/decisions.md`). Bands use the D18 thresholds. Dev results were used to check the"
        " gate's settings; test results are for reporting. Numbers in `k1.json`.",
    ]
    for split in ("test", "dev"):
        part = results.get(split)
        if not part:
            continue
        lines += ["", f"## {split.capitalize()} split", ""]
        if "synthetic" in part:
            lines += [
                "### Non-speech on its own",
                "",
                f"Generated signals, {SYNTHETIC_SECONDS:g} s each at about -20 dBFS, under each"
                " condition. The right answer is *Not assessed*.",
                "",
                "| Signal | Gate off | Gate on |",
                "|---|---|---|",
            ]
            for name, by_condition in part["synthetic"].items():
                cell = by_condition["all"]["non-speech"]
                lines.append(f"| {name} | {_bands(cell['off'])} | {_bands(cell['on'])} |")
            lines.append("")
        if "mixed" in part:
            any_mix = next(iter(part["mixed"].values()))
            conditions = [CONDITION_NAMES[c] for c in any_mix if c != "all"]
            n = any_mix["all"]
            clips = (n["genuine"]["off"]["n"] + n["synthetic"]["off"]["n"]) // len(conditions)
            lines += [
                "### Sounds added to real clips",
                "",
                f"{clips} In-the-Wild clips of {split} speakers, each with every sound added,"
                f" pooled over {', '.join(conditions)}. Off -> on. A fake that is *Not assessed*"
                " counts as not caught. Where the gate raises *Not assessed*, it goes back to the"
                " clip's own level: the added tone had padded short clips past the 1 s minimum.",
                "",
                "| Added sound | Genuine rated *Likely synthetic* | Fakes rated *Likely synthetic*"
                " | Genuine *Not assessed* | Fakes *Not assessed* |",
                "|---|---|---|---|---|",
            ]
            for name, by_condition in part["mixed"].items():
                g, s = by_condition["all"]["genuine"], by_condition["all"]["synthetic"]
                lines.append(
                    f"| {MIX_DESCRIPTIONS.get(name, name)} | {_change(g, 'likely_synthetic')}"
                    f" | {_change(s, 'likely_synthetic')} | {_change(g, 'not_assessed')}"
                    f" | {_change(s, 'not_assessed')} |"
                )
            lines.append("")
        if "speech" in part:
            lines += [
                "### Real speech: what the gate costs",
                "",
                "In-the-Wild clips as they are (clean: every clip of these speakers; codecs: the"
                " Phase 2 samples). Off -> on.",
                "",
                "| Condition | Sound removed | Clips touched | EER | Fakes caught at 1% FPR |"
                " Genuine rated *Likely synthetic* | Not assessed |",
                "|---|---|---|---|---|---|---|",
            ]
            for condition, e in part["speech"].items():
                off, on, tones = e["off"], e["on"], e["tones"]

                def pct(key: str, a: dict = off, b: dict = on) -> str:
                    return f"{a[key]:.2%} -> {b[key]:.2%}"

                na = [sum(x["not_assessed"].values()) / x["clips"] for x in (off, on)]
                lines.append(
                    f"| {CONDITION_NAMES[condition]} | {tones['share_of_activity_removed']:.3%}"
                    f" | {tones['clips_with_any_removed']} of {off['clips']}"
                    f" | {pct('eer')} | {pct('tpr_at_1pct_fpr')}"
                    f" | {pct('genuine_likely_synthetic')} | {na[0]:.2%} -> {na[1]:.2%} |"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("what", choices=["synthetic", "mixed", "speech", "report"])
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--clips", type=int, default=300, help="In-the-Wild clips for `mixed`")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    if args.what == "report":
        report(settings.data_dir, args.model)
        print(f"Wrote {RESULTS_DIR / args.model / 'k1.json'}")
        return 0

    from squeaktest.model import load_detector

    detector = load_detector(args.model, data_dir=settings.data_dir, device=args.device)
    root = k1_dir(settings.data_dir, args.model)
    meta = {
        "what": args.what,
        "split": args.split,
        "conditions": args.conditions,
        "clips": args.clips,
        "seed": args.seed,
        "model": args.model,
        "model_revision": MODELS[args.model].revision,
        "gates": {g: c.skip_tones for g, c in GATES.items()},
        "squeaktest_version": __version__,
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{args.what}-{args.split}.meta.json").write_text(json.dumps(meta, indent=2))
    if args.what == "synthetic":
        groups = ([item] for item in synthetic_items(args.split))
        out = root / f"synthetic-{args.split}.jsonl"
        run_items(groups, "synthetic", args.split, args.conditions, out, detector, settings)
    elif args.what == "mixed":
        clips = split_clips(load_in_the_wild(settings.data_dir), args.split, read_dev_speakers())
        clips = stratified_sample(clips, args.clips, args.seed)
        groups = mixed_items(args.split, clips, settings)
        out = root / f"mixed-{args.split}.jsonl"
        run_items(groups, "mixed", args.split, args.conditions, out, detector, settings)
    else:
        for condition in args.conditions:
            gated_speech(args.split, condition, detector, settings, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
