"""Turn cached window scores into evaluation results (Phase 2).

    uv run --group eval python -m eval.analyze

Reads the score files written by eval.score and writes eval/results/<model>/:
results.json (every number), report.md (tables) and plots.

Discipline (CLAUDE.md, Evaluation discipline):
- Choices are made on the dev speakers only: the Platt calibration, the band thresholds (on
  the dev speakers' clips under every codec, D18) and the minimum-speech rule. Test speakers
  are only ever used to report.
- Confidence intervals resample whole speakers, since clips of one speaker aren't independent.
- Clips the product wouldn't score ("not assessed") are left out of the metrics and reported
  separately, per class, so coverage is visible instead of silently dropped.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from eval.datasets import read_dev_speakers
from eval.degrade import CONDITIONS
from squeaktest.calibration import PlattCalibrator, fit_platt, shift_prior
from squeaktest.config import Settings
from squeaktest.inference import aggregate
from squeaktest.metrics import (
    Interval,
    at_fpr,
    bootstrap,
    brier_score,
    eer,
    expected_calibration_error,
    precision_at_prevalence,
    reliability_bins,
    roc_auc,
)

RESULTS_DIR = Path(__file__).parent / "results"
SYNTHETIC_BAND_MAX_FPR = 0.01  # "likely synthetic" may flag at most 1% of genuine dev clips
GENUINE_BAND_MIN_RECALL = 0.99  # "likely genuine" may hold at most 1% of fake dev clips
DURATION_BUCKETS = [(0, 2), (2, 4), (4, 8), (8, 1e9)]
PREVALENCES = (0.1, 0.01, 0.001)  # share of recordings that are fake, for the alert view


@dataclass(frozen=True)
class Policy:
    """The product's rules for which clips and windows get scored, and how scores combine."""

    min_speech_s: float = 1.0
    min_speech_fraction: float = 0.5
    top_k: int = 3


@dataclass
class Scored:
    """Clip-level arrays for the clips the policy assessed, plus coverage counts."""

    ids: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    speakers: np.ndarray
    durations: np.ndarray
    not_assessed: dict[int, int]  # label -> count
    total: dict[int, int]

    def subset(self, mask: np.ndarray) -> Scored:
        return Scored(
            self.ids[mask],
            self.labels[mask],
            self.scores[mask],
            self.speakers[mask],
            self.durations[mask],
            self.not_assessed,
            self.total,
        )


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clip_score(record: dict, policy: Policy) -> float | None:
    """The score the product would give this clip, or None if it wouldn't assess it."""
    if record["error"] or record["active_s"] < policy.min_speech_s:
        return None
    margins = [w[3] for w in record["windows"] if w[2] >= policy.min_speech_fraction]
    if not margins:
        return None
    probs = 1.0 / (1.0 + np.exp(-np.asarray(margins, dtype=np.float64)))
    return aggregate(probs, policy.top_k)


def score_records(records: list[dict], policy: Policy) -> Scored:
    rows, not_assessed, total = [], {0: 0, 1: 0}, {0: 0, 1: 0}
    for r in records:
        total[r["label"]] += 1
        s = clip_score(r, policy)
        if s is None:
            not_assessed[r["label"]] += 1
        else:
            rows.append((r["id"], r["label"], s, r["speaker"], r["duration_s"]))
    ids, labels, scores, speakers, durations = (np.array(col) for col in zip(*rows, strict=True))
    return Scored(
        ids,
        labels.astype(int),
        scores.astype(float),
        speakers,
        durations.astype(float),
        not_assessed,
        total,
    )


def _eer(y: np.ndarray, s: np.ndarray) -> float:
    return eer(y, s)[0]


def _tpr_at(max_fpr: float) -> Callable[[np.ndarray, np.ndarray], float]:
    return lambda y, s: at_fpr(y, s, max_fpr).tpr


def summarize(d: Scored, n_boot: int = 1000) -> dict:
    """Headline metrics with speaker-bootstrap 95% intervals."""

    def ci(metric) -> dict:
        return asdict(bootstrap(metric, d.labels, d.scores, groups=d.speakers, n_resamples=n_boot))

    return {
        "clips": int(d.labels.size),
        "fakes": int(d.labels.sum()),
        "speakers": int(np.unique(d.speakers).size),
        "eer": ci(_eer),
        "auc": ci(roc_auc),
        "tpr_at_1pct_fpr": ci(_tpr_at(0.01)),
        "tpr_at_5pct_fpr": ci(_tpr_at(0.05)),
    }


def coverage(d: Scored) -> dict:
    return {
        name: {
            "not_assessed": d.not_assessed[label],
            "of": d.total[label],
            "share": d.not_assessed[label] / d.total[label] if d.total[label] else 0.0,
        }
        for name, label in (("genuine", 0), ("synthetic", 1))
    }


def split_mask(d: Scored, dev_speakers: set[str], split: str) -> np.ndarray:
    in_dev = np.isin(d.speakers, list(dev_speakers))
    return in_dev if split == "dev" else ~in_dev


def choose_thresholds(dev_labels: np.ndarray, dev_probs: np.ndarray) -> dict:
    """Band thresholds, chosen on dev data only.

    Symmetric rule: each "likely" band may hold at most 1% of the other class on dev, and
    everything between the two thresholds is "uncertain".
    """
    synthetic = at_fpr(dev_labels, dev_probs, SYNTHETIC_BAND_MAX_FPR).threshold
    fake_probs = np.sort(dev_probs[dev_labels == 1])
    # Highest threshold that still leaves at least 99% of dev fakes at or above it.
    genuine = float(fake_probs[int(np.floor((1 - GENUINE_BAND_MIN_RECALL) * fake_probs.size))])
    return {
        "likely_synthetic_at": float(synthetic),
        "likely_genuine_below": min(genuine, float(synthetic)),
    }


def band_table(labels: np.ndarray, probs: np.ndarray, thresholds: dict) -> dict:
    bands = np.where(
        probs >= thresholds["likely_synthetic_at"],
        "likely_synthetic",
        np.where(probs < thresholds["likely_genuine_below"], "likely_genuine", "uncertain"),
    )
    table = {}
    for name, label in (("genuine", 0), ("synthetic", 1)):
        mask = labels == label
        table[name] = {
            b: float((bands[mask] == b).mean())
            for b in ("likely_synthetic", "uncertain", "likely_genuine")
        }
    return table


def alerts(d: Scored, threshold: float, n_boot: int) -> dict:
    """What a "likely synthetic" alert means, with speaker-bootstrap intervals.

    The false-positive rate here can rest on a handful of genuine clips, and the share of
    alerts that are real fakes at a low base rate hinges on it, so the intervals matter more
    than the estimates.
    """

    def ci(metric) -> dict:
        return asdict(bootstrap(metric, d.labels, d.scores, groups=d.speakers, n_resamples=n_boot))

    def rates(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
        flagged = s >= threshold
        return float(flagged[y == 1].mean()), float(flagged[y == 0].mean())

    def precision(prevalence: float) -> Callable[[np.ndarray, np.ndarray], float]:
        return lambda y, s: precision_at_prevalence(*rates(y, s), prevalence)

    return {
        "tpr": ci(lambda y, s: rates(y, s)[0]),
        "fpr": ci(lambda y, s: rates(y, s)[1]),
        "precision_at_prevalence": {f"{p:g}": ci(precision(p)) for p in PREVALENCES},
    }


def dev_threshold_pool(scores_dir: Path, dev: Scored, policy: Policy) -> tuple[Scored, list[str]]:
    """Dev clips under every condition scored for the dev speakers, pooled (decision D18).

    The bands are meant for the audio a SOC receives, which is mostly phone calls and
    compressed voice notes, so the thresholds are chosen on dev speakers under those codecs as
    well as clean. Each condition holds the same dev clips, so each gets equal weight.
    Conditions that weren't scored for the dev speakers are left out, and the result says so.
    """
    parts, fit_on = [dev], ["clean"]
    for name in CONDITIONS:
        path = scores_dir / f"{name}-dev.jsonl"
        if name != "clean" and path.exists():
            parts.append(score_records(load_records(path), policy))
            fit_on.append(name)
    pooled = Scored(
        np.concatenate([p.ids for p in parts]),
        np.concatenate([p.labels for p in parts]),
        np.concatenate([p.scores for p in parts]),
        np.concatenate([p.speakers for p in parts]),
        np.concatenate([p.durations for p in parts]),
        {},
        {},
    )
    return pooled, fit_on


def analyze(data_dir: Path, model: str, n_boot: int = 1000) -> dict:
    scores_dir = Path(data_dir) / "eval" / "scores" / model / "in-the-wild"
    dev_speakers = read_dev_speakers()
    policy = Policy()
    clean_records = load_records(scores_dir / "clean-all.jsonl")
    clean = score_records(clean_records, policy)
    dev = clean.subset(split_mask(clean, dev_speakers, "dev"))
    test = clean.subset(split_mask(clean, dev_speakers, "test"))
    test_records = [r for r in clean_records if r["speaker"] not in dev_speakers]
    test_cov = score_records(test_records, policy)
    results: dict = {"model": model, "policy": asdict(policy)}

    # 1. Headline: clean test speakers, product policy, raw scores
    results["test_clean"] = summarize(test, n_boot)
    results["test_clean"]["coverage"] = coverage(test_cov)
    results["dev_clean"] = summarize(dev, n_boot // 4)

    # 2. The length shortcut: how far does clip duration alone get?
    results["length_baseline_test"] = {
        "auc": roc_auc(test.labels, test.durations),
        "eer": eer(test.labels, test.durations)[0],
    }
    results["by_duration_test"] = []
    for low, high in DURATION_BUCKETS:
        mask = (test.durations >= low) & (test.durations < high)
        if mask.sum() > 50 and 0 < test.labels[mask].sum() < mask.sum():
            sub = test.subset(mask)
            results["by_duration_test"].append(
                {
                    "seconds": [low, None if high > 1e8 else high],
                    "clips": int(mask.sum()),
                    "fake_share": float(sub.labels.mean()),
                    "eer": _eer(sub.labels, sub.scores),
                    "auc": roc_auc(sub.labels, sub.scores),
                }
            )
    genuine = test.labels == 0
    results["genuine_score_vs_duration_spearman"] = _spearman(
        test.durations[genuine], test.scores[genuine]
    )

    # 3. Aggregation: mean vs top-3 vs max (decision D6)
    results["aggregation_test"] = {}
    for name, k in (("mean", 10_000), ("top_3", 3), ("max", 1)):
        variant = score_records(test_records, Policy(top_k=k))
        results["aggregation_test"][name] = {
            "eer": _eer(variant.labels, variant.scores),
            "auc": roc_auc(variant.labels, variant.scores),
            "tpr_at_1pct_fpr": at_fpr(variant.labels, variant.scores, 0.01).tpr,
        }

    # 4. Minimum speech (decision D5), evaluated on dev
    dev_records = [r for r in clean_records if r["speaker"] in dev_speakers]
    results["min_speech_dev"] = []
    for min_s in (0.0, 0.5, 1.0, 1.5, 2.0):
        variant = score_records(dev_records, Policy(min_speech_s=min_s))
        cov = coverage(variant)
        results["min_speech_dev"].append(
            {
                "min_speech_s": min_s,
                "eer": _eer(variant.labels, variant.scores),
                "not_assessed_genuine": cov["genuine"]["share"],
                "not_assessed_synthetic": cov["synthetic"]["share"],
            }
        )

    # 5. Calibration, fitted on dev
    calibrator = fit_platt(dev.labels, dev.scores)
    test_probs = calibrator(test.scores)
    results["calibration"] = {
        "platt": asdict(calibrator),
        "test_raw": {
            "ece": expected_calibration_error(test.labels, test.scores),
            "brier": brier_score(test.labels, test.scores),
        },
        "test_calibrated": {
            "ece": expected_calibration_error(test.labels, test_probs),
            "brier": brier_score(test.labels, test_probs),
        },
        "reliability_raw": [asdict(b) for b in reliability_bins(test.labels, test.scores)],
        "reliability_calibrated": [asdict(b) for b in reliability_bins(test.labels, test_probs)],
    }

    # 6. Bands: thresholds chosen on dev, reported on test
    # Bands use raw scores: the dev-fitted calibration made test calibration worse (D16).
    # Thresholds from clean dev clips alone are kept for comparison, to show what pooling the
    # codecs in (D18) costs on clean audio and buys on compressed audio.
    pool, fit_on = dev_threshold_pool(scores_dir, dev, policy)
    thresholds = choose_thresholds(pool.labels, pool.scores)
    clean_only = choose_thresholds(dev.labels, dev.scores)
    t_syn = thresholds["likely_synthetic_at"]
    results["bands"] = {
        "fit_on": fit_on,
        "dev_clips": int(pool.labels.size),
        "thresholds": thresholds,
        "thresholds_clean_dev_only": clean_only,
        "rules": {
            "likely_synthetic_max_dev_fpr": SYNTHETIC_BAND_MAX_FPR,
            "likely_genuine_min_dev_fake_recall": GENUINE_BAND_MIN_RECALL,
        },
        "test": band_table(test.labels, test.scores, thresholds),
        "test_clean_dev_only": band_table(test.labels, test.scores, clean_only),
        "test_alerts": alerts(test, t_syn, n_boot),
    }
    fitted_share = calibrator.fitted_fake_share
    results["prior_shift_example"] = {
        "fitted_fake_share": fitted_share,
        "calibrated_0.9_at_1_in_1000": float(shift_prior([0.9], fitted_share, 0.001)[0]),
    }

    # 7. In-domain reference: ASVspoof 2019 LA eval (seen data for AntiDeepfake)
    asv_path = (
        Path(data_dir) / "eval" / "scores" / model / "asvspoof2019-la" / "clean-all-n5000.jsonl"
    )
    if asv_path.exists():
        asv_records = load_records(asv_path)
        asv = score_records(asv_records, policy)
        results["asvspoof2019_la"] = summarize(asv, n_boot // 2)
        results["asvspoof2019_la"]["coverage"] = coverage(asv)

    # 8. Phone and compression conditions, on the stratified test sample
    results["conditions"] = {}
    for name in CONDITIONS:
        if name == "clean":
            continue
        path = scores_dir / f"{name}-test-n3000.jsonl"
        if not path.exists():
            continue
        cond_records = load_records(path)
        ids = {r["id"] for r in cond_records}
        cond = score_records(cond_records, policy)
        paired_clean = score_records([r for r in test_records if r["id"] in ids], policy)
        results["conditions"][name] = {
            "description": CONDITIONS[name].description,
            "metrics": summarize(cond, n_boot // 2),
            "coverage": coverage(cond),
            "same_clips_clean": {
                "eer": _eer(paired_clean.labels, paired_clean.scores),
                "auc": roc_auc(paired_clean.labels, paired_clean.scores),
            },
            "bands": band_table(cond.labels, cond.scores, thresholds),
            "bands_clean_dev_only": band_table(cond.labels, cond.scores, clean_only),
            "alerts": alerts(cond, t_syn, n_boot // 2),
        }
    return results


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Interval, PlattCalibrator)):
        return asdict(value)
    raise TypeError(f"not serializable: {type(value)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze cached evaluation scores.")
    parser.add_argument("--model", default="antideepfake-mms-300m")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    results = analyze(Settings.from_env().data_dir, args.model, args.bootstrap)
    out_dir = RESULTS_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(results, indent=2, default=_json_default) + "\n"
    (out_dir / "results.json").write_text(payload, encoding="utf-8")
    from eval.report import write_plots, write_report

    write_report(results, out_dir / "report.md")
    write_plots(results, out_dir)
    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
