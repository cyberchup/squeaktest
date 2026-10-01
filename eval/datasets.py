"""Evaluation datasets and the In-the-Wild speaker split.

In-the-Wild has 31,779 clips of 54 public figures. It is split by speaker, so no voice appears
in both parts:
- dev (about 20% of speakers): for fitting calibration and choosing band thresholds
- test (the rest): the reported numbers, never used for tuning

The dev speakers are chosen once, reproducibly, and committed in eval/splits/ so the split
can't drift.
"""

from __future__ import annotations

import csv
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

SPLITS_DIR = Path(__file__).parent / "splits"
ITW_DEV_SPEAKERS = SPLITS_DIR / "in_the_wild_dev_speakers.txt"


@dataclass(frozen=True)
class Clip:
    id: str  # unique within the dataset, e.g. "1234.wav"
    path: Path
    label: int  # 1 = synthetic, 0 = genuine
    speaker: str


def itw_root(data_dir: Path) -> Path:
    return Path(data_dir) / "datasets" / "in-the-wild" / "release_in_the_wild"


def load_in_the_wild(data_dir: Path) -> list[Clip]:
    root = itw_root(data_dir)
    with (root / "meta.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    labels = {"spoof": 1, "bona-fide": 0}
    return [Clip(r["file"], root / r["file"], labels[r["label"]], r["speaker"]) for r in rows]


def choose_dev_speakers(
    clips: Iterable[Clip],
    speaker_fraction: float = 0.2,
    clip_share: tuple[float, float] = (0.15, 0.25),
    max_fake_gap: float = 0.05,
    seed: int = 0,
    attempts: int = 100_000,
) -> list[str]:
    """Pick dev speakers at random (fixed seed) until the split is representative.

    Speakers differ wildly in size (Obama alone has over 3,600 clips) and in their share of
    fakes (from 4% to 85%), so a plain random pick can give a lopsided dev set. A pick is
    accepted when the dev set holds 15-25% of all clips and its share of fakes is within 5
    percentage points of the whole dataset's.
    """
    clips = list(clips)
    speakers = sorted({c.speaker for c in clips})
    total = len(clips)
    overall_fake = sum(c.label for c in clips) / total
    count = {s: 0 for s in speakers}
    fakes = {s: 0 for s in speakers}
    for c in clips:
        count[c.speaker] += 1
        fakes[c.speaker] += c.label
    n_dev = max(1, round(len(speakers) * speaker_fraction))
    rng = random.Random(seed)
    for _ in range(attempts):
        dev = rng.sample(speakers, n_dev)
        dev_clips = sum(count[s] for s in dev)
        share = dev_clips / total
        fake_share = sum(fakes[s] for s in dev) / dev_clips
        if (
            clip_share[0] <= share <= clip_share[1]
            and abs(fake_share - overall_fake) <= max_fake_gap
        ):
            return sorted(dev)
    raise RuntimeError("no representative speaker split found; loosen the constraints")


def read_dev_speakers(path: Path = ITW_DEV_SPEAKERS) -> set[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def split_clips(clips: Iterable[Clip], split: str, dev_speakers: set[str]) -> list[Clip]:
    """Return the clips of one split: "dev", "test" or "all"."""
    if split == "all":
        return list(clips)
    if split not in ("dev", "test"):
        raise ValueError("split must be dev, test or all")
    want_dev = split == "dev"
    return [c for c in clips if (c.speaker in dev_speakers) == want_dev]


def stratified_sample(clips: list[Clip], n: int, seed: int = 0) -> list[Clip]:
    """A reproducible random subset that keeps the dataset's ratio of fakes."""
    if n >= len(clips):
        return list(clips)
    rng = random.Random(seed)
    fake = [c for c in clips if c.label == 1]
    real = [c for c in clips if c.label == 0]
    n_fake = round(n * len(fake) / len(clips))
    picked = rng.sample(fake, n_fake) + rng.sample(real, n - n_fake)
    return sorted(picked, key=lambda c: c.id)
