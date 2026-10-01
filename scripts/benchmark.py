"""Measure how long the detector takes per minute of audio, on GPU and on CPU.

    uv run python scripts/benchmark.py

One minute of audio becomes 29 windows of 4 s (a window every 2 s), so the "per minute" figure
is the model time for 29 windows. The 2-thread CPU run is a rough stand-in for a 2-vCPU host
such as Hugging Face's free tier; real shared vCPUs are usually slower than desktop cores, so
treat it as a best case. Loading time and Python imports are reported separately.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from signals import parity_signals

from squeaktest.config import Settings
from squeaktest.model import load_detector

WINDOWS_PER_MINUTE = 29


def time_minute(detector, windows: list[np.ndarray]) -> float:
    detector(windows[:2])  # warm-up: first calls pay one-time setup costs
    if detector.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    detector(windows)
    if detector.device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start


def main() -> None:
    data_dir = Settings.from_env().data_dir
    window = parity_signals()["buzz_150hz_am_4_5s"][: 4 * 16_000]
    windows = [window] * WINDOWS_PER_MINUTE
    print(f"torch {torch.__version__}, {torch.get_num_threads()} CPU threads available\n")
    print(f"{'setting':28} {'load':>7} {'per window':>11} {'per audio minute':>17}")

    runs: list[tuple[str, str, int | None]] = []
    if torch.cuda.is_available():
        runs.append((f"GPU ({torch.cuda.get_device_name(0)})", "cuda", None))
    runs += [("CPU, all threads", "cpu", torch.get_num_threads()), ("CPU, 2 threads", "cpu", 2)]

    for label, device, threads in runs:
        if threads is not None:
            torch.set_num_threads(threads)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        detector = load_detector(data_dir=data_dir, device=device)
        load_s = time.perf_counter() - start
        minute_s = time_minute(detector, windows)
        extra = ""
        if device == "cuda":
            extra = f"   peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
        print(
            f"{label:28} {load_s:6.1f}s {minute_s / WINDOWS_PER_MINUTE * 1000:9.0f}ms"
            f" {minute_s:15.1f}s{extra}"
        )
        del detector


if __name__ == "__main__":
    main()
