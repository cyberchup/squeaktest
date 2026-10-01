"""Command-line interface: `squeaktest analyze FILE` and `squeaktest fetch-model`.

Exit codes:
  0  analysis finished (any band, including "not assessed")
  1  the input was rejected (unsupported, too large, too long, undecodable), or a download
     was declined
  2  usage or setup problem (bad arguments, ffmpeg missing, model not downloaded, no GPU)

Messages never include the input's filename, so output is safe to log.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from squeaktest import __version__
from squeaktest.audio import AudioError, FFmpegNotFoundError, load_audio_file
from squeaktest.config import Settings
from squeaktest.inference import WindowingConfig, analyze
from squeaktest.report import build_report, format_report

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_SETUP = 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="squeaktest",
        description="Estimate whether speech in an audio file is synthetic (deepfake).",
    )
    parser.add_argument("--version", action="version", version=f"squeaktest {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    analyze_cmd = commands.add_parser("analyze", help="score an audio file")
    analyze_cmd.add_argument("file", type=Path, help="wav, mp3, flac, ogg or m4a")
    analyze_cmd.add_argument("--json", action="store_true", help="print the report as JSON")
    analyze_cmd.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto", help="default: auto"
    )

    fetch_cmd = commands.add_parser("fetch-model", help="download the detection model")
    fetch_cmd.add_argument("--yes", action="store_true", help="don't ask for confirmation")

    args = parser.parse_args(argv)
    settings = Settings.from_env()
    if args.command == "analyze":
        return _analyze(args.file, settings, as_json=args.json, device=args.device)
    return _fetch_model(settings, assume_yes=args.yes)


def _analyze(file: Path, settings: Settings, *, as_json: bool, device: str) -> int:
    # Validate the input before loading the model: rejecting a bad file is instant, while
    # loading the model takes seconds.
    try:
        audio = load_audio_file(file, settings)
    except AudioError as exc:
        return _fail(f"input rejected: {exc}", EXIT_REJECTED)
    except FFmpegNotFoundError as exc:
        return _fail(str(exc), EXIT_SETUP)

    # Imported here so that `--help` and input errors don't wait for PyTorch to load.
    import torch

    from squeaktest.model import ModelNotFoundError, load_detector

    if device == "cuda" and not torch.cuda.is_available():
        return _fail("--device cuda was requested, but no CUDA GPU is available", EXIT_SETUP)
    try:
        detector = load_detector(
            data_dir=settings.data_dir, device=None if device == "auto" else device
        )
    except ModelNotFoundError as exc:
        return _fail(f"{exc} Run: squeaktest fetch-model", EXIT_SETUP)

    config = WindowingConfig()
    result = analyze(audio.samples, audio.sample_rate, detector, config)
    report = build_report(result, audio, detector.spec, config)
    print(json.dumps(report, indent=2) if as_json else format_report(report))
    return EXIT_OK


def _fetch_model(settings: Settings, *, assume_yes: bool) -> int:
    from squeaktest.model import DEFAULT_MODEL, MODELS, fetch_weights, weights_path

    spec = MODELS[DEFAULT_MODEL]
    target = weights_path(spec, settings.data_dir)
    print(f"Model:    {spec.name} ({spec.size_bytes / 1e9:.2f} GB)")
    print(f"Source:   {spec.source_url}")
    print(f"License:  {spec.license}, non-commercial use only")
    print(f"Save to:  {target}")
    if not target.exists() and not assume_yes:
        answer = input("Download it and accept the license? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Not downloaded.")
            return EXIT_REJECTED
    path = fetch_weights(spec, settings.data_dir)
    print(f"Ready:    {path} (SHA-256 verified)")
    return EXIT_OK


def _fail(message: str, code: int) -> int:
    print(f"squeaktest: {message}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
