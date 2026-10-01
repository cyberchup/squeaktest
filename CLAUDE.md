# squeaktest

> Real curds squeak. Fakes don't.

squeaktest is a deepfake voice detector. A user uploads an audio file and gets back an estimate of how likely the speech is to be synthetic (AI-generated or voice-cloned), along with where in the clip the suspicious segments are.

The name is a Wisconsin nod. Fresh cheese curds squeak against your teeth and stale ones don't, and the "squeak test" is how you tell. Sibling project: **promptbadger** (prompt injection detector).

## Purpose and audience

- This is a portfolio project. It should show detection-engineering thinking to security employers: honest evaluation, calibrated output, documented limitations, and secure handling of sensitive input.
- It should run locally (Docker or CLI) and be hosted publicly on Hugging Face Spaces, and later on squeaktest.com.
- Credibility matters more than flashy claims. Never overstate accuracy.

## Core principles

1. **Probabilistic output, never a verdict.** Show a score with bands: *Likely synthetic*, *Uncertain*, *Likely genuine*. Every result carries a short disclaimer that the score is an estimate and must not be the sole basis for any decision.
2. **Generalization is the real problem.** Detectors trained on lab datasets often fail on unseen generators and on real-world audio (phone, compression, noise). Evaluate on out-of-domain data and publish the results, including the bad ones.
3. **Voice is biometric data.** Treat every upload as sensitive (see Security & Privacy).
4. **Explainability over black box.** Where feasible, show per-segment scores on a timeline, and possibly a spectrogram view.

## Security and privacy requirements (non-negotiable)

- **No persistence.** Process uploads in memory or in temp files that are deleted in a `finally` block. Never write uploads to permanent storage.
- **No content logging.** Log request metadata only (timestamp, duration, size, result band). Never log audio, filenames, or raw scores tied to an identifier.
- **Input limits.** Set a maximum file size (default 25 MB) and a maximum duration (default 5 min). Both must be configurable.
- **Validate by content, not extension.** Sniff the container and codec and reject anything else. Allowed formats: wav, mp3, flac, ogg, m4a.
- **Decoder hardening.** Audio decoders (ffmpeg and others) are attack surface. Run decoding with timeouts, run the container as non-root with a read-only filesystem where possible, and keep ffmpeg patched.
- **Web hygiene.** Use rate limiting, security headers (CSP, etc.), no third-party analytics or trackers, and keep CORS locked down.
- **Supply chain.** Pin dependencies and run `pip-audit` and `bandit` in CI. Record the source, version, and license of every model weight.
- The README gets a plain-language "How your audio is handled" section.

## Stack (proposed; confirm in Phase 0)

- Python 3.12, managed with `uv`
- PyTorch with audio loading via `soundfile` or `torchaudio`, resampled to 16 kHz mono
- Backend: FastAPI. UI: Gradio for HF Spaces, which can be mounted on FastAPI.
- Tooling: `ruff` for lint and format, `pytest`, type hints throughout
- Packaging: `pyproject.toml` with a CLI entry point `squeaktest`
- Docker for local use and self-hosting

## Proposed repo layout

```
src/squeaktest/
  audio.py        # load, validate, resample, chunk
  model.py        # model loading / wrapper (swappable backends)
  inference.py    # per-segment scoring + aggregation + calibration
  cli.py          # `squeaktest analyze <file>`
  api.py          # FastAPI app
  ui.py           # Gradio interface
eval/             # evaluation harness, metrics, results
tests/
docs/             # research notes, model card, threat model
Dockerfile
pyproject.toml
README.md
```

## Roadmap

**Phase 0: Research spike (do this first, before writing app code).**
This field moves fast, so do not rely on memory. Research the current landscape and write `docs/research.md` covering:
- current pretrained anti-spoofing / deepfake speech detectors (e.g. SSL front-end + classifier approaches), with their licenses and reported out-of-domain results
- current datasets: ASVspoof editions, In-the-Wild, and newer multi-generator sets, with their licenses and download sizes
- a recommendation for the initial model and evaluation set

Then stop and get Dylan's sign-off. Also confirm the project license (MIT vs Apache-2.0).

**Phase 1: CLI inference.**
`squeaktest analyze clip.wav` loads and validates the file, chunks it into windows, scores each window, and prints an aggregate score, a band, and per-segment scores. Include JSON output.

**Phase 2: Evaluation harness.**
Report EER, ROC-AUC, and calibration (reliability diagram) on an in-domain set and at least one out-of-domain set. Write `docs/model-card.md` and put a results table in the README.

**Phase 3: Web app plus Docker.**
FastAPI with a Gradio UI for upload, score, band, and segment timeline. Every security requirement above must be implemented and tested. Add a `docker compose up` local run.

**Phase 4: Publish.**
Set up the GitHub repo (README, badges, CI) and an HF Spaces deployment, then point squeaktest.com at it.

**Phase 5: Stretch goals.**
Spectrogram and attention visualization, an ensemble of models, a robustness test suite (codecs, noise, telephone band), and `docs/threat-model.md` covering adversarial evasion.

## Working agreements

- Ask before adding heavy dependencies, before downloading anything larger than 1 GB, or before changing the stack.
- **Never commit real people's voice recordings.** Test fixtures must be synthetic tones or noise, or come from properly licensed datasets fetched by script.
- Keep the README current as features land. Use small, focused commits with conventional commit messages.
- Mark untested or unverified claims as such in docs.
