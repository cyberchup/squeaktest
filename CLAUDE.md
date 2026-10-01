# squeaktest

> Real curds squeak. Fakes don't.

squeaktest is a deepfake voice detector. A user uploads an audio file and gets back an estimate of how likely the speech is to be synthetic (AI-generated or voice-cloned), along with where in the clip the suspicious segments are.

The name is a Wisconsin nod. Fresh cheese curds squeak against your teeth and stale ones don't, and the "squeak test" is how you tell. Sibling project: **promptbadger** (prompt injection detector, `../promptbadger`). Match its conventions where they fit: SIEM event output, evaluation discipline, honest README.

Repo: https://github.com/cyberchup/squeaktest

## Purpose and audience

- This is a portfolio project. It should show detection-engineering thinking to security employers: honest evaluation, calibrated output, documented limitations, and secure handling of sensitive input.
- It should run locally (Docker or CLI) and be hosted publicly on Hugging Face Spaces, and later on squeaktest.com.
- Credibility matters more than flashy claims. Never overstate accuracy.

## Commands

```bash
uv sync                           # create .venv from uv.lock
uv run pytest                     # tests; audio tests need ffmpeg and ffprobe on PATH
uv run ruff check                 # lint
uv run ruff format                # format
uv run bandit -r src              # security lint
uv run python scripts/audit_deps.py  # known-vulnerable dependencies, PyTorch included
```

CI (`.github/workflows/ci.yml`) runs all of these on Ubuntu. Development is on Windows.

PyTorch comes from PyTorch's own index: CUDA 13.0 builds on Windows, CPU builds elsewhere (see `[tool.uv.sources]` in pyproject.toml). Model weights download into `SQUEAKTEST_DATA_DIR/models/` (on Dylan's machine, an E: drive folder set as a user environment variable). The parity tests in `tests/test_model.py` run only when the weights are present; `scripts/parity_reference.py` regenerates their reference values from NII's original fairseq code.

## Primary use case: SOC triage

The main user is a SOC analyst, or a helpdesk or finance team escalating to one, checking a suspicious voicemail, voice note or call recording (vishing, executive impersonation, helpdesk social engineering). This drives several choices:

- Phone-quality and compressed audio (8 kHz telephone band, lossy codecs) is the main domain to evaluate, not clean studio speech.
- Output includes a SIEM-ready JSON event, as promptbadger does, so results can feed Microsoft Sentinel or another SIEM.
- Describe operating points in alert terms: detection rate at a fixed false-positive rate, and what that means at realistic base rates.

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

## Stack

Confirmed (Phase 0 sign-off, 2026-09-30):
- Python 3.12, managed with `uv`
- License: MIT for the code (matches promptbadger)
- PyTorch, with models loaded through Hugging Face `transformers`. Weights are ported from the official checkpoints; never depend on fairseq. Audio is resampled to 16 kHz mono.
- Initial model: NII AntiDeepfake MMS-300M. Phase 2 bake-off shortlist: AntiDeepfake MMS-300M, Wav2Vec2-Large and Wav2Vec2-Small, plus TCM. See `docs/research.md`.
- The AntiDeepfake weights are CC BY-NC-SA 4.0, so squeaktest and squeaktest.com stay non-commercial, and any fine-tuned weights inherit that license. State this in the README and model card.
- Tooling: `ruff` for lint and format, `pytest`, type hints throughout
- Packaging: `pyproject.toml` with a CLI entry point `squeaktest`

Proposed (validate in Phase 1):
- Audio decoding with `ffmpeg` in a subprocess (timeout, explicit demuxer, restricted protocols); `soundfile` optional for WAV and FLAC. Not torchaudio, whose I/O is deprecated.

Planned (revisit in Phases 3-4, since free ZeroGPU hosting requires the Gradio SDK rather than Docker):
- Backend: FastAPI. UI: Gradio for HF Spaces, which can be mounted on FastAPI.
- Docker for local use and self-hosting

## Hosting and data

- **Hosting:** Hugging Face Spaces free CPU tier for now. The chosen model must run acceptably on CPU. Keep backends device-agnostic so a paid GPU tier can be added later without a rewrite. The hosted demo may use a lower maximum duration than local runs.
- **Data:** Datasets and model weights live outside the repo, in the directory set by `SQUEAKTEST_DATA_DIR`, and are fetched by script. Never commit them.

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

Each phase ends with a stop for Dylan's review.

**Phase 0: Research spike (done; signed off 2026-09-30, see `docs/research.md` section 6).**
This field moves fast, so do not rely on memory. Research the current landscape and write `docs/research.md` covering:
- current pretrained anti-spoofing / deepfake speech detectors (e.g. SSL front-end + classifier approaches), with their licenses and reported out-of-domain results
- current datasets: ASVspoof editions, In-the-Wild, and newer multi-generator sets, with their licenses and download sizes
- fit for this project: CPU inference cost (free hosting tier), install practicality on Windows and Python 3.12, and relevance to phone audio
- a shortlist of 2-3 models for the Phase 2 bake-off, and a recommendation for the initial model and evaluation sets

Then stop and get Dylan's sign-off.

**Phase 1 (v0.1): CLI inference.**
`squeaktest analyze clip.wav` loads and validates the file, chunks it into windows, scores each window, and prints an aggregate score, a band, and per-segment scores. Include plain JSON output. (The SIEM event was moved to Phase 5 on 2026-10-01.)
- Set up CI (ruff, pytest, pip-audit, bandit) with the first code commit.
- Build the input security into `audio.py` now, since the CLI is an input surface too: content sniffing, size and duration limits, decode timeouts. Test with synthetic fixtures.
- Label scores as uncalibrated until Phase 2 calibrates them.
- Measure CPU and GPU latency and memory per minute of audio.

**Phase 2 (v0.2): Evaluation harness and model bake-off.**
- Evaluate on an in-domain set, at least one out-of-domain set, and compressed versions of them (low-bitrate MP3, Opus, simulated phone line).
- Report the metrics listed under Evaluation discipline.
- Run the shortlisted models through the same harness and pick one with evidence.
- Deepfake-Eval-2024 (gated access) is deferred; revisit later as a second held-out set.
- Fit calibration and band thresholds on development data only.
- Write `docs/model-card.md` and put a results table, with known gaps, in the README.
- Decision gate: if out-of-domain results are poor, Dylan decides whether to ship with documented limits, switch models, or fine-tune.

**Phase 3 (v0.3): Web app plus Docker.**
FastAPI with a Gradio UI for upload, score, band, and segment timeline. Every security requirement above must be implemented and tested: rate limiting, security headers, locked-down CORS, a non-root container with a read-only filesystem. Add a `docker compose up` local run and a first `docs/threat-model.md` for the upload pipeline.

**Phase 4 (v1.0): Publish.**
README badges, an HF Spaces deployment, then point squeaktest.com at it.

**Phase 5: Stretch goals.**
Spectrogram and attention visualization, an ensemble of models, a full robustness suite (noise, reverb and more codecs), the adversarial-evasion section of `docs/threat-model.md`, and SIEM integration as in promptbadger: the SIEM event (proposal and open decisions in `docs/siem-event.md`) plus Microsoft Sentinel content (analytics rule and hunting KQL over those events).

## Evaluation discipline

- Keep development data and held-out data separate. Calibration, band thresholds and any tuning are fit on development data only. Report held-out numbers as-is, including bad ones.
- If a held-out set gets used for tuning, move it to the development list and say so in the README.
- Report EER, ROC-AUC, detection rate at fixed false-positive rates (e.g. 1% and 5%), and calibration (reliability diagram), per dataset and per condition (clean, MP3, Opus, phone line).
- Eval sets are roughly balanced; real traffic is mostly genuine. State the base-rate assumption behind any probability, and show what precision looks like at realistic prevalence.
- Weak spots found in held-out results go under "Known gaps" in the README. Do not patch them against the held-out set.
- Watch for shortcut learning (models keying on silence or recording conditions instead of the voice) and test for it where feasible.

## Working with Dylan

Dylan is a detection engineer (SOC/MSSP background, Sentinel/KQL) and is using this project to learn ML hands-on. Frame choices in detection-engineering terms (fidelity, false-positive rate, tuning, coverage), and:

- Explain the why before the how. Introduce each new ML concept briefly (what it is, why it matters here) with one good source before relying on it.
- Recommend, don't decide. Dylan makes the key calls (model, datasets, metrics, thresholds) after hearing the trade-offs.
- Claude writes the code, including the core ML pieces (decided 2026-10-01). For every ML choice, explain why it was chosen over the alternatives and its impact on detection quality (missed fakes, false positives, cost). Record significant choices in `docs/decisions.md` (decision, why, impact, when to revisit) so Dylan can defend them in interviews.

## Working agreements

- Ask before adding heavy dependencies, before downloading anything larger than 1 GB, or before changing the stack.
- **Never commit real people's voice recordings.** Test fixtures must be synthetic tones or noise, or come from properly licensed datasets fetched by script.
- Keep the README current as features land. Use small, focused commits with conventional commit messages.
- Mark untested or unverified claims as such in docs.
