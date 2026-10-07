# Design decisions

Each entry records what was chosen, why it beat the alternatives, what it does to detection
quality, and when to revisit it. Numbers marked *toy* come from the synthetic-tone tests in
`tests/test_inference.py`, not from real speech. D1-D15 were set before any real data was
scored. D16 onward come from Phase 2, measured on In-the-Wild: dev speakers for choices, test
speakers for reported numbers (see `eval/results/`).

**Phase 2 revisit status:**

| Entry | Planned check | Phase 2 outcome |
|---|---|---|
| D1 | Windows skipped under phone codecs | Not done; carried forward |
| D2 | Activity bar 0.25 / 0.5 / 0.75 | Not done; carried forward |
| D3 | 4 s versus 8 s windows | Not done; carried forward |
| D5 | Accuracy against amount of speech | Done: 1 s kept (see D5) |
| D6 | Mean, max and top-k | Done: top-3 kept (see D6) |
| D9 | Calibration | Done: not calibrated (D16) |
| D10 | Per-window versus whole-clip normalization | Not done; carried forward |
| D14 | Data-driven bands | Done: replaced by D18 |
| K1 | Tones and music | Not measured; still open |

## D1. Energy-based voice activity detection at -45 dBFS (step 3, `vad.py`)

**Decision:** a 25 ms frame counts as active if its RMS level is above -45 dBFS.

**Why:** the goal is narrow: find silence, so it can be trimmed and skipped (research
section 1, shortcut learning). An energy threshold does that with no model, no dependency and
no extra attack surface, and every value it produces can be checked by hand. A model-based VAD
(such as Silero) can tell speech from music or noise, but that isn't needed to remove silence.

**Impact:**
- If the threshold is too high (say -30 dBFS), quiet talkers or low-gain phone recordings count
  as silence. Their windows go unscored, and a clip can end up with no score at all.
- If it's too low (say -60 dBFS), line hiss counts as activity, so silence reaches the model,
  which is the shortcut this step exists to prevent.
- As a rough rule of thumb (unverified on our data), speech sits around -30 to -15 dBFS and a
  quiet line's noise floor around -60, so -45 sits between them.
- Music and noise count as "active". That's acceptable here, because the model scores them;
  it just means this step doesn't filter them.

**Revisit:** in Phase 2, check the share of windows skipped on In-the-Wild and on the
phone-codec versions. If recording levels vary a lot, switch to a threshold relative to each
clip's loudest frames.

## D2. Trim leading and trailing silence; skip windows under 50% activity (step 4)

**Decision:** windowing starts at the first active frame and ends at the last. A window is
scored only if at least half of its frames are active.

**Why:** in ASVspoof 2019, a model can reach 15% EER from the length of leading silence alone,
and Deepfake-Eval-2024 found that silence cuts audio detectors' accuracy sharply. Trimming
removes the silence a model could key on, and skipping keeps mostly-silent windows (long pauses,
hold music gaps) from producing scores at all.

**Impact:** natural pauses inside speech are kept, because removing them would distort timing.
A window that's 40% speech goes unscored, so a fake hidden in a stretch of sparse speech could
be missed. Lowering the bar to 25% scores more of those windows, but lets more silence in.

**Revisit:** in Phase 2, compare 0.25, 0.5 and 0.75 on dev data, and check that scores don't
correlate with the amount of silence in a clip.

## D3. 4-second windows every 2 seconds (step 4)

**Decision:** `window_s=4.0`, `hop_s=2.0` (50% overlap).

**Why:**
- Window length is a trade-off between evidence and localization. A longer window gives the
  model more speech to judge, and so steadier scores, but it blurs a short fake into the real
  speech around it. A shorter window localizes better, but each score is noisier.
- About 4 seconds is a common input length for anti-spoofing models (AASIST-family models use
  about 4 s), and it gives a 2-second timeline resolution, enough to point an analyst at the
  right part of a voicemail.
- NII reports AntiDeepfake doing better with longer inputs (for XLS-R-2B fine-tuned on
  Deepfake-Eval-2024, 9.68% EER at 50 s versus 12.14% at 4 s), so longer windows may be worth
  their blur.
- 50% overlap means every moment of audio lands in two windows, so a fake that straddles a
  window boundary isn't cut in half and missed.

**Impact:**
- Cost doubles: a 60-second clip produces 29 windows, about 116 seconds of audio for the model.
  On the free CPU tier, that matters (research section 4.2).
- Fakes much shorter than 2 seconds are diluted inside their windows.

**Revisit:** in step 5, check the input length AntiDeepfake was trained on. In Phase 2, compare
4 s and 8 s windows on dev data. For hosting, consider `hop_s=window_s` to halve the cost.

## D4. An extra window aligned to the end of the clip (step 4)

**Decision:** if the regular grid of windows stops short of the end of the speech, add one more
window that ends exactly at the end.

**Why:** without it, up to `hop_s` seconds at the end of every clip would never be scored, which
is a coverage gap: anything placed there would be invisible to the detector.

**Impact:** at most one extra window per clip. It overlaps its neighbor by more than usual, so
the final seconds count slightly more in the aggregate.

## D5. No score with less than 1 second of speech (step 4)

**Decision:** clips with less than `min_speech_s=1.0` of detected activity get no score, with a
note saying why. The same happens if every window is mostly silent.

**Why:** a score computed from half a second of audio would look as confident as any other.
For a SOC, "not enough speech to assess" is honest and actionable; a coin-flip number dressed
up as a probability isn't.

**Impact:** very short voicemails get no score. The output, and later the SIEM event, must
report that state explicitly, so it's never mistaken for "likely genuine".

**Phase 2 result (W2V-Large, dev speakers, measured):**

| Minimum | EER | Genuine not assessed | Synthetic not assessed |
|---|---|---|---|
| 0 s | 1.5% | 0.1% | 0.2% |
| 0.5 s | 1.4% | 1.6% | 1.4% |
| 1 s | 0.9% | 10.2% | 9.2% |
| 1.5 s | 0.6% | 20.6% | 19.6% |
| 2 s | 0.4% | 31.5% | 31.2% |

Kept at 1 s (Dylan, 2026-10-07). Going from 0.5 s to 1 s cuts EER by about a third, at the
cost of about 9% more clips unscored. Each further half second buys less accuracy and leaves
about 10% more clips unscored. In-the-Wild is full of very short clips, so on the test speakers
16.0% of genuine and 11.3% of fake clips get *Not assessed*. Voicemails are usually longer.
Known gap K3 covers the short clips that do get scored.

## D6. Clip score = mean of the top 3 window scores (step 4)

**Decision:** `aggregate(scores, top_k=3)`.

**The options, with *toy* numbers:**

| Clip | Mean of all windows | Top-3 mean | Max |
|---|---|---|---|
| 4 s fake spliced into 24 s of real speech | 0.18 (missed) | 0.67 (caught) | 1.00 (caught) |
| Real call with one spiking window (a cough, a glitch) | low | 0.36 (not flagged) | 0.97 (false alarm) |

**Why:**
- The mean dilutes a short fake into the real speech around it. A cloned sentence ("approve the
  transfer") spliced into a real call is exactly the case a SOC most needs to catch.
- The max catches that, but a single bad window decides the whole clip. Its false-positive rate
  also grows with clip length: if each window has a 1% chance of a false spike, a 60-second clip
  with 29 windows has a 1 − 0.99²⁹ ≈ 25% chance of at least one.
- Top-3 sits between them. With 50% overlap, every moment lands in two windows, so a
  three-window requirement means the suspicion has to persist for more than a moment, roughly 4
  seconds of audio, before it moves the clip score much.

**Impact:**
- For fully synthetic clips (the typical deepfake voicemail), all three methods agree, so this
  choice mainly affects partial fakes and long real calls.
- Fakes shorter than about 4 seconds are partly diluted.
- Top-3 still rises somewhat with clip length (more windows give more chances for high scores),
  so Phase 2 calibration must check scores against clip length.

**Phase 2 result (W2V-Large, test speakers, measured):** mean, top-3 and max all give 1.3-1.4%
EER and catch 98.3-98.4% of fakes at 1% false positives. As predicted, they agree on
In-the-Wild, where every fake is a whole clip. Top-3 stays for the partial-fake case, which is
still untested.

**Revisit:** test on a partial-spoof dataset such as PartialSpoof (license unverified).

## D7. Check the scorer's output instead of trusting it (step 4)

**Decision:** `analyze` raises an error if the scorer returns the wrong number of scores, NaN, or
values outside [0, 1].

**Why:** a model wrapper bug, such as a shape mix-up or a NaN from bad input, would otherwise
become a normal-looking clip score. Failing loudly beats reporting a wrong number with
confidence.

**Impact:** a broken model produces an error instead of a result. That is the intended
behavior.

## D8. Rebuild NII's model in `transformers` instead of running fairseq (step 5, `model.py`)

**Decision:** recreate the AntiDeepfake network with `transformers`' `Wav2Vec2Model` plus mean
pooling and `Linear(1024, 2)`, and rename NII's fairseq-layout weights to fit it.

**Why:** NII's official code needs fairseq, which is unmaintained, doesn't install on Python
3.12, and pins old libraries (its config library only installs with pip older than 24.1).
Depending on it would freeze the whole project on old, unpatched software. The network is a
standard wav2vec 2.0 model that `transformers` implements, so the same computation can run on
current, maintained libraries.

**Evidence (measured):** five deterministic synthetic signals were run through NII's original
code (Python 3.9, fairseq 0.12.2, PyTorch 2.8, weights loaded with strict key matching) and
through the port. On CPU, the largest logit difference was 3.8e-5, and P(fake) matched to five
decimal places on every signal. `tests/test_model.py` repeats the check whenever the weights
are present, against values stored in `tests/data/`.

**Impact:** no fairseq anywhere in squeaktest. The cost is that the port is ours to maintain:
a future `transformers` release could change its internals. The parity test catches that
locally, but CI can't run it, because CI doesn't download 1.27 GB of weights.

**Revisit:** when adding the other shortlisted models in Phase 2, each needs its own parity
check before its numbers count.

## D9. The window score is P(fake) from NII's softmax, labeled uncalibrated (step 5)

**Decision:** score = softmax(logits)[0], NII's "fake" probability.

**Why:** it keeps NII's definition, so their published results and ours describe the same
quantity.

**Impact:** this number is not yet a trustworthy probability. NII's own model card shows the
EER threshold for this model ranging from 0.62 on In-the-Wild to 0.99 on Deepfake-Eval-2024: the
same raw score means very different things in different domains. Until Phase 2 calibrates it,
output must say "uncalibrated" and the bands must not be read as probabilities.

**Phase 2 outcome:** the score stays uncalibrated (D16), and the output says it is not a
probability.

## D10. Normalize each window on its own (step 5)

**Decision:** each window is scaled to zero mean and unit variance before the model sees it.

**Why:** NII normalizes each input it scores, and NII scores whole utterances. squeaktest's
model inputs are windows, so normalizing per window is the faithful equivalent.

**Impact:** loudness differences between windows don't leak into scores: a quiet window isn't
judged differently because a loud one sits next to it.

**Revisit:** Phase 2 can compare per-window with whole-clip normalization on dev data.

## D11. Batch windows of equal length; never pad (step 5)

**Decision:** windows with the same number of samples run as one batch. Different lengths run
separately.

**Why:** padding short windows with zeros would put silence into the time average the
classifier sees, which reintroduces the silence shortcut (D2) and changes the score.

**Impact:** almost every window is exactly 4 s, so nearly all run in one batch. At most one or
two odd-length windows (the end-aligned one, short clips) run alone. A test checks that batched
and one-at-a-time scores agree.

## D12. Full float32 on GPU: TF32 convolutions off (step 5)

**Decision:** `load_detector` sets `torch.backends.cudnn.allow_tf32 = False` on CUDA.

**Why (measured on the RTX 3050):** with cuDNN's default TF32 convolutions, GPU logits drifted
up to 2.1e-3 from NII's reference; with full float32 they matched within 7.6e-5. Speed was the
same: 38.7 ms versus 38.5 ms per 4-second window.

**Impact:** the same clip gets the same score on CPU and GPU, which keeps Phase 2 results
reproducible. It's a process-wide PyTorch setting, which is fine for squeaktest.

## D13. Pinned weights, checked by SHA-256, loaded only from safetensors (step 5)

**Decision:** each model is pinned to a Hugging Face commit, its file's SHA-256 is checked
before every load, and weights load only from `.safetensors`.

**Why:** supply chain. A repository can change after you depend on it, and a pinned commit
can't. The checksum catches corruption and tampering. And safetensors files contain only
numbers: loading them can't run code, unlike PyTorch's pickle-based `.pt`, `.bin` and `.pth`
files. That matters later, because TCM ships a `.pth` file; it will need `weights_only=True`
loading or a one-time conversion.

**Impact:** about 3 seconds of hashing per load for the 1.27 GB file.

## D14. Provisional bands: likely synthetic at 0.90 and above, likely genuine below 0.50 (step 7)

**Status:** replaced by D18 in Phase 2.

**Decision:** `band_for` maps the raw clip score to *Likely synthetic* (0.90 and above),
*Uncertain* (0.50 to 0.90) or *Likely genuine* (below 0.50). No score maps to *Not assessed*.

**Why:** these bands sit on an uncalibrated score (D9), so no threshold can be justified with
data yet. NII's model card puts this model's EER threshold anywhere from 0.62 (In-the-Wild) to
0.99 (Deepfake-Eval-2024). On some real-world audio, then, genuine clips score up to 0.99. A
high bar for "likely synthetic" and a wide uncertain middle state that honestly: most of the
range doesn't support a confident call yet.

**Impact:** some real deepfakes will land in *Uncertain* (missed as alerts), and some genuine
recordings, especially compressed or non-English ones, can still reach *Likely synthetic*.
Every output says the bands are provisional.

**Revisit:** Phase 2 replaces both thresholds with values chosen on the In-the-Wild dev split
at a target false-positive rate, after calibration.

## D15. Accept 15-second CLI startup; cap batches at 8 windows (step 7)

**Measured on Dylan's machine** (i5-11600K, RTX 3050, one minute of audio = 29 windows):

| Setting | Per 4 s window | Per audio minute |
|---|---|---|
| GPU | 40 ms | 1.2 s (peak GPU memory 1.9 GB) |
| CPU, 6 cores | 492 ms | 14.3 s |
| CPU, 2 threads (free-tier stand-in) | 854 ms | 24.8 s |

Loading the model takes 2-3 s. Importing PyTorch and `transformers` takes about 15 s on
Windows, most of it `transformers` loading modules squeaktest never uses.

**Decisions:**
- **Keep `transformers` despite the startup cost.** A long-running web app pays it once. The
  alternative, our own wav2vec 2.0 code, would remove the cost but leave us maintaining the
  architecture.
- **Cap batches at 8 windows.** The first convolution layer alone needs about 26 MB per window,
  so one batch for a 5-minute clip would need about 4 GB for that layer.
- **Build the model on PyTorch's meta device and verify the exact bytes that are loaded.**
  Random initialization was taking 20 s, and loading the bytes that were verified closes the
  gap between checking the file and reading it.

**Impact on hosting:** on the free 2-vCPU tier, expect at least 25 s per minute of audio,
probably more on shared CPUs. A 60-second cap on the hosted demo keeps waits under about a
minute; free ZeroGPU would bring it near 1 s (research section 4.2).

**Revisit:** startup time, if hosting moves to something that starts per request.

## D16. Report the raw score; don't calibrate it (Phase 2)

**Decision:** squeaktest reports the model's raw P(fake) (D9) without calibration. The output
says the score is not a probability, and the bands (D18) carry the meaning.

**Background:** a calibrated score means what it says: of all clips scored 0.8, about 80% are
fake. Platt scaling is the standard fix. It fits a two-parameter curve that maps raw scores to
observed frequencies, on data held apart from the test set. Because the curve only stretches the
scores and never reorders them, it changes no detection metric (EER, AUC, detection rate at a
fixed false-positive rate). It only changes what the number claims. Source: scikit-learn's
[probability calibration guide](https://scikit-learn.org/stable/modules/calibration.html).

**Evidence (measured; Platt fitted on dev speakers, measured on test speakers; lower is better):**

| Model | ECE raw | ECE after Platt | Brier raw | Brier after Platt |
|---|---|---|---|---|
| W2V-Large | 0.004 | 0.009 | 0.010 | 0.011 |
| MMS-300M | 0.004 | 0.010 | 0.014 | 0.015 |
| W2V-Small | 0.018 | 0.012 | 0.025 | 0.023 |

**Why:**
- For the two large models, the raw score was already close to calibrated on In-the-Wild, and
  Platt made it worse. There are only 11 dev speakers, so the fit learns their quirks rather
  than a general rule (overfitting at the speaker level). W2V-Small improved, but it isn't the
  default (D17).
- Calibration only holds for one domain at one share of fakes. The dev set is 42% fake, but a
  SOC queue is almost all genuine. Shifted to a 1-in-1,000 base rate, a "0.9" calibrated on this
  dev set corresponds to about 1.2% (`prior_shift_example` in each `results.json`). An analyst
  who reads 0.9 as "90% likely fake" would be misled even by a well-calibrated score.

**Impact:** no change in detection quality, since calibration never reorders scores. The cost
is that the number can't be read as a probability, so the output says so, and the README shows
what alerts mean at realistic base rates.

**Revisit:** with a larger, more varied dev set (such as Deepfake-Eval-2024 as dev data), or
when the SIEM event (Phase 5) wants a probability at a configured base rate
(`calibration.shift_prior` already implements the shift).

## D17. Default model: AntiDeepfake Wav2Vec2-Large (Phase 2 bake-off)

**Decision:** the default model changes from MMS-300M to W2V-Large (Dylan, 2026-10-07).

**Evidence (measured, In-the-Wild test speakers; full tables and intervals in
`eval/results/bakeoff.md`):**

| | W2V-Large | MMS-300M | W2V-Small |
|---|---|---|---|
| EER, clean | 1.3% | 1.9% | 3.2% |
| EER, G.711 / AMR-NB / Opus / MP3 | 2.3 / 4.3 / 6.1 / 2.2% | 2.5 / 4.5 / 9.9 / 3.2% | 4.5 / 5.7 / 6.1 / 4.4% |
| Fakes caught at 1% FPR, clean | 98.3% | 96.8% | 92.3% |
| Fakes caught at 1% FPR, Opus 12k | 73.7% | 6.2% | 76.9% |
| EER, ASVspoof 2019 LA (seen in training) | 1.1% | 1.9% | 4.9% |
| Weights | 1.27 GB | 1.27 GB | 0.38 GB |

Paired speaker bootstrap, MMS-300M minus W2V-Large: clean EER +0.6 points (95% interval 0.0
to +1.0, borderline); Opus EER +3.8 points (+1.1 to +5.7), and 67.5 points fewer fakes caught
at 1% FPR (-73.8 to -20.1). Both Opus differences exclude zero.

**Why:**
- W2V-Large is best or tied on every English condition. Its largest margin is on Opus, the
  worst condition for every model and common in voice notes and VoIP calls, and that margin is
  well outside the noise.
- It is the same size as MMS-300M (24 layers, 1024 wide), and costs the same to run. Measured
  with `scripts/benchmark.py` on the D15 machine: 1.1 s per audio minute on GPU (peak memory
  1.9 GB), 14.6 s on 6 CPU threads and 28.1 s on 2 (MMS-300M: 1.2, 14.3 and 24.8 s).
- W2V-Small matches it on Opus but is clearly worse everywhere else (significantly worse EER on
  clean, G.711 and MP3). Its one advantage is size, which could matter on the free CPU tier.

**Impact:**
- Fewer missed fakes and fewer false alarms on English audio in every tested condition.
- **Unmeasured on other languages.** Every evaluation clip is English. MMS-300M's front-end was
  pre-trained on over 1,400 languages, and NII reports it ahead of W2V-Large on Chinese (ADD2023
  EER 7.9% versus 13.3%) and on DeepVoice (2.35% versus 4.44%). For non-English calls,
  W2V-Large may be the weaker choice. The model card says so.
- Bands are now per model (D18). MMS-300M and W2V-Small keep their code and parity tests for
  evaluation, but have no bands, so the CLI won't report with them.

**Revisit:** when a non-English held-out set is available, or if hosted CPU latency is too slow
(then measure W2V-Small for the demo).

## D18. Bands chosen on dev speakers under clean audio and all four codecs (Phase 2)

**Decision:** for W2V-Large, *Likely synthetic* at raw score 0.9515 and above, *Likely genuine*
below 0.4856, *Uncertain* in between (Dylan, 2026-10-07). `eval/analyze.py` chooses them on the
11 dev speakers' clips, pooled with equal weight across clean audio and the G.711, AMR-NB, Opus
and MP3 re-encodings (21,682 clips). The rule is symmetric: each "likely" band may hold at most
1% of the other class in that pool. The thresholds live on the model's spec, because a raw
score means something different for every model, and a test checks them against
`results.json`.

**Why:**
- A threshold only holds for the kind of audio it was chosen on (the same lesson as D9 and
  D16). Chosen on clean dev audio alone, both thresholds landed at 0.658, so there was no
  *Uncertain* band at all, and 15.8% of genuine Opus test clips landed in *Likely synthetic*.
- A SOC mostly receives phone calls and compressed voice notes, so the bands are chosen on audio
  like that. Re-encoding the dev speakers keeps the test speakers untouched.
- Equal weight per condition rather than a worst-case rule ("at most 1% on every codec"). On dev
  data, the worst-case rule would put *Likely synthetic* at 0.993, which is set entirely by Opus,
  and would cut the share of dev fakes reaching *Likely synthetic* from 97.3% to 93.1% on clean
  audio and from 86.8% to 73.9% on AMR-NB.
- The 1% rule matches the alert framing: *Likely synthetic* is an alert with a known
  false-positive budget, and *Likely genuine* is a clearance with a known miss budget.

**Impact (measured on test speakers; each cell shows clean-only thresholds → these thresholds):**

| Test audio | Genuine rated *Likely synthetic* | Fakes rated *Likely synthetic* | Fakes rated *Likely genuine* |
|---|---|---|---|
| Clean | 0.6% → 0.1% | 97.7% → 94.2% | 2.3% → 1.8% |
| G.711 | 0.6% → 0.1% | 95.2% → 90.4% | 4.8% → 3.5% |
| AMR-NB | 4.4% → 0.3% | 95.8% → 86.7% | 4.2% → 3.3% |
| Opus 12k | 15.8% → 8.6% | 98.1% → 95.7% | 1.9% → 1.5% |
| MP3 32k | 0.9% → 0.1% | 95.0% → 89.8% | 5.0% → 3.6% |

- False alarms drop in every condition, and fewer fakes reach *Likely genuine*. The cost is that
  2-9% of fakes move from *Likely synthetic* into *Uncertain*. They are still surfaced, just not
  as an alert.
- **The dev set understated Opus.** On dev speakers, these thresholds put 3.4% of genuine Opus
  clips in *Likely synthetic*; on test speakers it's 8.6%. How much Opus hurts varies a lot from
  voice to voice, and 11 dev speakers didn't capture that spread. Known gap K2 carries the test
  figure.
- At a 1-in-1,000 base rate, about 58% of clean-audio alerts would be real fakes (speaker
  bootstrap interval 36-100%; `eval/results/antideepfake-w2v-large/report.md`). On Opus, about 1%
  would be.

**Revisit:** with more dev speakers, or real phone recordings for dev data (such as ASVspoof 5's
codec conditions). The thresholds have to be re-chosen whenever the model changes.

## D19. Warn when the input is Opus (Phase 2)

**Decision:** when the probe step identifies the codec as Opus, the report carries a warning
(`warnings` in the JSON, code `opus_audio`) quoting the measured false-alarm rate (Dylan,
2026-10-07).

**Why:** Opus at 12 kbit/s is the weakest tested condition even after D18. 8.6% of genuine
test clips land in *Likely synthetic*, so at realistic base rates most Opus alerts are false
alarms. The codec is known exactly from the probe step, so the warning costs nothing and never
fires by guesswork. The analyst sees the caveat next to the result it applies to, instead of in
a README.

**Impact:**
- Covers only Opus in the file itself. Audio that went through Opus earlier (a voice note
  re-saved as M4A, a VoIP call recorded to WAV) gets no warning. That case is documented in K2.
- The numbers were measured at 12 kbit/s. Higher bitrates are probably less damaging but are
  untested, and the warning says so.
- A test fails if the quoted numbers drift from `results.json`.

**Revisit:** when Opus is measured at other bitrates, or if another weak condition shows up in
a format squeaktest accepts.

## Known gaps

### K1. Non-speech audio gets confident "fake" scores (found in step 5)

**Finding (measured):** on the synthetic parity signals, the real model gave a 440 Hz tone
P(fake) = 0.999 and a harmonic buzz 0.96, while noise got 0.01 and a frequency sweep 0.44. The
model only learned about speech, so anything else produces confident-looking but meaningless
scores.

**Why it matters for a SOC:** voicemails contain system beeps, DTMF tones, ringback and hold
music. The energy VAD (D1) treats all of these as activity, so they get scored. Top-3
aggregation (D6) absorbs a single short beep, but not a long stretch of music. The likely result
is false positives on real voicemails.

**Options:** a speech-versus-non-speech check before scoring (a model-based VAD, or a simple
tonality or music detector), or excluding windows that look tonal.

**Status:** open. Not measured in Phase 2 (In-the-Wild has no tones or music to test with). A
synthetic 300 Hz tone sent through the CLI after Phase 2 still scored 1.00, *Likely
synthetic*. Fix it before any claim that squeaktest is ready for SOC use.

### K2. Opus voice notes and VoIP audio cause false alarms (found in Phase 2)

**Finding (measured, In-the-Wild test speakers re-encoded as Opus at 12 kbit/s):**

| Model | EER (same clips clean) | Fakes caught at 1% FPR | Genuine rated *Likely synthetic* |
|---|---|---|---|
| W2V-Large (default, D18 bands) | 6.1% (1.6%) | 73.7% | 8.6% |
| MMS-300M (clean-dev bands) | 9.9% (2.5%) | 6.2% | 30.5% |
| W2V-Small (clean-dev bands) | 6.1% (3.7%) | 76.9% | 10.1% |

Opus pushes genuine scores up for every model. The model ranking still separates real from fake
reasonably well (AUC 0.984 for W2V-Large), but genuine clips move into the range where fakes
sit. If 1 recording in 100 were fake, about 90% of W2V-Large's *Likely synthetic* results on
Opus audio would be false alarms.

**Why it matters for a SOC:** WhatsApp and Telegram voice notes use Opus, as do WebRTC-based
calls (Opus is mandatory in WebRTC). That is a large share of what an analyst would check.

**Mitigations in place:** D18 thresholds (genuine false alarms on Opus 15.8% → 8.6%) and the D19
warning. **Not covered:** audio that was Opus earlier and was converted before it reached
squeaktest, and bitrates other than 12 kbit/s.

**Status:** open. The real fix is training the model on Opus-compressed audio (codec
augmentation), which is a decision for the Phase 2 gate.

### K3. Very short clips are less reliable (found in Phase 2)

**Finding (measured, W2V-Large, clean test speakers):** EER is 5.1% for clips under 2 seconds,
against 1.1% for 2-4 seconds, 0.4% for 4-8 seconds and 1.7% for 8 seconds and up (why longer
clips do worse again is unexplained). Among genuine clips, shorter clips score
higher (Spearman correlation between score and duration: -0.45), so short genuine clips are
the ones most likely to be flagged.

**Also checked:** duration alone, with no model, separates In-the-Wild's classes with AUC 0.62,
because the dataset's fakes tend to be longer. The model's AUC is at least 0.985 in every
duration bucket, so it isn't just reading length.

**Status:** open. Options: a warning on clips under 2 seconds, or a higher minimum (D5 shows the
coverage cost).
