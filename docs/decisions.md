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
| K1 | Tones and music | Measured after Phase 2; tones fixed by D31, music open |

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

## D20. Ship W2V-Large as it is, with its limits documented (Phase 2 gate)

**Decision:** at the end of Phase 2, squeaktest keeps NII's W2V-Large weights unchanged, with
the D18 bands, the D19 warning and known gaps K1-K3 published. It doesn't switch models or
fine-tune (Dylan, 2026-10-13).

**The options:**
- **Ship with documented limits:** keep the measured model and publish where it fails.
- **Switch models:** pick a different detector.
- **Fine-tune:** continue training W2V-Large's weights on new labeled examples. For K2, that
  means codec augmentation: re-encoding the training clips through Opus and other codecs, so the
  model learns that codec damage isn't evidence of a fake.

**Why:**
- **The held-out results are good enough to ship.** On test speakers the model never saw, it
  has 1.3% EER on clean audio, catches 98.3% of fakes at 1% false positives, and flags at most
  0.3% of genuine clips as *Likely synthetic* under G.711, AMR-NB and MP3 (D18).
- **Switching doesn't remove the weak spot.** Every measured model is worst on Opus (K2). The
  one that matches W2V-Large there, W2V-Small, is significantly worse on clean, G.711 and MP3
  audio (D17). The permissive alternatives, TCM and Wav2Vec2-AASIST, were trained only on
  ASVspoof 2019 and report 7.8-11.2% EER on *clean* In-the-Wild (research section 2), worse than
  W2V-Large's measured 6.1% on Opus.
- **Fine-tuning is a project of its own, with real risks:**
  - It needs labeled training audio from speakers outside both In-the-Wild splits. The 11 dev
    speakers are too few, and they're already used to choose the bands.
  - Fine-tuning a model post-trained on 74,000 hours of varied audio on one smaller dataset can
    erode the generalization that makes it good: the lab-overfitting failure from research
    section 1. That risk is general knowledge, not measured here.
  - The result has to be evaluated again from scratch, the bands chosen again, and the new
    weights inherit CC BY-NC-SA 4.0.
  - Whether a 317M-parameter model fine-tunes within the RTX 3050's 8 GB is untested.
- **The weak spots are visible where they matter.** Opus files carry a warning (D19), clips
  with too little sound are *Not assessed* (D5), and the README and model card list K1-K3.

**Impact:**
- No code change. Phase 3 builds on the current model and bands.
- Opus stays the weak spot: 8.6% of genuine Opus test clips are rated *Likely synthetic*, and
  audio that was Opus before it was converted gets no warning (K2).
- K1 stands: tones and music can score as confidently synthetic, so squeaktest can't be called
  ready for SOC use until K1 is fixed.

**Revisit:** with the Phase 5 robustness work (codec augmentation, then fine-tuning), if
Deepfake-Eval-2024 access arrives and shows large drops, or if a codec-robust detector with
usable weights is released.

## D21. TCM deferred: shortlisted, not run (Phase 2 bake-off)

**Decision:** TCM, the fourth model on the Phase 0 shortlist (research section 5), was not
evaluated. The bake-off stands at three AntiDeepfake models (Dylan, 2026-10-13).

**Why:**
- **Its main job was as a licensing fallback, and the fallback wasn't needed.** TCM is
  MIT-licensed, so it was on the list in case the AntiDeepfake weights' non-commercial license
  was rejected. Phase 0 accepted that license (research section 6, decision 1).
- **Its other job is answered by published numbers, roughly.** TCM was also meant to show how
  much training-data diversity buys. Trained only on ASVspoof 2019 LA, it reports 7.79% EER on
  In-the-Wild (Speech DF Arena), against W2V-Large's measured 1.3%. Those come from different
  protocols, so they aren't directly comparable. But the gap is large enough that TCM would need
  to do far better under squeaktest's protocol than reported to change the decision.
- **It costs more than the other three did.** It needs an estimated 1.3 GB download, and its
  weights are on OneDrive as a pickle-based `.pth` file (D13). It also needs a port of its
  conformer back-end, which has no `transformers` equivalent, plus its own parity check against
  the original code before its numbers count (D8).

**Impact:**
- The bake-off compares three models with the same training data and back-end, so it shows
  what the choice of front-end does, not what training data or architecture does.
- No permissively licensed model has been measured. If squeaktest ever needs one (for example,
  commercial use), TCM or Wav2Vec2-AASIST has to go through the harness first.

**Revisit:** if the license position changes, or for a Phase 5 ensemble. A model trained on
different data makes different mistakes, which is what an ensemble needs.

## D22. Host on a Docker Space on the free CPU tier, not ZeroGPU (Phase 3)

**Status:** replaced by D27 on 2026-10-19. Docker Spaces now need a paid plan, and there is no
hosted demo for now.

**Decision:** the public demo runs squeaktest's own Docker image on Hugging Face's free CPU
tier (2 vCPU, 16 GB). The same image serves local `docker compose` use (Dylan, 2026-10-13).

**The alternative:** a Gradio-SDK Space on ZeroGPU, Hugging Face's free shared GPU. A free
account can host two, and each visitor gets 2 minutes of GPU time a day (5 when logged in).
ZeroGPU only works with the Gradio SDK, where Hugging Face builds and runs the container.

**Why:**
- The demo exists to show careful handling of sensitive input. With our own image we control
  the base image and ffmpeg version, run as a non-root user (Hugging Face runs Docker Spaces
  as UID 1000), and make the application's files unwritable. On ZeroGPU, none of that is ours.
- One image everywhere: what runs locally is what runs in public, so local tests of the
  container also cover the demo.
- ZeroGPU's published PyTorch support stops at 2.13.0; squeaktest uses 2.14.1. Moving to it
  would mean a downgrade and a new parity check (D8).

**Impact:**
- **Speed.** On 2 threads of a desktop CPU, W2V-Large takes about 28 s per minute of audio
  (D17). Shared cloud vCPUs are usually slower, so this is a best case; the real figure is
  unmeasured until the image runs on a Space. A 60-second cap on the hosted demo bounds the
  wait, and voicemails are usually shorter.
- **One analysis at a time.** On 2 vCPUs, running two at once would only make both slower,
  so requests queue, and a full queue is refused (see the threat model).
- Detection quality is unchanged: the same model, windows and bands run on CPU and GPU (D12).

**Revisit:** if measured latency on the Space is unacceptable. The options then are the
W2V-Small model (D17), fewer windows per minute (D3), quantization (all three need new
evaluation runs), or ZeroGPU. The scoring code stays a plain function so that a
`@spaces.GPU` wrapper remains possible.

## D23. A plain HTML page instead of Gradio (Phase 3)

**Decision:** the web UI is one static HTML page with plain JavaScript and CSS, served by
squeaktest's FastAPI app. Gradio, planned in Phase 0, is dropped (Dylan, 2026-10-13).

**Why:**
- Gradio's main advantage is native support on Hugging Face Spaces, and ZeroGPU requires it.
  With a Docker Space (D22), any web page works.
- Two Gradio defaults conflict with "no third-party analytics or trackers": telemetry is on
  unless `GRADIO_ANALYTICS_ENABLED` turns it off, and the default theme loads fonts from Google
  Fonts, which sends each visitor's IP address to Google. Both can be turned off, but a page
  with no third-party code can't leak by default.
- The page can run under a strict Content Security Policy: scripts, styles and requests from
  squeaktest's own origin only, and no inline code. That blocks most injected-script attacks
  outright, and a test checks the page complies.
- Fewer dependencies. Gradio brings a large dependency tree that pip-audit would have to keep
  clean; FastAPI and uvicorn are much smaller.

**Impact:** the upload form and the timeline chart are hand-written (a few hundred lines),
and there's no ready-made Gradio API client. Detection quality is unaffected.

**Revisit:** if ZeroGPU becomes necessary (D22).

## D24. Uploads are the raw request body, not a multipart form (Phase 3)

**Decision:** `POST /api/analyze` takes the file as the request body with
`Content-Type: application/octet-stream`. Anything else gets 415. The usual web approach, a
`multipart/form-data` form and FastAPI's `UploadFile`, isn't used, and python-multipart isn't
installed.

**Why:**
- **One less parser in front of untrusted input.** python-multipart, the parser behind
  FastAPI's form handling, has had several denial-of-service flaws in exactly this code path
  (CVE-2024-53981, CVE-2026-40347, CVE-2026-42561). A raw body needs no parsing at all.
- **No filename ever arrives.** A multipart upload carries the file's name, which often holds
  a person's or company's name. With a raw body there is nothing to log by accident.
- **Uploads stay in memory.** Starlette writes file parts over 1 MB to a temporary file on
  disk while parsing. squeaktest reads the raw body into memory instead, with a hard cap: it
  refuses an oversized `Content-Length` before reading anything, and stops reading a body
  without one as soon as it passes the limit. A time limit stops slow uploads from holding a
  connection open indefinitely.
- **Cross-site uploads need permission they won't get.** `application/octet-stream` isn't one
  of the content types a browser sends cross-site without asking first (a CORS preflight).
  squeaktest answers no preflight, so another website can't make a visitor's browser submit
  files.

**Impact:** a plain HTML form can't send this, so the page uses a few lines of JavaScript.
From the command line:
`curl --data-binary @voicemail.m4a -H "Content-Type: application/octet-stream" http://127.0.0.1:8000/api/analyze`.
No effect on detection quality.

## D25. One analysis at a time, a short queue, and per-client limits (Phase 3)

**Decision:**
- The model scores one clip at a time on a single worker thread. Up to 4 more requests wait
  (`SQUEAKTEST_MAX_WAITING`). Beyond that, the server answers 503 straight away.
- Each client may have one upload or analysis under way at a time (429 otherwise).
- Each client may make 6 requests a minute, refilled evenly, one every 10 s
  (`SQUEAKTEST_RATE_LIMIT_PER_MINUTE`; 429 with `Retry-After`).
- At most 32 open connections (uvicorn's `limit_concurrency`).

**Why:** on 2 vCPUs, two analyses at once each run at about half speed, so running them one
at a time finishes the first sooner and the total no later. The limits are about availability
and memory, since uploads sit in memory while they wait:
- The queue bounds memory for queued uploads at 5 × 25 MB, and the connection cap bounds
  uploads still arriving at 32 × 25 MB.
- The one-at-a-time rule stops a single client from filling the queue with simultaneous
  uploads and locking everyone else out. The rate limit only slows a client over a minute.
- Six a minute leaves room to retry after picking the wrong file, while a client can't keep
  the server busy by itself for long.

**How a client is identified:** by the address that connected, unless
`SQUEAKTEST_PROXY_HOPS` says how many reverse proxies are in front. Each proxy appends the
address it saw to `X-Forwarded-For`, so with one proxy (Hugging Face's) the client is the last
entry. Earlier entries are written by the client and ignored, so a client can't make up a new
address for each request. If the hop count is wrong on a Space, either everyone shares one
limit (too low: safe, but slow) or clients can choose their address (too high: limits
bypassed). The queue and connection caps still hold either way. The right value for Hugging
Face is unverified until the image runs there.

**Privacy:** client addresses are held in memory only and never logged. A client's entry is
dropped once its limit has fully reset (a minute after its last request), and the app checks
for such entries every minute, so an address is held for at most about two minutes. No more
than 10,000 are held at once; past that, the least recently seen are dropped first.

**Impact:** no effect on detection quality. On a busy demo, some visitors are told to try
again rather than waiting minutes in a long queue.

**Revisit:** with real latency and traffic from the Space.

## D26. The container: pinned images, Debian's ffmpeg behind a decoder whitelist, a locked-down runtime (Phase 3)

**Decision:**
- Base images are pinned by digest: Python 3.12 on Debian 13 (trixie), and uv. Python packages
  are installed exactly as `uv.lock` pins them (CPU-only PyTorch).
- ffmpeg comes from Debian's packages (7.1.5 at the first build). For every file, ffmpeg is
  told which decoders it may open (`-codec_whitelist`): only those for the allowed codecs of
  the sniffed container. Files with embedded images (cover art) or extra streams are rejected
  (Dylan, 2026-10-13).
- The model is either built in (`WEIGHTS=included`, for the Space), with its SHA-256 checked at
  build time and again at every load, or mounted read-only (`WEIGHTS=mounted`, for local
  `docker compose`).
- At runtime the app runs as UID 1000. Its code and model belong to root, so it can't change
  them, and only `/tmp` is writable. Hugging Face's libraries are set to offline mode.
  `compose.yaml` adds a read-only root filesystem, `/tmp` in memory, no Linux capabilities, no
  privilege escalation, and CPU, memory and process limits.

**The ffmpeg options:**

| Option | Decoders an upload can reach | Who patches ffmpeg |
|---|---|---|
| Debian's package with a decoder whitelist (chosen) | Only the allowed ones; ffmpeg refuses the rest | Debian's security team; a rebuild picks up fixes |
| A custom minimal build | Only the ones compiled in | squeaktest: watch ffmpeg's advisories, rebuild |
| A third-party static build | All of them | The third party, another supplier to trust |

**Why:**
- Decoders are the main attack surface of an upload service, and a bug in a decoder that never
  runs can't be exploited. Before this decision, ffprobe opened a decoder for every stream in
  a file while probing, before squeaktest's allowlist check, and the check only looked at the
  first audio stream. Cover art reached the JPEG and PNG decoders, and a second stream could
  reach any decoder ffmpeg has (verified with test files). Now ffmpeg itself refuses them, and
  tests prove it with cover-art and two-stream files.
- With the whitelist, Debian's full build exposes about the same decoders as a minimal build
  would, and patching stays with a team whose job it is. What's left in common with a minimal
  build is the container parsers (demuxers) for the five formats, which run on every upload
  either way. What's left over is the whitelist itself: a bug in how ffmpeg enforces it would
  expose the other decoders.
- Cover art is rejected rather than allowed through `mjpeg` and `png`, because image decoders
  have a long history of security bugs, and voicemails, voice notes and call recordings rarely
  carry cover art.

**What only `compose.yaml` enforces:** on a Hugging Face Space, Hugging Face runs the container,
so the read-only root filesystem, dropped capabilities, no-new-privileges and resource limits
don't apply there. The Space keeps what's built into the image: the non-root user, root-owned
code and model, offline libraries, and pinned contents.

**Impact:**
- MP3 and M4A files with cover art are rejected, with a message that says what's allowed.
  Detection quality is unaffected.
- The image gets Debian's security fixes, and new base images, only when it's rebuilt and the
  digests are bumped. Neither is automated yet.
- `WEIGHTS=included` adds about 1.3 GB to the image; the container itself never needs the
  network.
- **Untested:** `docker compose up` itself (no local container engine was available). CI builds
  the image without the model and checks its hardening on every push
  (`scripts/check_image.sh`). A `WEIGHTS=included` build was first run on 2026-10-19, by the
  manual latency workflow (D27), which also ran it with `compose.yaml`'s restrictions.

**Residual risk (for the threat model):** a code-execution bug in an allowed decoder or demuxer
would still run as UID 1000 inside the container, the same user as the app. From there it may
be able to read uploads waiting in the queue and reach the network. Decoding in a separate
sandbox (its own user or container, a seccomp profile) is a Phase 5 candidate.

**Revisit:** set up a scheduled rebuild with digest updates; act at once when an ffmpeg advisory
affects an allowed decoder or demuxer; and revisit if Hugging Face lets Spaces set runtime
restrictions.

## D27. No hosted demo for now: the code on GitHub, self-hosted on Dylan's homelab (Phase 3)

**Decision:** squeaktest isn't hosted publicly for now. The code, results and Docker image
stay on GitHub, and Dylan runs it on a homelab for testing. Replaces D22 (Dylan, 2026-10-19).

**Why:**
- D22's premise is gone: Hugging Face now needs PRO ($9 a month) to host a Docker or Gradio
  Space. Free accounts can only make Static Spaces, which can't run a server.
- The project's audience is security employers reading the repository: the evaluation, the
  decision records, the threat model and the hardening. A live demo adds little to that, and
  self-hosting the image is a fair test of the image itself.
- The other free options don't fit. Free tiers with 512 MB of memory can't hold the model (2.84
  GB peak, below). Running the model in the visitor's browser would keep audio on their device,
  but this model is too large for that; it would mean a smaller or quantized model and new
  evaluation runs. Google Cloud Run might be free for light use but needs a card on file and
  spending caps.

**Measured** with `.github/workflows/latency.yml` (run by hand) on 2026-10-19: the
`WEIGHTS=included` image on a GitHub runner (Intel Xeon Platinum 8370C), limited to 2 CPUs, with
`compose.yaml`'s other restrictions. The clip is 60 s of synthetic buzz with every window
scored, the worst case for its length.

| Measure | Result |
|---|---|
| 60 s clip, first request | 21.9 s |
| 60 s clip, three more requests | 20.4, 20.7, 20.8 s |
| Startup, until the server answers | 7 s |
| Peak container memory | 2.84 GB |
| Image size | 2.95 GB |

That's faster than D17's 28.1 s on 2 desktop threads, so CPU speed isn't a reason to change
the model (D22's revisit options). One run on one runner type: another host's 2 vCPUs may be
slower.

**Impact:**
- Nothing in the code changes. The settings meant for a Space stay as general options:
  `SQUEAKTEST_PROXY_HOPS` for any reverse proxy in front of the app, and
  `SQUEAKTEST_FRAME_ANCESTORS` for any site that frames the page.
- Wherever squeaktest runs now, its operator controls the container. So `compose.yaml`'s
  read-only filesystem, dropped capabilities and resource limits can apply everywhere, which
  closes the gap D26 noted for Spaces.
- Still unverified: the proxy hop count behind a real reverse proxy, and `docker compose up`
  itself. Both get checked on the homelab.
- The threat model (Phase 3) covers a self-hosted deployment, not a public one.

**Revisit:** if a public demo is wanted, on Hugging Face or squeaktest.com. The latency and
memory figures above size what it would need.

## D28. Answer only to known host names, against DNS rebinding (Phase 3)

**Decision:** the web app answers a request only if its `Host` header names the machine itself
(`localhost`, `127.0.0.1`, `::1`) or a host the operator lists in `SQUEAKTEST_ALLOWED_HOSTS`.
Anything else gets 400, with a message naming the setting, and one metadata-only log line
(`"outcome": "wrong_host"`). Names compare case-insensitively, and ports are ignored. Requested
by Dylan on 2026-10-19 as the first fix from the threat model.

**Why:**
- It closes T3 in `docs/threat-model.md`. Without it, any website a LAN user visited could
  point its own domain at squeaktest's address and use that user's browser to send crafted
  files to the decoder. A rebound request still carries the attacker's host name, so it's
  refused before any route runs.
- The loopback names are always allowed. A rebinding page can't make the browser send them,
  and the image's health check and local use depend on them. So the default is safe and needs
  no setup, and the setting is only needed once the server listens on another address.
- **Our own middleware, not Starlette's `TrustedHostMiddleware`.** Starlette's compares names
  case-sensitively and answers in plain text. More to the point for a detection engineer, it
  logs nothing. A refused `Host` is the only trace a rebinding attempt leaves, so it should be
  a log line a SIEM can count. Ours is about 30 lines plus tests.
- Wildcards and `*` are refused in the setting. An allowlist that allows everything would turn
  the check off without anyone noticing.

**Impact:** no effect on detection quality. Anyone reaching the server by a name or address
that isn't listed gets the 400 and a message saying what to set. Behind a reverse proxy, either the
proxy passes the original `Host` through (Caddy and Traefik do by default; nginx needs
`proxy_set_header Host $host`), or the name it sends upstream must be listed.

**Revisit:** if squeaktest moves behind a proxy that rewrites `Host`, or gets a public name.

## D29. Log decoder crashes apart from rejections, without telling the uploader (Phase 3)

**Decision:** when ffprobe or ffmpeg is killed instead of exiting (a negative return code on
Linux, such as -11 for SIGSEGV; an NTSTATUS code such as `0xC0000005` on Windows), squeaktest
raises `DecoderCrashedError`. The web app logs it as `"outcome": "decoder_crashed"` at WARNING,
with the step (`probe` or `decode`) and the return code. The uploader gets exactly the 400 and
message any unreadable file gets. Found by a security check on 2026-10-19; Dylan asked for the
fix the same day.

**Why:**
- Before this, a crash was logged as `"outcome": "rejected"` with the same reason as a text
  file renamed `.mp3`. Exploit attempts and fuzzing against the decoder (T1, the most valuable
  attack on squeaktest) were invisible among ordinary mistakes.
- A crash is rare for honest files, so it's a high-fidelity signal: one is worth a look, and a
  run of them from the same time window is worth an alert. WARNING also makes it stand out in
  `docker logs` without a SIEM.
- **The uploader isn't told.** A different message would tell an attacker which of their files
  crashed the decoder, which is the feedback a fuzzer needs. They can still time responses or
  fuzz their own copy of ffmpeg, so this denies little, but it costs nothing.
- The out-of-memory killer's SIGKILL counts as a crash too. A file that makes the decoder run
  out of memory is just as worth knowing about.

**Impact:** no effect on detection quality or on what users see. Log consumers get one more
outcome value. A working exploit doesn't crash anything, so this catches attempts, not
success; container egress blocking and a decode sandbox (threat model T1) remain the controls
for that.

**Revisit:** with the SIEM event (Phase 5), which should carry this outcome; or if decoding
moves to a separate sandbox, which would report crashes its own way.

## D30. Publish on one named address, loopback by default, and test `docker compose up` in CI (Phase 3)

**Decision:** `compose.yaml` publishes port 8000 on the address in `SQUEAKTEST_LISTEN_ADDRESS`,
or on `127.0.0.1` when that's unset, and passes `SQUEAKTEST_ALLOWED_HOSTS` into the container.
On the homelab both name the server's LAN address, and the docs say not to use `0.0.0.0`. CI
runs `docker compose up` on every push with the real model mounted: first on loopback, then on
the runner's network address. Dylan asked for both on 2026-10-28 (item 3 of the threat model's
list of fixes).

**Why:**
- **Loopback by default** keeps `docker compose up` safe on a laptop: nothing else on the
  network can reach the decoder until someone chooses to publish it.
- **One named address, not `0.0.0.0`.** Docker writes its own firewall rules for published
  ports, ahead of `ufw`'s, so a port on `0.0.0.0` is open on every network the host is on
  (a VPN, a guest network, another bridge), whatever the host firewall says. A named address
  is the only place it can be reached.
- **An address, not a "LAN mode" switch.** Compose can't work out which address is the LAN's;
  the operator knows.
- **The allowlist comes along.** Requests to the LAN address would otherwise get D28's 400.
  Compose passes `SQUEAKTEST_ALLOWED_HOSTS` through, empty by default, which the app treats as
  unset.
- **Documented, not enforced.** Compose can't validate a variable, and the app inside the
  container can't see which address Docker published it on, so nothing refuses `0.0.0.0`.
- **CI uses the real model.** Without one, `serve` exits by design (`scripts/check_image.sh`
  tests that), so a compose test without the model couldn't check the HTTP side, an analysis
  or the restart. The 1.27 GB model is cached with a key on `model.py`, which pins it, so it's
  downloaded again only when the pin changes or GitHub drops the cache (after 7 days unused).
  The cache isn't trusted: the app checks the file's SHA-256 on every load.
- **Killed from the host, not with `docker kill`.** Docker counts `docker kill` as stopping
  the container by hand, so `unless-stopped` wouldn't restart it and the test would prove
  nothing. A SIGKILL to the server's process is what the out-of-memory killer sends.

**Impact:** no effect on detection quality. The homelab can run `compose.yaml` as it is, with a
`.env` file. The container job takes longer, and downloads the model from Hugging Face on a
cache miss.

**Revisit:** behind a reverse proxy (T7, T8), squeaktest should go back to loopback and the
proxy should take the LAN address. For IPv6, the address needs brackets (`[fd00::5]`), which
hasn't been tried.

## D31. Leave steady tones out of scoring with a tone gate in the VAD (K1)

**Decision:** before windowing, squeaktest looks for steady tones. A 25 ms frame is a tone when,
in the 256 ms of audio centred on it, at least 99% of the energy above 200 Hz sits in the 10
strongest frequency bins: one or two sharp lines, the way a beep, a dial tone or a ringing tone
looks. Each run of tone frames is widened by 128 ms on both sides, and tone frames don't count
as activity. Everything after that is D1-D5 unchanged: tones at the ends are trimmed like
silence, a window needs 50% activity that isn't tone, and a clip needs 1 s of it or it is *Not
assessed*. The report gives the seconds left out (`audio.tones_s`) and a `steady_tones` warning
that says music isn't filtered. `WindowingConfig(skip_tones=False)` turns the gate off. Dylan
chose "measure first, then a tone gate" on 2026-10-29, from the K1 options (a tone check now, a
model-based VAD for music later).

**What K1 turned out to be (measured with `eval.k1`, gate off; full tables in
`eval/results/antideepfake-w2v-large/k1.md`):**
- **Non-speech on its own gets arbitrary bands.** Generated signals, 12 s each, under all five
  codecs (test variants): a 600 Hz tone and a UK dial tone are *Likely synthetic* every time; a
  1400 Hz tone is mostly *Likely genuine*; a busy tone is anything from *Likely genuine* to
  *Likely synthetic*. Synthetic music is *Likely genuine* every time (score 0.00), which is false
  reassurance rather than a false alarm. White and pink noise land mostly in *Uncertain*.
- **On real clips the bigger harm is hidden fakes, not false alarms.** 200 In-the-Wild test
  clips, each with a sound added, pooled over clean, G.711 and Opus: with nothing added, 80.4% of
  fakes are rated *Likely synthetic*. A busy tone after the speech drops that to 28.3%, ringing
  before it to 67.6%, key presses (DTMF) to 55.7%, and synthetic hold music before it to 29.7%.
  Most In-the-Wild clips are one or two windows long, so the top-3 mean (D6) averages the
  sound's low-scoring windows in with the speech. Genuine clips rated *Likely synthetic* stayed at
  0.0-2.1% with every sound.

**Why this gate:**
- **Long frames separate tones from voices.** Over 256 ms a steady tone stays one sharp line,
  while a voice's pitch moves and smears its many harmonics across neighbouring bins. In a first
  sample of 300 dev clips (110,000 frames with sound), the share never reached 0.99 in speech;
  generated tones, dual tones and DTMF digits all score 0.999 or more. Shorter frames didn't
  separate them: at 64 ms, 1.6% of speech frames passed 0.99. Over all 4,826 dev clips, 8 had a
  stretch of 0.20-0.37 s flagged, at 210-285 Hz: most likely a held vowel or an "uhh" at a very
  steady pitch, mostly in old, band-limited recordings. None of them changed a score.
- **Above 200 Hz** because the only tone-like stretches found in dev speech were mains hum in
  the pauses of old recordings (59-62 Hz). Hum's own harmonics above 200 Hz still count, so a
  clip of nothing but hum is left out too.
- **Widening by half a frame** because a long frame that straddles a tone's start or end also
  holds silence or speech, so it reads as less pure than the tone is. Without it, a 0.5 s busy
  burst would lose half its length to the edges.
- **Same pattern as D1:** no dependency, no model, no download, and every value can be checked by
  hand. It adds about 0.4 s to a 60-second clip on CPU (the model takes about 21 s) and under
  100 MB of memory for a 5-minute clip.
- **Music is left for later.** It is the case a model-based VAD (Silero) or an AudioSet
  classifier would be for. That needs a dependency or format decision (D13) and licensed music
  to test on, both of which are Dylan's calls.

**Impact (measured, test speakers and test signals):**
- **Fixed:** every steady tone, call-progress tone (dial, ringing, busy, reorder, special
  information tones), slow DTMF and fax tone is *Not assessed* on its own, under every codec but
  two (below). On real clips with a beep, ringing or a busy tone added, the gate gives back the
  original clip's band 97.3-99.5% of the time, with a median score change of 0.0000: fakes
  caught return from 28.3% (busy tone) and 67.6% (ringing) to 80.4-80.8%.
- **What it costs on real speech:** over every In-the-Wild test clip (26,953 clean, plus the
  3,000-clip codec samples), the gate removed 0.03-0.05% of the sound. EER, fakes caught at 1%
  false positives and genuine clips rated *Likely synthetic* are unchanged to two decimals under
  every codec. It isn't free, though. 126 clean clips lost a stretch at a voice's pitch
  (180-310 Hz), mostly from speakers who hold vowels or sing (Alec Guinness, Mr. Rogers, Frank
  Sinatra), up to 2.2 s in one clip. Four of them were clips under 2 s that fell below the 1 s
  minimum and became *Not assessed* (14.30% of clips before, 14.32% after), and one of those was
  a fake that had been *Likely synthetic*. Dev showed the same thing on a smaller scale (8 clips,
  no band changed), so this is a held-out finding, reported as it is rather than tuned away.
- **A tone under speech doesn't hide it.** Frames where a voice is heard over a tone or hum
  aren't a single sharp line, so they stay in. With a tone 20 dB under the speech, a clip's
  speech time was at most 0.28 s shorter than without the tone, and no shorter at all in 95% of
  clips.
- **Not fixed:**
  - **Fast DTMF** (80-100 ms digits): a 256 ms frame spans several digits and gaps, which
    blurs the lines. Fakes after 4 s of key presses are still caught only 55.7% of the time.
  - **Music:** synthetic hold music before a fake still drops detection to 29.7%, and music
    15 dB under the speech to 46.1%. Real music is unmeasured.
  - **Noise, sweeps and other sounds** are scored as before.
  - **Two codec misses:** a 2100 Hz fax answer tone under AMR-NB and 50 Hz hum under Opus. Both
    codecs smear a pure tone enough that it no longer reads as one line.
- **More *Not assessed* clips** where the only sound was a tone: they never had speech to judge.

**Revisit:** when music is tackled (a model-based VAD would replace this check, or sit beside
it); if held vowels and singing matter (for example, AI-generated songs), with a rule chosen on
dev data only; if fast DTMF matters on real recordings, a DTMF detector at the standard frequencies (the
method telephone exchanges use, designed not to fire on speech); and on real voicemails, which
nothing here is.

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

**Measured (2026-10-29, `eval.k1`, details in D31):** non-speech gets arbitrary bands, not
always "fake": some tones are *Likely synthetic* every time, synthetic music is *Likely
genuine* every time. On real clips the bigger harm turned out to be hidden fakes: a busy tone
after a fake drops detection from 80.4% to 28.3%, synthetic hold music before it to 29.7%.

**Status:** partly fixed. Steady tones (beeps, dial, ringing, busy and fax tones, slow DTMF) are
left out of scoring by the tone gate (D31). Still open: music, fast DTMF, noise, and two codec
cases the gate misses. Real voicemails haven't been tested. Music is the remaining blocker
before any claim that squeaktest is ready for SOC use.

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
augmentation). The Phase 2 gate deferred that to the Phase 5 robustness work (D20).

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
