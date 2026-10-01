# Design decisions

Each entry records what was chosen, why it beat the alternatives, what it does to detection
quality, and when to revisit it. Numbers marked *toy* come from the synthetic-tone tests in
`tests/test_inference.py`, not from real speech. None of these defaults has been tuned on
real data yet; Phase 2 does that on the In-the-Wild dev split.

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

**Revisit:** in Phase 2, measure accuracy against speech length on dev data, and set this
minimum where accuracy stops collapsing.

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

**Revisit:** in Phase 2, compare mean, max and top-k on In-the-Wild (whole-clip fakes) and, if
licensing allows, on a partial-spoof dataset such as PartialSpoof (license unverified).

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

**Status:** open. Measure it in Phase 2 on audio with tones and music, and fix it before any
claim that squeaktest is ready for SOC use.
