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
