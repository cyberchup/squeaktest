# Model card: squeaktest's default detector

This card covers what squeaktest actually runs: NII's AntiDeepfake Wav2Vec2-Large weights,
inside squeaktest's own pipeline (silence trimming, 4-second windows, top-3 aggregation and
bands). squeaktest did not train or fine-tune the weights; it measured them. Every number below
was measured by squeaktest unless it says *reported by NII*. The format follows
[Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) (Mitchell et al., 2019).

## Model details

| | |
|---|---|
| Weights | AntiDeepfake Wav2Vec2-Large, by NII's Yamagishi Lab ([paper](https://arxiv.org/abs/2506.21090), [model card](https://huggingface.co/nii-yamagishilab/wav2vec-large-anti-deepfake)) |
| Version | [`nii-yamagishilab/wav2vec-large-anti-deepfake`](https://huggingface.co/nii-yamagishilab/wav2vec-large-anti-deepfake/tree/7ccfcbb27d3098dafeb67596d4bb26e51e4627fc) at commit `7ccfcbb`, SHA-256 `b27943fe…c2bb5`, checked on every load |
| License | CC BY-NC-SA 4.0: non-commercial use only. Fine-tuned versions would inherit it |
| Architecture | wav2vec 2.0 Large (24 layers, 1024 wide, 317M parameters), mean pooling over time, then `Linear(1024, 2)` |
| Runtime | Hugging Face `transformers`, not NII's fairseq code. Outputs match NII's code within 7e-6 (logits) on five test signals ([D8](decisions.md)) |
| Input | A wav, mp3, flac, ogg or m4a file with allowed audio streams only (no cover art or extra streams), decoded by ffmpeg to 16 kHz mono |
| Output | A band (*Likely synthetic*, *Uncertain*, *Likely genuine* or *Not assessed*), a raw score from 0 to 1, and a score for each 4-second window |
| Chosen | Phase 2 bake-off, 2026-10-07 ([D17](decisions.md)), over AntiDeepfake MMS-300M and Wav2Vec2-Small. TCM, also shortlisted, wasn't run ([D21](decisions.md)). Shipped unchanged with its limits documented ([D20](decisions.md)) |

**The pipeline:** trim leading and trailing silence; refuse to score clips with under 1 second
of sound; cut 4-second windows every 2 seconds; skip windows that are mostly silent; score each
window; average the three highest window scores; map that score to a band. Each step's reasons
are in [docs/decisions.md](decisions.md) (D1-D6, D18).

## Intended use

- **Triage of a suspicious recording** by a SOC analyst, or by a helpdesk or finance team
  escalating to one: a voicemail, voice note or call recording in a possible vishing,
  executive-impersonation or helpdesk social-engineering case.
- **One signal among several.** A *Likely synthetic* result should prompt verification through
  another channel (call back on a known number, ask a question only the real person can answer),
  not a conclusion.

**Out of scope:**

- The sole basis for any decision. Every result says so.
- Legal or forensic evidence, employment decisions, or accusations against a person.
- Identifying who is speaking. The model answers "synthetic or not", never "who".
- Commercial use (the weights' license).
- Live call screening. It was evaluated on recordings only.

## Training data (reported by NII)

About 74,000 hours from 29 datasets in over 100 languages, including ASVspoof 2019, 2021 and 5,
MLAAD, CodecFake, SpoofCeleb, WaveFake and DiffSSD. In-the-Wild and Deepfake-Eval-2024 were
not used, which is why In-the-Wild works as a held-out test here. ASVspoof 2019 was used, so
results on it are *seen data*.

## Evaluation data

- **In-the-Wild** ([Müller et al. 2022](https://www.isca-archive.org/interspeech_2022/muller22_interspeech.pdf), CC BY-SA 4.0): genuine and
  deepfake speech of English-speaking public figures, collected from the web. 31,779 clips from
  54 speakers. Split by speaker, so no voice appears on both sides:
  - **Dev:** 11 speakers, 4,826 clips. Used for every choice (band thresholds, the minimum
    amount of sound, the calibration test).
  - **Test:** 43 speakers, 26,953 clips. Used only to report.
- **Phone and compression conditions:** a 3,000-clip stratified sample of the test speakers,
  re-encoded with ffmpeg as G.711 μ-law at 8 kHz (landline), AMR-NB at 12.2 kbit/s (mobile),
  Opus at 12 kbit/s (VoIP and voice notes) and MP3 at 32 kbit/s. The dev speakers were also
  re-encoded under all four codecs, for choosing the bands. These are simulations: one encode
  pass, with no packet loss, line noise or echo cancellation.
- **ASVspoof 2019 LA eval**, a 5,000-clip sample: lab-generated fakes, seen in training. A
  ceiling for comparison, not a held-out result.

## Results

In-the-Wild test speakers unless noted. Brackets are 95% intervals from resampling whole
speakers. The full report is [eval/results/antideepfake-w2v-large/report.md](../eval/results/antideepfake-w2v-large/report.md),
and the comparison with the other two models is in [eval/results/bakeoff.md](../eval/results/bakeoff.md).

**How to read these:** EER is the error rate at the threshold where false alarms and missed
fakes are equally common (lower is better). "Fakes caught at 1% false positives" is the
detection rate if the alert threshold is set to flag 1% of genuine recordings. Both measure the
model's ranking, independent of squeaktest's bands.

### Detection quality

| Audio | EER | ROC-AUC | Fakes caught at 1% false positives |
|---|---|---|---|
| Clean (23,098 scored clips) | 1.3% (0.8-2.4%) | 0.999 | 98.3% (96.2-99.4%) |
| Landline: G.711 μ-law, 8 kHz | 2.3% (1.6-3.5%) | 0.998 | 96.1% |
| Mobile: AMR-NB 12.2 kbit/s | 4.3% (2.8-7.0%) | 0.993 | 90.3% |
| Voice note or VoIP: Opus 12 kbit/s | 6.1% (3.7-9.1%) | 0.984 | 73.7% |
| Compressed: MP3 32 kbit/s | 2.2% (1.2-3.7%) | 0.998 | 95.9% |
| ASVspoof 2019 LA (seen in training) | 1.1% (0.4-1.5%) | 1.000 | 98.8% |

Codec rows use the 3,000-clip sample. On those same clips, clean audio has 1.6% EER.

### With squeaktest's bands

*Likely synthetic* at raw score 0.9515 and above, *Likely genuine* below 0.4856 ([D18](decisions.md)).

| Audio | Genuine: *Likely synthetic* | Genuine: *Uncertain* | Fakes: *Likely synthetic* | Fakes: *Uncertain* | Fakes: *Likely genuine* |
|---|---|---|---|---|---|
| Clean | 0.1% | 0.8% | 94.2% | 4.0% | 1.8% |
| G.711 | 0.1% | 1.0% | 90.4% | 6.1% | 3.5% |
| AMR-NB | 0.3% | 6.5% | 86.7% | 9.9% | 3.3% |
| Opus 12k | 8.6% | 10.6% | 95.7% | 2.9% | 1.5% |
| MP3 32k | 0.1% | 1.1% | 89.8% | 6.6% | 3.6% |

Clips with under 1 second of sound get *Not assessed* instead of a band: 16.0% of genuine and
11.3% of fake test clips. In-the-Wild has many very short clips ([D5](decisions.md)).

### What an alert means at realistic base rates

The test set is 38% fake; a real queue of voicemails is almost all genuine. Share of *Likely
synthetic* results that would be real fakes:

| Share of recordings that are fake | Clean | G.711 | AMR-NB | Opus 12k | MP3 32k |
|---|---|---|---|---|---|
| 1 in 10 | 99% (98-100%) | 99% (98-100%) | 97% (92-99%) | 55% (43-74%) | 99% (98-100%) |
| 1 in 100 | 93% (85-100%) | 94% (79-100%) | 73% (52-92%) | 10% (7-21%) | 94% (82-100%) |
| 1 in 1,000 | 58% (36-100%) | 59% (27-100%) | 21% (10-52%) | 1% (1-3%) | 59% (30-100%) |

The wide intervals come from the false-positive rate resting on a handful of genuine clips
(0.1% of about 14,000).

### Compared with NII's published numbers

NII reports 1.91% EER on In-the-Wild for these weights, and an independent benchmark reports
2.89% for an identical copy. squeaktest's 1.3% isn't directly comparable. It covers test
speakers only, it leaves out the 14% of clips with under 1 second of sound (likely among the
hardest), and it scores 4-second windows rather than whole clips. On dev speakers, dropping the
1-second minimum raises EER from 0.9% to 1.5%.

## Known limitations

- **Current real-world deepfakes: probably much worse, and not measured.** In-the-Wild was
  published in 2022, so its fakes come from older generators. On Deepfake-Eval-2024, deepfakes
  that circulated online in 2024, NII reports 33.36% EER for these weights, little better than
  a coin flip. squeaktest hasn't measured that set (access is gated). Read every result above as
  a best case for modern voice cloning.
- **English only.** Every evaluation clip is English, so other languages are untested ([D17](decisions.md)).
- **Opus audio causes false alarms** ([K2](decisions.md)). See the tables above. squeaktest
  warns when the file itself is Opus, but can't tell when audio was Opus before being converted.
- **Very short clips are less reliable** ([K3](decisions.md)): 5.1% EER under 2 seconds.
- **Non-speech audio** such as tones, beeps and hold music can score as confidently synthetic
  ([K1](decisions.md)). This is not measured, and still open.
- **The codec conditions are simulations:** one encode pass, with no packet loss, line noise
  or echo cancellation. Real calls may do worse.
- **Partial fakes are untested.** In-the-Wild's fakes are whole clips. A cloned sentence spliced
  into a real call is the case top-3 aggregation was designed for ([D6](decisions.md)), but it
  has no measurement.
- **The bands rest on 11 dev speakers.** The dev set understated the Opus false-alarm rate
  (3.4% on dev against 8.6% on test).
- **Deliberate evasion isn't evaluated.** An attacker who adds noise or re-encodes audio to slip
  past a detector is a Phase 5 topic.

## Ethical considerations

- **Both error types hurt someone.** A false *Likely synthetic* can delay or discredit a genuine
  caller, such as a real executive's urgent request or a customer accused of fraud. A false
  *Likely genuine* can reassure a team that is being targeted. That is why squeaktest shows bands
  instead of a verdict, keeps an *Uncertain* band, and attaches a disclaimer to every result.
- **Voice is biometric data.** squeaktest decodes each recording through a temporary file in a
  private folder, deleted as soon as decoding finishes (even on errors), and its output never
  includes the file's name.
- **Evaluation speakers are public figures** speaking in public videos, from a published,
  openly licensed dataset. No voice recordings are stored in the repository.
