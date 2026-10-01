# Phase 0 research: detectors, datasets, recommendation

*Researched 2026-09-30. Every accuracy number in this document is **reported** by a model's authors or by a public benchmark. squeaktest has not reproduced any of them yet; Phase 2 does that. Items marked **(unverified)** still need checking.*

## Summary

- **Initial model: NII's AntiDeepfake MMS-300M** (317M parameters). Of the models small enough for the free CPU tier, it has the best reported results on audio it was not trained on: 2.90% EER on In-the-Wild. The catches: its weights are **non-commercial (CC BY-NC-SA 4.0)**, the official code needs an outdated library (fairseq) that we would replace, and it still has 32.84% EER on real deepfakes collected from social media in 2024.
- **Bake-off shortlist for Phase 2:** AntiDeepfake MMS-300M, AntiDeepfake Wav2Vec2-Large, AntiDeepfake Wav2Vec2-Small (95M, the cheap option), and TCM (MIT license, trained only on lab data).
- **Evaluation:** In-the-Wild as the main held-out set, since none of the candidates trained on it. Split it by speaker into a calibration part and a test part, and re-encode the test part through phone-style codecs for the SOC case. ASVspoof 2019 LA serves as the in-domain reference.
- **The honest headline:** on real-world deepfakes, open detectors are far from reliable. On Deepfake-Eval-2024 the best open models sit around 26–33% EER. On the multilingual ML-ITW set, older lab-trained models land at 40–50%, close to a coin flip. The README has to say this up front.

## 1. Primer: how these detectors work

### Front-end and back-end

Most current detectors have two parts:

- **Front-end:** a large self-supervised speech model (wav2vec 2.0, XLS-R, MMS, WavLM). It was pretrained on huge amounts of unlabeled speech by learning to fill in masked chunks, so it already "knows" what natural speech sounds like. MMS-300M, for example, was pretrained on about 500,000 hours of speech in more than 1,400 languages.
- **Back-end:** a small classifier on top that outputs "bona fide" (real) or "spoof" (fake). It can be as simple as averaging over time plus one linear layer (AntiDeepfake), or something more elaborate: graph attention (AASIST), a conformer (TCM), or a state-space model (XLSR-Mamba).

The whole stack is then fine-tuned on labeled real and fake audio.

### Why the front-end matters

Small models trained from scratch on one lab dataset learn the fingerprints of the specific generators in that dataset. They look excellent in the lab and fall apart elsewhere. Results from the Speech DF Arena benchmark, for models all trained on ASVspoof 2019:

| Model | Size | EER on ASVspoof 2019 (in-domain) | EER on In-the-Wild (out-of-domain) |
|---|---|---|---|
| RawNet-2 (no front-end) | 17.6M | 4.59% | 49.00% |
| TCM (XLS-R front-end) | 319M | 0.18% | 7.79% |
| XLSR+SLS (XLS-R front-end) | 340M | 0.23% | 7.45% |

In detection-engineering terms, in-domain versus out-of-domain is the difference between testing a rule on the logs you wrote it from and deploying it at a different customer.

Training data diversity matters as much as architecture. AntiDeepfake took the same kind of front-end and post-trained it on about 74,000 hours from 29 datasets in over 100 languages. Its models reach 1.7–4.2% EER on In-the-Wild.

### Metrics

- **EER (equal error rate):** the error rate at the threshold where the false-positive rate equals the false-negative rate. Lower is better, and 50% means a coin flip. It is a single number that is good for comparing models, but it is not an operating point anyone would deploy.
- **ROC-AUC:** the probability that a random fake clip scores higher than a random real one. 0.5 is chance, 1.0 is perfect.
- **Detection rate at a fixed false-positive rate:** the SOC number. "With 1% of real calls flagged, what fraction of deepfakes do we catch?"
- **Calibration:** whether a score of 0.8 really means that about 80% of clips scored like that are fake. A reliability diagram plots predicted scores against observed fractions. Raw model outputs are usually not calibrated, and calibration does not carry over between domains. The AntiDeepfake Wav2Vec2-Large model card lists its EER threshold as 0.33 on In-the-Wild and 0.9995 on Deepfake-Eval-2024. Same model, completely different meaning for the same score. That is why squeaktest calibrates on data resembling what it will see, and labels raw scores as uncalibrated until then.

### Shortcut learning

A model learns whatever separates the classes in its training data, even if it isn't the voice. In ASVspoof 2019, real clips have longer leading and trailing silence than fake ones. A classifier that looks only at the length of leading silence reaches 15% EER, and trimming the silence pushes a real detector from 3% to 15% EER ([Müller et al. 2021](https://arxiv.org/pdf/2106.12914)). Deepfake-Eval-2024 found that audio models lose about 35% accuracy on clips with silence and 18% on clips with background music. It is the same failure as a detection rule that keys on your lab's hostname instead of the behavior. squeaktest should trim silence and skip windows that contain no speech instead of scoring them.

### Read first, in this order

1. [Does Audio Deepfake Detection Generalize?](https://www.isca-archive.org/interspeech_2022/muller22_interspeech.pdf) (Müller et al., Interspeech 2022): the In-the-Wild paper.
2. [Speech is Silver, Silence is Golden](https://arxiv.org/pdf/2106.12914) (Müller et al., 2021): shortcut learning.
3. [Speech DF Arena](https://arxiv.org/abs/2509.02859) (2025): 15 detectors compared on 14 datasets.
4. [Deepfake-Eval-2024](https://arxiv.org/abs/2503.02857) (2025): how detectors do on real 2024 deepfakes.
5. [Post-training for Deepfake Speech Detection](https://arxiv.org/abs/2506.21090) (2025): the AntiDeepfake models.

## 2. Candidate models

ITW = In-the-Wild, DE-2024 = Deepfake-Eval-2024. All EERs are reported, not measured.

| Model | Size | Trained on | ITW EER | Other out-of-domain EER | License | Weights | Verdict |
|---|---|---|---|---|---|---|---|
| AntiDeepfake MMS-300M | 317M | 74k h, 29 datasets | 2.90% | DE-2024 32.84%, ADD2023 7.93%, FakeOrReal 1.40% | CC BY-NC-SA 4.0 | HF, safetensors, 1.27 GB | **Shortlist (initial)** |
| AntiDeepfake W2V-Large | 317M | same | 1.91% | DE-2024 33.36%, ADD2023 13.25%, FakeOrReal 0.67% | CC BY-NC-SA 4.0 | HF, safetensors | **Shortlist** |
| AntiDeepfake W2V-Small | 95M | same | 4.24% | DE-2024 33.33%, ADD2023 13.02%, FakeOrReal 22.03% | CC BY-NC-SA 4.0 | HF | **Shortlist (cheap)** |
| AntiDeepfake 1B / 2B models | 0.96–2.2B | same | 1.65–1.82% | DE-2024 25.8–27.8% | CC BY-NC-SA 4.0 | HF | Too heavy for free CPU |
| TCM | 319M | ASVspoof 2019 LA | 7.79% | Arena average 15.77% | MIT | OneDrive | **Shortlist (permissive)** |
| XLSR-Mamba | 319M | ASVspoof 2019 LA | 6.70% | Arena average 14.21% | MIT | HF | Excluded: needs CUDA-only kernels |
| XLSR+SLS | 340M | ASVspoof 2019 LA | 7.45% | ML-ITW 48.83% | **No license file** | Google Drive | Excluded: no license |
| Wav2Vec2-AASIST | 318M | ASVspoof 2019 LA | 11.19% | ML-ITW 49.40% | MIT | Google Drive | Backup permissive option |
| Nes2NetX | 318M | ASVspoof 2019 | 7.75% | Arena average 16.11% | Apache-2.0 for the singing repo; speech repo **(unverified)** | Google Drive, HF | Not shortlisted |
| AASIST | 0.3M | ASVspoof 2019 LA | 43.00% | ML-ITW 41.85% | MIT **(unverified)** | GitHub | Optional teaching baseline |
| DF Arena 1B V1 | ~1B | ASVspoof 2019 and 5, CodecFake, MLAAD, SpoofCeleb and others | 0.91% | Arena average 5.92% | Custom non-commercial | HF, needs `trust_remote_code` | Excluded: runs remote code, too heavy |
| Commercial (Whispeak, Syntra, Resemble AI) | 0.1–2.1B | Undisclosed | 1.26–3.97% | Arena averages 3.05–10.69% | Proprietary | None | Context only |

Notes:

- **The commercial results can't be checked.** Their training data is undisclosed, so overlap with the test sets can't be ruled out.
- **DF Arena 1B is partly in-domain on its own benchmark.** Several of its training datasets (CodecFake, DFADD, LibriSeVoc, ASVspoof) are also Arena test datasets, in different splits.
- **Reported numbers are not stable.** The same AntiDeepfake Wav2Vec2-Large weights show 1.91% ITW EER on NII's model card and 2.89% on a bit-exact copy evaluated by a separate benchmark (SpeechAntiSpoofingBenchmarks). Preprocessing and protocol details move results by a full point, which is one more reason to measure ourselves.
- **XLSR-Mamba requires `mamba-ssm` and `causal-conv1d`,** CUDA-compiled kernels that are hard to install on Windows and don't help on a CPU-only host.
- **What AntiDeepfake trained on:** ASVspoof 2019, 2021 and 5, MLAAD, CodecFake, SpoofCeleb, WaveFake, DiffSSD and others. In-the-Wild and Deepfake-Eval-2024 were not used. Variants ending in `-nda` were trained without data augmentation.

## 3. Datasets

| Dataset | What it is | Size | License | Access | Use in squeaktest |
|---|---|---|---|---|---|
| [In-the-Wild](https://huggingface.co/datasets/mueller91/In-The-Wild) | Real and fake speech of 58 English-speaking public figures, from web videos. 37.9 h, 19,963 real and 11,816 fake clips | 8.16 GB | CC BY-SA 4.0 | Open (HF) | **Main held-out set** |
| [ASVspoof 2019 LA](https://datashare.ed.ac.uk/handle/10283/3336) | Lab TTS and voice conversion, clean audio. Has the silence artifact | 7.12 GB | **(unverified)**, license file not read | Open | **In-domain reference** |
| [ASVspoof 2021 LA eval](https://zenodo.org/records/4837263) | 2019-style attacks sent over telephony and VoIP channels | 7.8 GB | ODC-By 1.0 | Open; answer keys from asvspoof.org | Later, as a telephony check (AntiDeepfake trained on it) |
| [ASVspoof 2021 DF eval](https://zenodo.org/records/4835108) | Media codecs (MP3, M4A, OGG) | 34.5 GB | ODC family **(unverified variant)** | Open | Not needed |
| [ASVspoof 5](https://zenodo.org/records/14498691) | Crowdsourced, about 2,000 speakers, 7 adversarial attacks. The eval split has codec conditions including 8 kHz Opus, AMR, Speex and a PSTN chain | 142 GB (eval is about 93 GB in 11 shards of about 8.5 GB) | ODC-By 1.0 | Open | Later: one eval shard as a real phone-codec test for the lab-trained models |
| [Deepfake-Eval-2024](https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024) | Real deepfakes circulated in 2024: 56.5 h of audio, 52 languages, 88 websites | **(unverified)** | CC BY-SA 4.0 | Gated until May 2027 to people verifiably working on deepfake detection | Ideal second held-out set; request access |
| [ML-ITW](https://arxiv.org/abs/2603.05852) (2026) | Real circulated fakes from 7 platforms: 180 public figures, 14 languages, 28.39 h, 18,168 real and 6,361 fake | **(unverified)** | **(unverified)**; the HF page requires login | Gated | Candidate second held-out set |
| [MLAAD v8](https://huggingface.co/datasets/mueller91/MLAAD) | Multilingual TTS output | 183 GB | CC BY-NC 4.0 | Gated | Not needed (AntiDeepfake trained on it) |

Partition sizes for ASVspoof 5: train 183k utterances, dev 141k, eval 681k. Only the eval split has codec conditions.

## 4. Fit for squeaktest

### 4.1 The SOC use case

- **Where it sits in ATT&CK:** [T1566.004 Spearphishing Voice](https://attack.mitre.org/techniques/T1566/004/) (Scattered Spider posing as IT staff on phone calls is a listed procedure example) and [T1656 Impersonation](https://attack.mitre.org/techniques/T1656/).
- **It is happening now:** in [FBI PSA I-051525-PSA](https://www.ic3.gov/PSA/2025/PSA250515) (May 2025), the FBI warned of AI-generated voice messages impersonating senior US officials.
- **The FBI's main defense is a process control:** verify through a known phone number. squeaktest should be pitched as a triage and enrichment signal that supports that process, not as a replacement for call-back verification.
- **Phone audio is untested.** No candidate publishes held-out results for 8 kHz telephone audio. Re-encoding the In-the-Wild test split through phone codecs creates that evidence ourselves, and is a good portfolio piece.
- **Base rates decide whether alerts are useful.** Suppose 1 in 1,000 voicemails is a deepfake and the detector catches 90% of them:

| False-positive rate | Share of alerts that are real deepfakes | False alerts per true alert |
|---|---|---|
| 5% | 1.8% | ~55 |
| 1% | 8.3% | ~11 |
| 0.1% | 47% | ~1.1 |

So the operating point has to be chosen against an alert budget, and the band labels must not imply more certainty than this.

### 4.2 Free hosting

- **CPU basic** (free): 2 vCPU, 16 GB RAM.
- **ZeroGPU** (also free): personal accounts with a verified email that are more than 30 days old can host up to 2 ZeroGPU Spaces. Each visitor gets a daily GPU quota: 2 minutes unauthenticated, 5 minutes with a free account. The catch is that ZeroGPU works only with the Gradio SDK, not Docker Spaces, so we would lose control of the container (non-root, read-only filesystem, our own security headers). That is a Phase 4 decision.
- **Rough compute estimate (to measure in Phase 1):** a 300M wav2vec2-large-class model needs about 37 GFLOPs per second of audio. The 95M model needs about 13, and a 1B model about 100. On 2 vCPUs, a 300M model might take very roughly 0.5–1.5 seconds per second of audio, so 30–90 seconds for a 1-minute voicemail.
  - Mitigations: cap hosted clips at about 60 seconds (voicemails are usually shorter), int8 quantization, the 95M model, or ZeroGPU.
  - How the estimate works: the transformer costs about 24 layers × 12 × 1024² multiply-adds per frame at 50 frames per second, plus about 2.5 billion multiply-adds per second for the convolutional feature encoder.

### 4.3 Installing on Windows and Python 3.12

- **fairseq won't be a dependency.** Every SSL candidate's official code pins an old fairseq (0.12.2 or a 2021 commit) on Python 3.7–3.9, and fairseq is effectively unmaintained.
- **The plan:** load the front-end with Hugging Face `transformers` (`Wav2Vec2Model`, the same architecture) and port the weights. AntiDeepfake ships safetensors, and its back-end is just average pooling plus `Linear(1024, 2)`; output index 0 is fake and index 1 is real. TCM's conformer back-end is plain PyTorch, so it's a moderate port.
- **Verifying the port:** run the official code once (WSL2, Python 3.9, fairseq) on a handful of files and require matching outputs. Then check that our In-the-Wild EER lands near the published figure.
- **New dependency for sign-off:** `transformers`. It's a heavy dependency, which the brief says to ask about.
- **Avoid `trust_remote_code`.** Running a model repo's unreviewed code at load time is a supply-chain risk, which is one reason DF Arena 1B is excluded.

### 4.4 Audio decoding

- **torchaudio is out.** It entered maintenance mode in 2.9, and its `load`/`save` now forward to TorchCodec, so it's a poor choice for new code.
- **Proposal:** decode with `ffmpeg` in a subprocess, with a timeout, an explicit demuxer chosen from the sniffed file type, and protocols restricted to the input pipe. ffmpeg has had local-file-disclosure bugs triggered by crafted playlist files, and a subprocess also contains crashes. `soundfile` is an option for WAV and FLAC. To validate in Phase 1.
- **Silence:** trim leading and trailing silence, and mark windows without speech as "no speech" instead of scoring them (section 1).

## 5. Recommendation

### Initial model for Phase 1: AntiDeepfake MMS-300M

- It has the best reported out-of-domain results of any model that fits the free CPU tier.
- It was trained on the broadest data: 29 datasets, more than 100 languages, including codec-generated speech.
- Its multilingual front-end suits SOC traffic that isn't all English. It beats the English-only Wav2Vec2-Large on ADD2023 (Chinese, 7.93% vs 13.25%) and DeepVoice (2.35% vs 4.44%), but loses on In-the-Wild (2.90% vs 1.91%). The bake-off settles that with our own numbers.
- Its back-end is a single linear layer, which makes it the easiest to port and to explain.

Trade-offs:
- The CC BY-NC-SA 4.0 weights mean squeaktest.com must stay non-commercial, and any version we fine-tune inherits the same license. The MMS base model is itself CC BY-NC 4.0.
- Its 32.84% EER on Deepfake-Eval-2024 means real-world reliability is limited.

### Bake-off shortlist for Phase 2

1. **AntiDeepfake MMS-300M:** the initial model, multilingual.
2. **AntiDeepfake Wav2Vec2-Large:** best reported In-the-Wild result, English front-end. Same code path as 1.
3. **AntiDeepfake Wav2Vec2-Small:** 95M parameters, about 3× cheaper. This is the free-tier fallback. Same code path.
4. **TCM:** MIT license, trained only on ASVspoof 2019. It is the permissive fallback, and it shows how much training-data diversity buys.

Optional: AASIST (0.3M) as a teaching baseline that shows lab overfitting.

### Evaluation plan

- **Held-out set:** In-the-Wild, split by speaker so no speaker appears in both parts. About 20% of speakers form the dev split (calibration and band thresholds), and about 80% form the test split (reported numbers). Dev numbers are reported separately. Calibrating on the same collection is optimistic, and the README will say so.
- **Phone and codec conditions:** the In-the-Wild test split, re-encoded as G.711 μ-law at 8 kHz, AMR-NB, Opus at about 12 kbps, and MP3 at 32 kbps.
- **In-domain reference:** ASVspoof 2019 LA eval. It is in-domain for TCM. AntiDeepfake trained on it, so its numbers there are labeled "seen data".
- **Later:** Deepfake-Eval-2024 if access is granted, ML-ITW, and one ASVspoof 5 eval shard for real phone codecs.

### Downloads this implies

These all go to `SQUEAKTEST_DATA_DIR`, and I will ask before each one:

- In-the-Wild: 8.16 GB
- ASVspoof 2019 LA: 7.12 GB
- AntiDeepfake MMS-300M and Wav2Vec2-Large: about 1.3 GB each
- AntiDeepfake Wav2Vec2-Small: about 0.4 GB (estimated from parameter count)
- TCM: about 1.3 GB (estimated)

Including extracted copies, plan for about 35 GB in total.

## 6. Decisions (signed off by Dylan, 2026-09-30)

1. **Non-commercial weights:** accepted. squeaktest.com stays non-commercial. (The rejected alternative was TCM or Wav2Vec2-AASIST under MIT, with weaker reported results: 7.8–11% EER on In-the-Wild.)
2. **Shortlist and evaluation plan:** approved as in section 5.
3. **New dependency:** approved. Models are ported to `transformers`; no fairseq.
4. **Deepfake-Eval-2024:** deferred. Revisit later.
5. **Hosting:** still open until Phase 4: CPU basic with a cap of about 60 seconds, or free ZeroGPU without the Docker hardening.

## 7. Still to verify

- ASVspoof 2019 license text (`LICENSE_text.txt`)
- The exact ODC license variant for ASVspoof 2021 DF
- Which ASVspoof 5 partitions AntiDeepfake trained on
- ML-ITW license and access
- AASIST license, and the license of Nes2Net's speech repository
- That In-the-Wild's metadata includes speaker names, needed for the speaker-disjoint split
- CPU latency (an estimate only, measured in Phase 1)
- Whether the `soundfile` wheels for Windows decode MP3

## Sources

Models and benchmarks
- Speech DF Arena paper: https://arxiv.org/abs/2509.02859 (HTML: https://arxiv.org/html/2509.02859)
- Speech DF Arena toolkit: https://github.com/Speech-Arena/speech_df_arena
- AntiDeepfake code: https://github.com/nii-yamagishilab/AntiDeepfake
- AntiDeepfake paper: https://arxiv.org/abs/2506.21090
- AntiDeepfake model cards: https://huggingface.co/nii-yamagishilab/mms-300m-anti-deepfake, https://huggingface.co/nii-yamagishilab/wav2vec-large-anti-deepfake, https://huggingface.co/nii-yamagishilab/wav2vec-small-anti-deepfake, https://huggingface.co/nii-yamagishilab/mms-1b-anti-deepfake
- AntiDeepfake collection: https://huggingface.co/collections/nii-yamagishilab/antideepfake-685a1788fc514998e841cdfc
- Arena copy of Wav2Vec2-Large AntiDeepfake: https://huggingface.co/SpeechAntiSpoofingBenchmarks/Wav2Vec2-Large-AntiDeepfake
- MMS-300m base model: https://huggingface.co/facebook/mms-300m
- SSL_Anti-spoofing (Wav2Vec2-AASIST): https://github.com/TakHemlata/SSL_Anti-spoofing
- XLSR+SLS: https://github.com/QiShanZhang/SLSforADD
- TCM: https://github.com/ductuantruong/tcm_add
- XLSR-Mamba: https://github.com/swagshaw/XLSR-Mamba, https://huggingface.co/AustinXiao/XLSR-Mamba-DF
- Nes2Net: https://github.com/Liu-Tianchi/Nes2Net
- DF Arena 1B: https://huggingface.co/Speech-Arena-2025/DF_Arena_1B_V_1

Datasets
- In-the-Wild: https://huggingface.co/datasets/mueller91/In-The-Wild
- ASVspoof 2019: https://datashare.ed.ac.uk/handle/10283/3336
- ASVspoof 2021 LA eval: https://zenodo.org/records/4837263
- ASVspoof 2021 DF eval: https://zenodo.org/records/4835108
- ASVspoof 5: https://zenodo.org/records/14498691; paper: https://arxiv.org/html/2502.08857v2
- Deepfake-Eval-2024: https://arxiv.org/abs/2503.02857, https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024
- ML-ITW: https://arxiv.org/abs/2603.05852
- MLAAD: https://huggingface.co/datasets/mueller91/MLAAD

Background
- Does Audio Deepfake Detection Generalize?: https://www.isca-archive.org/interspeech_2022/muller22_interspeech.pdf
- Speech is Silver, Silence is Golden: https://arxiv.org/pdf/2106.12914
- MITRE ATT&CK T1566.004: https://attack.mitre.org/techniques/T1566/004/
- MITRE ATT&CK T1656: https://attack.mitre.org/techniques/T1656/
- FBI PSA I-051525-PSA: https://www.ic3.gov/PSA/2025/PSA250515
- Hugging Face ZeroGPU: https://huggingface.co/docs/hub/en/spaces-zerogpu
- Hugging Face Spaces overview: https://huggingface.co/docs/hub/en/spaces-overview
- TorchAudio's future: https://github.com/pytorch/audio/issues/3902, https://docs.pytorch.org/audio/main/torchaudio.html
