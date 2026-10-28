# squeaktest

![squeaktest: deepfake voice detector](docs/assets/banner.jpg)

> Real curds squeak. Fakes don't.

**squeaktest** estimates whether speech in an audio clip is synthetic (AI-generated or voice-cloned). Instead of a yes/no verdict, it gives a score and one of three bands (*Likely synthetic*, *Uncertain*, *Likely genuine*), and shows which parts of the clip look suspicious. Its accuracy is measured on audio and generators it wasn't trained on, and all results are published, including the weak ones. Voice counts as biometric data, so audio is never kept or logged ([How your audio is handled](#how-your-audio-is-handled)). It runs locally as a command-line tool or a small web app, directly or in Docker. There's no hosted demo: it runs wherever you run it.

## Status

Early development. The command-line tool works (Phase 1), and its accuracy has been measured on data it was never trained or tuned on (Phase 2): see [Results](#results) and [Known gaps](#known-gaps). The web app and its hardened Docker image are done (Phase 3): CI builds the image, runs it with `docker compose up` and checks it on every push. There's no hosted demo for now ([D27](docs/decisions.md)); it's self-hosted, and the [threat model](docs/threat-model.md) covers running it on a home network.

The research behind the model and dataset choices is in [docs/research.md](docs/research.md). Every design choice, with its effect on missed fakes and false positives, is in [docs/decisions.md](docs/decisions.md). What the model is for, and what it isn't, is in the [model card](docs/model-card.md). What could go wrong with the web app, what stops it, and what's still open is in the [threat model](docs/threat-model.md).

## Usage

```bash
uv run squeaktest fetch-model             # one-time: download the model (1.27 GB, asks first)
uv run squeaktest analyze voicemail.m4a   # human-readable result and timeline
uv run squeaktest analyze voicemail.m4a --json
```

The result is a band (*Likely synthetic*, *Uncertain*, *Likely genuine*, or *Not assessed* when
there isn't enough sound), a score from 0 to 1, and a per-window timeline showing where the
suspicious parts are. The score is the model's raw output, not a probability; the band is what
to act on ([D16](docs/decisions.md), [D18](docs/decisions.md)). Opus files, common for voice
notes, get a warning, because Opus is the model's weakest condition ([K2](#known-gaps)). Output
never includes the file's name, so it is safe to log.

Exit codes: `0` analysis finished, `1` input rejected, `2` setup problem (for example, the
model hasn't been downloaded).

Speed, measured on one machine: about 1 second per minute of audio on an RTX 3050 GPU, and
15-28 seconds on CPU ([D15](docs/decisions.md), [D17](docs/decisions.md)). The Docker image,
limited to 2 CPUs on a GitHub runner, takes about 21 seconds for a 60-second clip and peaks at
2.8 GB of memory ([D27](docs/decisions.md)).

### Web app

```bash
uv run squeaktest serve                   # then open http://127.0.0.1:8000
```

A single page: upload a file, get the band, score and timeline. It listens on this machine
only unless you pass `--host`. If you do, list the name or address people will use in
`SQUEAKTEST_ALLOWED_HOSTS`. The server answers only to `localhost` and the names listed there,
so a website can't use DNS rebinding to reach it through your browser. The same analysis is
available as an API, which takes the file as the raw request body:

```bash
curl --data-binary @voicemail.m4a -H "Content-Type: application/octet-stream" http://127.0.0.1:8000/api/analyze
```

The response is the same JSON as `analyze --json`. The server scores one clip at a time and
limits each client to one analysis at a time and 6 requests a minute
([D25](docs/decisions.md)). Settings, all optional:

| Variable | Default | Meaning |
|---|---|---|
| `SQUEAKTEST_MAX_FILE_MB` | 25 | Largest accepted file |
| `SQUEAKTEST_MAX_DURATION_S` | 300 | Longest accepted audio |
| `SQUEAKTEST_RATE_LIMIT_PER_MINUTE` | 6 | Requests per client per minute |
| `SQUEAKTEST_MAX_WAITING` | 4 | Requests that may wait for the model; more are told the server is busy |
| `SQUEAKTEST_UPLOAD_TIMEOUT_S` | 60 | Longest an upload may take |
| `SQUEAKTEST_PROXY_HOPS` | 0 | Reverse proxies in front of the app; set it only behind one you control |
| `SQUEAKTEST_FRAME_ANCESTORS` | `'none'` | Sites allowed to show the page in a frame |
| `SQUEAKTEST_ALLOWED_HOSTS` | none | Host names or addresses the server answers to besides `localhost`, comma-separated (for example `squeaktest.lan,192.168.1.50`) |

### Docker

```bash
docker compose up                         # then open http://127.0.0.1:8000
```

This uses the model you downloaded with `fetch-model`: set `SQUEAKTEST_DATA_DIR` to that folder
first, and it's mounted read-only. The container runs as a non-root user with a read-only
filesystem, no Linux capabilities, and CPU and memory limits ([D26](docs/decisions.md)). It
restarts if it's killed, and its log is capped at 30 MB.

On a home server, to reach it from the rest of the LAN, put the settings in a `.env` file next
to `compose.yaml` (`.gitignore` keeps it out of git), with the server's own LAN address:

```dotenv
SQUEAKTEST_DATA_DIR=/srv/squeaktest-data
SQUEAKTEST_LISTEN_ADDRESS=192.168.1.50
SQUEAKTEST_ALLOWED_HOSTS=192.168.1.50,squeaktest.lan
```

Name the address rather than using `0.0.0.0`: Docker's published ports go around host firewalls
such as `ufw`, so `0.0.0.0` would reach every network the server is on
([D30](docs/decisions.md)). Without Python on the server, the image can download the model
itself (1.27 GB, checked against its SHA-256):

```bash
docker compose build
mkdir -p /srv/squeaktest-data/models
docker run --rm --user "$(id -u):$(id -g)" -e HF_HUB_OFFLINE=0 \
  -v /srv/squeaktest-data/models:/opt/squeaktest-data/models squeaktest:local squeaktest fetch-model --yes
```

Then `docker compose up -d --wait`, and `sh scripts/check_compose.sh` runs CI's checks against
it (below).

To build an image with the model inside, so nothing needs mounting, run this. Keep every flag:
they're the same protections `compose.yaml` sets, and a plain `docker run` has none of them.

```bash
docker build -t squeaktest .
docker run -d --name squeaktest -p 127.0.0.1:8000:7860 \
  --read-only --tmpfs /tmp:size=256m,mode=1777 --cap-drop ALL \
  --security-opt no-new-privileges --cpus 2 --memory 6g --pids-limit 256 \
  -e OMP_NUM_THREADS=2 --restart unless-stopped \
  --log-opt max-size=10m --log-opt max-file=3 squeaktest
```

On every push, CI builds the image and checks its hardening (`scripts/check_image.sh`), then
runs `docker compose up` with the model mounted (`scripts/check_compose.sh`). It checks that the
running container has every protection above and scores a clip. It also checks that the app
refuses other host names, comes back by itself after being killed, and on a LAN address answers
there and nowhere else. A workflow run by hand (`.github/workflows/latency.yml`) builds the
image with the model inside and times it under `compose.yaml`'s limits. *Not yet tested:* the
`docker run` command above as a whole (the latency workflow uses its hardening flags, but not
the restart and log options).

## Results

Measured by squeaktest on [In-the-Wild](https://huggingface.co/datasets/mueller91/In-The-Wild), real and deepfake speech of public figures that the model never saw in training. Only the 43 test speakers count here; the other 11 speakers were used for every choice (thresholds, minimum length), so these numbers weren't tuned on. The phone and compression rows re-encode the test clips the way a SOC would receive them.

| Audio | EER | Fakes caught at 1% false positives | Genuine rated *Likely synthetic* | Fakes rated *Likely genuine* |
|---|---|---|---|---|
| Clean | 1.3% | 98.3% | 0.1% | 1.8% |
| Landline (G.711, 8 kHz) | 2.3% | 96.1% | 0.1% | 3.5% |
| Mobile (AMR-NB) | 4.3% | 90.3% | 0.3% | 3.3% |
| Voice note or VoIP (Opus, 12 kbit/s) | 6.1% | 73.7% | **8.6%** | 1.5% |
| Compressed (MP3, 32 kbit/s) | 2.2% | 95.9% | 0.1% | 3.6% |

EER is the error rate where false alarms and missed fakes are equally common; lower is better. The rest of each flagged or cleared share falls in *Uncertain*.

**What an alert means.** The test set is 38% fake, but a real voicemail queue is almost all genuine. If 1 recording in 1,000 were fake, about 58% of *Likely synthetic* results on clean audio would be real fakes (95% interval 36-100%), and about 1% on Opus audio. Opus voice notes are the weak spot.

**Read this as a best case.** In-the-Wild was published in 2022. On deepfakes that circulated online in 2024 ([Deepfake-Eval-2024](https://arxiv.org/abs/2503.02857)), NII reports 33% EER for these same weights. squeaktest hasn't measured that set yet (access is gated).

Intervals, per-condition details and the other models are in the [model card](docs/model-card.md), the [full report](eval/results/antideepfake-w2v-large/report.md) and the [bake-off](eval/results/bakeoff.md).

## Known gaps

Measured weaknesses and untested areas. Details are in [docs/decisions.md](docs/decisions.md#known-gaps).

- **Opus voice notes and VoIP audio** (K2): 8.6% of genuine Opus clips are rated *Likely synthetic*. squeaktest warns when the file is Opus, but not when the audio was Opus before being converted.
- **Very short clips** (K3): 5.1% EER for clips under 2 seconds. Clips with under 1 second of sound aren't scored at all.
- **Tones, beeps and music** (K1): non-speech audio can score as confidently synthetic. Not yet measured or fixed.
- **Recent real-world deepfakes:** probably much worse than the table above (see *Read this as a best case*).
- **Languages other than English:** untested.
- **Untested:** partial fakes (a cloned sentence spliced into a real call), real phone networks (the codecs above are simulated), and deliberate evasion.

The model is shipped as NII published it, with these limits documented, rather than fine-tuned or replaced ([D20](docs/decisions.md)).

## How your audio is handled

A voice can identify a person, so squeaktest treats every recording as sensitive. This describes the command-line tool and the web app running on your own machine or server. There's no hosted demo, so no one else's servers are involved.

- **It stays on your machine.** Analysis runs locally, and squeaktest makes no network requests while analyzing. The only download is the model, and only when you run `fetch-model`.
- **It isn't kept.** To decode a file, squeaktest copies it into a new temporary folder under a fixed name (never your file's name), decodes it, and deletes the folder straight away, whether decoding worked or not. The decoded audio is held in memory and is gone when the command finishes. Your original file is only read, never changed. One exception: if the computer crashes or loses power mid-decode, the copy can be left in your temp folder (or in `SQUEAKTEST_TEMP_DIR`, if you set it).
- **It isn't logged.** squeaktest writes no log files. Results and error messages never include the file's name or any of its audio.
- **It's checked before it's decoded.** Files are identified by their content, not their extension. Only wav, mp3, flac, ogg and m4a files with expected audio codecs are accepted. Files over 25 MB or 5 minutes are rejected; set `SQUEAKTEST_MAX_FILE_MB` and `SQUEAKTEST_MAX_DURATION_S` to change that. Files with embedded images (such as cover art) or extra streams are rejected, and ffmpeg is only allowed to use the decoders for the accepted formats. Decoding runs in a separate ffmpeg process with network access turned off, and is stopped after 30 seconds (`SQUEAKTEST_DECODE_TIMEOUT_S`). Audio decoders are a common home for security bugs, so keep ffmpeg up to date.

The web app adds:

- **Only the audio is sent.** The page sends the file's bytes, never its name. The server holds them in memory, never in a file of its own, and then handles them exactly as above.
- **Logs are metadata only.** One line per request: the outcome, band, file size, audio length, codec and time taken. Never the audio, a file name, the score or your IP address. The web server's usual access log, which records addresses, is turned off.
- **Your IP address is used only for rate limiting**, held in memory, and forgotten within about two minutes of your last request ([D25](docs/decisions.md)).
- **No cookies, trackers or third-party requests.** The page loads nothing from other sites, and its Content Security Policy stops it from doing so. Results are marked never to be cached.

## Model

| Model | Source (pinned) | License | How it's checked |
|---|---|---|---|
| AntiDeepfake Wav2Vec2-Large, by NII's Yamagishi Lab (default) | [`nii-yamagishilab/wav2vec-large-anti-deepfake`](https://huggingface.co/nii-yamagishilab/wav2vec-large-anti-deepfake/tree/7ccfcbb27d3098dafeb67596d4bb26e51e4627fc) at commit `7ccfcbb` | CC BY-NC-SA 4.0 (non-commercial) | SHA-256 verified on every load. Outputs match NII's original code within 7e-6 (logits) |
| AntiDeepfake MMS-300M (evaluation only) | [`nii-yamagishilab/mms-300m-anti-deepfake`](https://huggingface.co/nii-yamagishilab/mms-300m-anti-deepfake/tree/7928e119e09f616b2ce698892e00ca22dbaa40bf) at commit `7928e11` | CC BY-NC-SA 4.0 (non-commercial) | Same checks; within 4e-5 |
| AntiDeepfake Wav2Vec2-Small (evaluation only) | [`nii-yamagishilab/wav2vec-small-anti-deepfake`](https://huggingface.co/nii-yamagishilab/wav2vec-small-anti-deepfake/tree/9a13264b5dcc827a8d5a4f8e01fccefa392f886b) at commit `9a13264` | CC BY-NC-SA 4.0 (non-commercial) | Same checks; within 7e-6 |

squeaktest runs the models with Hugging Face `transformers` rather than NII's fairseq-based code; [docs/decisions.md](docs/decisions.md) (D8) explains why and how the two were compared. Wav2Vec2-Large became the default in the Phase 2 bake-off ([D17](docs/decisions.md), [eval/results/bakeoff.md](eval/results/bakeoff.md)). Weights are downloaded on request into `SQUEAKTEST_DATA_DIR`, never committed.

The evaluation data: [In-the-Wild](https://huggingface.co/datasets/mueller91/In-The-Wild) (CC BY-SA 4.0), and [ASVspoof 2019 LA](https://datashare.ed.ac.uk/handle/10283/3336) through a Hugging Face mirror checked file by file against the official archive (`scripts/verify_asvspoof_mirror.py`). Both are fetched by script, never committed.

## Development

You need [uv](https://docs.astral.sh/uv/) and [ffmpeg](https://ffmpeg.org/) (including `ffprobe`) on your PATH.

```bash
uv sync          # create the virtual environment from uv.lock
uv run pytest    # run the tests
```

Test audio is synthetic (tones generated by ffmpeg at test time). The repository never contains real voice recordings.

## License

Code: [MIT](LICENSE). The model weights are licensed separately by NII for non-commercial use only (CC BY-NC-SA 4.0), so squeaktest as a whole is a non-commercial project.
