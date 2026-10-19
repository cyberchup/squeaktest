#!/bin/sh
# Time the web app on a 60-second clip, the hosted demo's longest, with 2 CPUs.
#
#   scripts/measure_latency.sh [engine] [image] [requests]    defaults: docker squeaktest:local 4
#
# The image must have the model inside (WEIGHTS=included). It runs with compose.yaml's limits:
# 2 CPUs, 6 GB of memory, a read-only filesystem and no capabilities. The clip is a synthetic
# buzz that never falls silent, so every window is scored: the worst case for its length. The
# first request pays one-time setup costs, so it's reported apart from the rest. Needs curl
# and jq. Writes a Markdown summary to $GITHUB_STEP_SUMMARY when that is set.
set -eu

ENGINE="${1:-docker}"
IMAGE="${2:-squeaktest:local}"
REQUESTS="${3:-4}"
CLIP_S=60
PORT=8000
NAME="squeaktest-latency-$$"
WORK="$(mktemp -d)"

fail() { echo "FAIL: $*" >&2; exit 1; }
cleanup() { "$ENGINE" rm -f "$NAME" >/dev/null 2>&1 || true; rm -rf "$WORK"; }
trap cleanup EXIT

# 150 Hz with two harmonics, its loudness swinging at a syllable-like 4 Hz without reaching zero.
# The image's own ffmpeg makes it, so the host needs none.
signal="0.2*(sin(2*PI*150*t)+sin(2*PI*300*t)/2+sin(2*PI*450*t)/3)*(0.6+0.4*sin(2*PI*4*t))"
"$ENGINE" run --rm "$IMAGE" sh -c "ffmpeg -hide_banner -loglevel error \
  -f lavfi -i 'aevalsrc=$signal:s=16000:d=$CLIP_S' -c:a pcm_s16le /tmp/clip.wav \
  && cat /tmp/clip.wav" > "$WORK/clip.wav"

"$ENGINE" run -d --name "$NAME" \
  --read-only --tmpfs /tmp:size=256m,mode=1777 --cap-drop ALL \
  --security-opt no-new-privileges --cpus 2 --memory 6g --pids-limit 256 \
  -e OMP_NUM_THREADS=2 -e SQUEAKTEST_MAX_DURATION_S="$CLIP_S" \
  -p "127.0.0.1:$PORT:7860" "$IMAGE" > /dev/null

# Startup covers importing PyTorch, checking the model's SHA-256 and loading it.
started="$(date +%s)"
until curl -fsS "http://127.0.0.1:$PORT/healthz" > /dev/null 2>&1; do
  if [ "$("$ENGINE" inspect -f '{{.State.Running}}' "$NAME")" != "true" ]; then
    "$ENGINE" logs "$NAME" >&2
    fail "the container exited during startup"
  fi
  if [ "$(($(date +%s) - started))" -ge 300 ]; then
    "$ENGINE" logs "$NAME" >&2
    fail "not ready after 300 s"
  fi
  sleep 1
done
startup_s="$(($(date +%s) - started))"

times=""
i=1
while [ "$i" -le "$REQUESTS" ]; do
  out="$(curl -sS -o "$WORK/response.json" -w '%{http_code} %{time_total}' \
    --data-binary @"$WORK/clip.wav" -H "Content-Type: application/octet-stream" \
    "http://127.0.0.1:$PORT/api/analyze")"
  code="${out% *}"
  [ "$code" = "200" ] || { cat "$WORK/response.json" >&2; fail "request $i got HTTP $code"; }
  times="$times ${out#* }"
  i=$((i + 1))
done

# Every response is for the same clip, so the last one stands for all of them.
read -r audio_s windows scored <<EOF
$(jq -r '"\(.audio.duration_s) \(.windows | length) \([.windows[] | select(.score != null)] | length)"' "$WORK/response.json")
EOF
first_s="$(echo "$times" | awk '{ printf "%.1f", $1 }')"
warm_s="$(echo "$times" | awk '{ for (i = 2; i <= NF; i++) print $i }' | sort -n | awk '
  { a[NR] = $1 }
  END { if (NR == 0) print "n/a"; else if (NR % 2) printf "%.1f", a[(NR + 1) / 2];
        else printf "%.1f", (a[NR / 2] + a[NR / 2 + 1]) / 2 }')"
all_s="$(echo "$times" | awk '{ for (i = 1; i <= NF; i++) printf "%s%.1f", (i > 1 ? ", " : ""), $i }')"

# Read before anything else runs in the container, so it's the server's own peak.
memory="$("$ENGINE" exec "$NAME" cat /sys/fs/cgroup/memory.peak 2> /dev/null \
  | awk '{ printf "%.2f GB", $1 / 1e9 }')" || true
threads="$("$ENGINE" exec "$NAME" python -c "import torch; print(torch.get_num_threads())")"
image_gb="$("$ENGINE" image inspect -f '{{.Size}}' "$IMAGE" | awk '{ printf "%.2f", $1 / 1e9 }')"
cpu="$(grep -m 1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')"

summary="$(cat <<EOF
| Measure | Value |
|---|---|
| Host CPU | $cpu ($(nproc) visible to the host; container limited to 2) |
| PyTorch threads in the container | $threads |
| Image size | $image_gb GB |
| Startup, until /healthz answers | $startup_s s |
| Clip | $audio_s s of synthetic buzz, $scored of $windows windows scored |
| First request | $first_s s |
| Later requests, median | $warm_s s |
| All requests | $all_s s |
| Peak container memory | ${memory:-n/a} |
EOF
)"
echo "$summary"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  printf '## Latency on 2 CPUs\n\n%s\n' "$summary" >> "$GITHUB_STEP_SUMMARY"
fi

echo
echo "Server log lines for the analyses (metadata only):"
"$ENGINE" logs "$NAME" 2>&1 | grep '"event": "analysis"' || true
