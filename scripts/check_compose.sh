#!/bin/sh
# Check a running `docker compose up`: that Docker gave the container compose.yaml's
# protections, that the app answers on the listen address and only there, and that it scores a
# clip with the mounted model.
#
#   scripts/check_compose.sh [--restart]
#
# Run it from the repository root after `docker compose up -d --wait`, with the same
# SQUEAKTEST_LISTEN_ADDRESS and SQUEAKTEST_ALLOWED_HOSTS. --restart also kills the server the way
# the out-of-memory killer would and checks that it comes back by itself (docs/threat-model.md
# T12); that needs root or passwordless sudo. Needs curl and jq.
set -eu

ADDRESS="${SQUEAKTEST_LISTEN_ADDRESS:-127.0.0.1}"
URL="http://$ADDRESS:8000"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

id="$(docker compose ps -q squeaktest)"
[ -n "$id" ] || fail "no squeaktest container; run docker compose up -d --wait first"

# What Docker applied to the running container, not just what compose.yaml asks for.
setting() {  # setting <what> <docker inspect template> <expected>
  got="$(docker inspect -f "$2" "$id")"
  [ "$got" = "$3" ] || fail "$1 is '$got', expected '$3'"
}
setting "user" '{{.Config.User}}' "1000:1000"
setting "read-only root" '{{.HostConfig.ReadonlyRootfs}}' "true"
setting "/tmp" '{{index .HostConfig.Tmpfs "/tmp"}}' "size=256m,mode=1777"
setting "dropped capabilities" '{{.HostConfig.CapDrop}}' "[ALL]"
setting "security options" '{{.HostConfig.SecurityOpt}}' "[no-new-privileges:true]"
setting "CPU limit" '{{.HostConfig.NanoCpus}}' "2000000000"
setting "memory limit" '{{.HostConfig.Memory}}' "6442450944"
setting "process limit" '{{.HostConfig.PidsLimit}}' "256"
setting "restart policy" '{{.HostConfig.RestartPolicy.Name}}' "unless-stopped"
setting "log driver" '{{.HostConfig.LogConfig.Type}}' "json-file"
setting "log file size" '{{index .HostConfig.LogConfig.Config "max-size"}}' "10m"
setting "log file count" '{{index .HostConfig.LogConfig.Config "max-file"}}' "3"
setting "model mount" \
  '{{range .Mounts}}{{if eq .Destination "/opt/squeaktest-data/models"}}{{.RW}}{{end}}{{end}}' \
  "false"
setting "published port" '{{json .HostConfig.PortBindings}}' \
  "{\"7860/tcp\":[{\"HostIp\":\"$ADDRESS\",\"HostPort\":\"8000\"}]}"
echo "ok: the container has compose.yaml's protections, published on $ADDRESS:8000"

code="$(curl -sS -o /dev/null -w '%{http_code}' "$URL/healthz")"
[ "$code" = "200" ] || fail "$URL/healthz answered $code"
curl -sS -D "$WORK/headers" -o /dev/null "$URL/"
grep -qi '^content-security-policy:' "$WORK/headers" || fail "the page has no Content Security Policy"
echo "ok: the page answers on $URL with its security headers"

if [ "$ADDRESS" != "127.0.0.1" ]; then
  if curl -sS -o /dev/null --max-time 5 "http://127.0.0.1:8000/healthz" 2> /dev/null; then
    fail "it also answers on 127.0.0.1, so it isn't published on $ADDRESS alone"
  fi
  echo "ok: not published on 127.0.0.1"
fi

# A rebound request carries the attacker's name in Host (T3).
code="$(curl -sS -o "$WORK/body" -w '%{http_code}' -H "Host: rebind.example" "$URL/healthz")"
[ "$code" = "400" ] && [ "$(jq -r .error "$WORK/body")" = "wrong_host" ] \
  || fail "a request for another host name got $code: $(cat "$WORK/body")"
echo "ok: refuses other host names"

# A 6-second synthetic buzz (as in measure_latency.sh), made by the image's own ffmpeg.
image="$(docker inspect -f '{{.Config.Image}}' "$id")"
signal="0.2*(sin(2*PI*150*t)+sin(2*PI*300*t)/2+sin(2*PI*450*t)/3)*(0.6+0.4*sin(2*PI*4*t))"
docker run --rm "$image" sh -c "ffmpeg -hide_banner -loglevel error \
  -f lavfi -i 'aevalsrc=$signal:s=16000:d=6' -c:a pcm_s16le /tmp/clip.wav \
  && cat /tmp/clip.wav" > "$WORK/clip.wav"
code="$(curl -sS -o "$WORK/result.json" -w '%{http_code}' --data-binary @"$WORK/clip.wav" \
  -H "Content-Type: application/octet-stream" "$URL/api/analyze")"
[ "$code" = "200" ] || fail "the analysis answered $code: $(cat "$WORK/result.json")"
band="$(jq -r .band "$WORK/result.json")"
[ -n "$band" ] && [ "$band" != "null" ] || fail "the analysis has no band: $(cat "$WORK/result.json")"
docker logs "$id" 2>&1 | grep -q '"event": "analysis", "outcome": "ok"' \
  || fail "the analysis left no log line"
echo "ok: scored a clip with the mounted model (band: $band) and logged it"

[ "${1:-}" = "--restart" ] || exit 0

# SIGKILL from outside, as the out-of-memory killer sends it. `docker kill` wouldn't do: Docker
# counts that as stopping the container by hand, and unless-stopped then leaves it down.
pid="$(docker inspect -f '{{.State.Pid}}' "$id")"
restarts="$(docker inspect -f '{{.RestartCount}}' "$id")"
kill -KILL "$pid" 2> /dev/null || sudo -n kill -KILL "$pid"
waited=0
until [ "$(docker inspect -f '{{.RestartCount}}' "$id")" -gt "$restarts" ] \
  && curl -fsS -o /dev/null "$URL/healthz" 2> /dev/null; do
  if [ "$waited" -ge 180 ]; then
    docker logs --tail 20 "$id" >&2
    fail "not back 180 s after being killed"
  fi
  sleep 2
  waited=$((waited + 2))
done
echo "ok: came back by itself after being killed (about $waited s)"
