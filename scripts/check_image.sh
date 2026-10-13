#!/bin/sh
# Smoke-test a squeaktest image built with WEIGHTS=mounted (no model inside).
#
#   scripts/check_image.sh [engine] [image]      defaults: docker squeaktest:local
#
# Checks the hardening the image itself provides: it runs as UID 1000, can't change its own
# code or model folder, works on a read-only filesystem with no capabilities, and refuses to
# serve without a model instead of crashing.
set -eu

ENGINE="${1:-docker}"
IMAGE="${2:-squeaktest:local}"
LOCKED="--read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges"

fail() { echo "FAIL: $*" >&2; exit 1; }
run() { "$ENGINE" run --rm "$@"; }

uid="$(run "$IMAGE" id -u)"
[ "$uid" = "1000" ] || fail "runs as UID $uid, expected 1000"
echo "ok: runs as UID 1000"

for path in /app/.venv /opt/squeaktest-data/models; do
  if run "$IMAGE" sh -c "touch $path/probe" 2>/dev/null; then
    fail "$path is writable by the app"
  fi
done
echo "ok: code and model folder are read-only to the app"

# shellcheck disable=SC2086  # LOCKED is a list of flags
version="$(run $LOCKED "$IMAGE" squeaktest --version)"
echo "ok: starts on a read-only filesystem without capabilities ($version)"

run "$IMAGE" ffmpeg -hide_banner -version | head -n 1

# Without a model, serve must exit with the setup error code (2), not a traceback.
set +e
# shellcheck disable=SC2086
output="$(run $LOCKED "$IMAGE" 2>&1)"
code=$?
set -e
[ "$code" = "2" ] || { echo "$output" >&2; fail "serve without a model exited $code, expected 2"; }
echo "$output" | grep -q "weights not found" || fail "unexpected message: $output"
echo "ok: refuses to serve without a model"
