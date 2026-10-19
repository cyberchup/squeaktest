# squeaktest web app: one image for local use (`docker compose up`) and self-hosting.
#
# Build argument WEIGHTS:
#   included (default)  The model is downloaded at build time and its SHA-256 checked, so the
#                       running container never needs the network or a mounted folder.
#   mounted             No model in the image. Mount a folder holding models/ at
#                       /opt/squeaktest-data/models, read-only (compose.yaml does this).
#
# Base images are pinned by digest, so a moved tag can't change what gets built. ffmpeg comes
# from Debian's security-maintained packages: rebuild regularly to pick up its fixes
# (docs/decisions.md D26).

ARG WEIGHTS=included

# Python 3.12.15 on Debian 13 (trixie); the digest is the multi-architecture index.
FROM python:3.12-slim-trixie@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS python-base


FROM python-base AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.21@sha256:a7aed3216253ee804de3e2d8afa5073baa1a177335345d43845cd4165e43b711 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /src
# Dependencies first, so a code change doesn't reinstall PyTorch. --locked installs exactly
# what uv.lock pins (CPU-only PyTorch on Linux); dev and eval tools are left out.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project --no-editable
COPY README.md LICENSE ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable


FROM builder AS weights-included
# fetch-model checks the file's size and SHA-256 against the pinned values before finishing.
RUN SQUEAKTEST_DATA_DIR=/opt/squeaktest-data /app/.venv/bin/squeaktest fetch-model --yes \
 && rm -rf /opt/squeaktest-data/models/*/.cache

FROM builder AS weights-mounted
RUN mkdir -p /opt/squeaktest-data/models

FROM weights-${WEIGHTS} AS weights


FROM python-base AS runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 user
# Copied as root and left that way: the app runs as UID 1000 (also what Hugging Face needs) and
# can read its code and model but not change them. It only writes temporary files under /tmp.
COPY --from=builder /app/.venv /app/.venv
COPY --from=weights /opt/squeaktest-data /opt/squeaktest-data
ENV PATH=/app/.venv/bin:$PATH \
    SQUEAKTEST_DATA_DIR=/opt/squeaktest-data \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/huggingface \
    XDG_CACHE_HOME=/tmp/cache \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
USER 1000
WORKDIR /home/user
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/healthz', timeout=4)"]
CMD ["squeaktest", "serve", "--host", "0.0.0.0", "--port", "7860"]
