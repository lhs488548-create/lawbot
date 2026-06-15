# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------- #
# lawbot API image — production build.
#
# Two stages:
#   1. builder  : resolve + install pinned deps into an isolated virtualenv
#                 using uv (fast, deterministic). Build tools stay out of the
#                 final image.
#   2. runtime  : python:3.12-slim + the prebuilt venv + app source, run as a
#                 non-root user under uvicorn.
#
# The app entrypoint is `api.main:app` (FastAPI), per _BUILD_CONTRACT.md (e).
# Secrets are injected at runtime via environment variables (see DEPLOY.md);
# no secret is ever baked into the image.
# ---------------------------------------------------------------------------- #

# ----------------------------- builder stage -------------------------------- #
FROM python:3.12-slim AS builder

# uv: standalone, reproducible installs. Copied from the official distroless
# image so we don't depend on pip bootstrap or network-fetched install scripts.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Create the venv the runtime will use, then install ONLY pinned requirements.
# Copying requirements.txt first keeps this layer cached unless deps change.
COPY requirements.txt ./
RUN uv venv /opt/venv --python 3.12 \
    && VIRTUAL_ENV=/opt/venv uv pip install --no-cache -r requirements.txt

# ----------------------------- runtime stage -------------------------------- #
FROM python:3.12-slim AS runtime

# Runtime-only OS deps:
#   - curl: container HEALTHCHECK hits /healthz.
#   - libgomp1 / libstdc++ pulled in transitively by some wheels; slim already
#     ships libstdc++. We keep the layer minimal.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user (defense in depth; never run web apps as root).
RUN groupadd --system --gid 10001 lawbot \
    && useradd --system --uid 10001 --gid lawbot --home-dir /app --no-create-home lawbot

ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    # Default to in-container Qdrant service name from docker-compose; override
    # with QDRANT_URL for Qdrant Cloud. Never set OPENAI_API_KEY here.
    QDRANT_URL="http://qdrant:6333"

WORKDIR /app

# Bring in the prebuilt virtualenv from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Copy application source. .dockerignore keeps secrets/artifacts/venv out.
# config.py, ingest/, embed/, search/, api/, eval/ are the runtime packages.
COPY --chown=lawbot:lawbot . .

# The SQLite key store + any runtime-writable paths must be writable by the
# non-root user. Mount a volume here in production (see docker-compose.yml).
RUN mkdir -p /app/data && chown -R lawbot:lawbot /app/data
ENV API_KEYS_DB="/app/data/lawbot_keys.db"

USER lawbot

EXPOSE 8000

# Liveness: the API exposes GET /healthz returning {"ok": true, ...}.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/healthz" || exit 1

# Single-process uvicorn. Scale horizontally with more containers/replicas
# rather than in-process workers (keeps per-key rate-limit state simple and
# lets the platform/Caddy load-balance). Override CMD to add --workers if the
# deployment uses a shared rate-limit backend (Redis).
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
