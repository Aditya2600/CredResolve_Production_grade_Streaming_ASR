# syntax=docker/dockerfile:1.7
# itn_service runtime image.
#
# Stage 1 (builder): installs build toolchain, builds the Pynini wheel
#   against the OpenFst headers (pip wheel of pynini requires this), and
#   generates the proto stubs. Build context is the itn_service/
#   directory of the monorepo (see `docker build -f Dockerfile .` below).
#
# Stage 2 (runtime): slim Python image. Installs only the runtime
#   wheels (grpcio, pydantic, pyyaml, pyicu, pynini, regex) and copies
#   the service code plus FAR archives compiled in the builder stage.
#   The image deliberately omits build tooling from the runtime layer so
#   a startup-time FAR compile is impossible: `compile.py` itself is
#   copied so engineers can audit the FARs that were baked in, but the
#   entrypoint refuses to start the server unless the FAR directory is
#   populated.
#
# Usage:
#   docker build -f itn_service/Dockerfile -t itn-service:$GIT_SHA itn_service/

# =============================================================================
# Stage 1: builder
# =============================================================================
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# pynini needs g++ + OpenFst headers; PyICU needs libicu-dev + pkg-config.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        libicu-dev \
        libfst-dev \
        pkg-config

WORKDIR /build

# Install runtime dependencies into a venv we can copy to the final
# image — keeps the runtime image free of build tooling.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install in two passes for layer caching: heavy native wheels first,
# then the rest of the project on every code change.
COPY pyproject.toml /build/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel \
    && pip install \
        "grpcio>=1.62" \
        "grpcio-tools>=1.62" \
        "pydantic>=2.6,<3" \
        "pyyaml>=6.0" \
        "PyICU>=2.13" \
        "regex>=2024.5.15" \
        "pynini>=2.1.6"

# Copy the source needed to generate proto stubs. The full source is
# brought in by the runtime stage; here we only need service/.
COPY service /build/itn_service/service
RUN bash /build/itn_service/service/build_protos.sh

# Compile grammar FARs at image build time. Runtime startup must only
# load these artifacts; it must never construct WFST graphs.
COPY __init__.py /build/itn_service/__init__.py
COPY compile.py /build/itn_service/compile.py
COPY grammars /build/itn_service/grammars
ENV PYTHONPATH=/build
RUN python -m itn_service.compile --out-dir /build/compiled_grammars \
 && test -s /build/compiled_grammars/hi.far \
 && test -s /build/compiled_grammars/mr.far

# =============================================================================
# Stage 2: runtime
# =============================================================================
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# libicu and libfst runtime shared libs (no -dev headers needed here).
# Use exact-major library packages so apt doesn't get tempted to pull
# in toolchains.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libicu72 \
        libfst22

# Bring the prepared venv over from the builder.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Source layout: install the package at /app/itn_service so the
# ``itn_service.*`` imports resolve as in development. We deliberately
# do not run ``pip install -e .`` — the venv is already complete and
# pip's editable install would reintroduce setuptools at runtime.
COPY runtime         /app/itn_service/runtime
COPY grammars        /app/itn_service/grammars
COPY configs         /app/itn_service/configs
COPY __init__.py     /app/itn_service/__init__.py
COPY compile.py      /app/itn_service/compile.py
COPY conftest.py     /app/itn_service/conftest.py

# Generated proto stubs (and service code).
COPY --from=builder /build/itn_service/service /app/itn_service/service

ENV PYTHONPATH=/app

# ---------------------------------------------------------------------------
# Pre-compiled FAR artifacts. The FARs MUST be present at image build
# time; the entrypoint refuses to start when the directory is empty.
# This enforces the invariant "image does not compile FARs at startup".
# ---------------------------------------------------------------------------
COPY --from=builder /build/compiled_grammars /app/itn_service/compiled_grammars

# Strip executable bits off compile.py so a stray `python compile.py`
# in the container looks intentional, not accidental. The file remains
# importable for audit / debugging.
RUN chmod -x /app/itn_service/compile.py

# ---------------------------------------------------------------------------
# Entrypoint: guard-rails + server launch.
# ---------------------------------------------------------------------------
COPY <<'EOF' /usr/local/bin/itn-entrypoint.sh
#!/bin/sh
set -e

FAR_DIR=/app/itn_service/compiled_grammars
if [ ! -d "$FAR_DIR" ] || [ -z "$(ls -A "$FAR_DIR"/*.far 2>/dev/null)" ]; then
    echo "FATAL: $FAR_DIR is missing or contains no .far files." >&2
    echo "       FARs must be baked into the image at build time." >&2
    echo "       The image MUST NOT compile FARs at startup." >&2
    exit 78  # EX_CONFIG
fi

echo "ITN: serving with FARs:"
ls -l "$FAR_DIR"/*.far

exec python -m itn_service.service.grpc_server "$@"
EOF
RUN chmod +x /usr/local/bin/itn-entrypoint.sh

EXPOSE 50051

# Tiny non-root user. The server only needs to read its own code +
# FARs; no shell, no package management.
RUN groupadd -r itn && useradd -r -g itn -d /app -s /sbin/nologin itn \
 && chown -R itn:itn /app
USER itn

ENTRYPOINT ["/usr/local/bin/itn-entrypoint.sh"]
CMD ["--bind", "[::]:50051"]
