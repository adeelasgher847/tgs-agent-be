# syntax=docker/dockerfile:1
#
# Multi-stage build: compile in a full Debian userland, run on Google's
# distroless image (no shell, no package manager, minimal CVE surface).
#
# Builder is pinned to python:3.11-slim — NOT 3.12 — because
# gcr.io/distroless/python3-debian12 bundles CPython 3.11.2 (Debian 12's
# default python3). Compiled C-extension wheels (psycopg2-binary, asyncpg,
# pymupdf, bcrypt, greenlet, ...) are tagged to a specific CPython ABI;
# building them under 3.12 would make them fail to import under the
# distroless runtime's 3.11 interpreter. Verified against the actual
# distroless image (`python3 --version` -> 3.11.2) before picking this.

# ---------------------------------------------------------------------------
# Stage 1: builder — compile wheels that need gcc/libpq-dev
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: ffmpeg — used by LiveKitAudioProcessor to resample non-16kHz PCM
# for Google STT. distroless has no apt, so fetch the BtbN/FFmpeg-Builds GPL
# release (codecs linked in statically; only base glibc is dynamic — verified
# via `ldd`: libc/libm/libdl/librt/libpthread/libmvec/libgcc_s, all of which
# the distroless/python3-debian12 base already ships). Do NOT copy any .so
# files here — copying glibc from a python:slim builder over distroless's
# own glibc causes a GLIBC_PRIVATE symbol mismatch at runtime (verified by
# hitting exactly that failure with the previous ldd-closure-copy approach).
# ---------------------------------------------------------------------------
FROM debian:12-slim AS ffmpeg-builder

ARG TARGETARCH

RUN apt-get update && apt-get install -y --no-install-recommends curl xz-utils ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    case "$TARGETARCH" in \
        amd64) FF_ARCH=linux64 ;; \
        arm64) FF_ARCH=linuxarm64 ;; \
        *) echo "unsupported arch: $TARGETARCH" >&2; exit 1 ;; \
    esac; \
    curl -fL "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-${FF_ARCH}-gpl.tar.xz" -o /tmp/ffmpeg.tar.xz; \
    tar -xJf /tmp/ffmpeg.tar.xz -C /tmp; \
    mkdir -p /dist-root/usr/bin; \
    cp "/tmp/ffmpeg-master-latest-${FF_ARCH}-gpl/bin/ffmpeg" /dist-root/usr/bin/ffmpeg

# ---------------------------------------------------------------------------
# Stage 3: runtime — distroless, non-root
# ---------------------------------------------------------------------------
FROM gcr.io/distroless/python3-debian12:nonroot AS runtime

WORKDIR /app

ENV PYTHONPATH=/usr/local/lib/python3.11/site-packages \
    PATH=/usr/local/bin:/usr/bin:/bin \
    PYTHONUNBUFFERED=1

COPY --from=builder /install /usr/local
COPY --from=ffmpeg-builder /dist-root/ /
COPY --chown=nonroot:nonroot . /app

EXPOSE 8001

ENTRYPOINT ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
