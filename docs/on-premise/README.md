# On-Premise / Self-Hosted Deployment Guide

Run the entire platform — API, real-time voice transport, database, queue,
and reverse proxy — inside your own cloud account or data centre with a
single command. Built for customers with data-sovereignty requirements who
need to bring their own LLM, TTS, and telephony providers instead of the
multi-tenant SaaS defaults.

## Prerequisites

- Docker Engine 24+ and the Docker Compose v2 plugin (`docker compose version`)
- 8 GB RAM, 4 CPU cores minimum (the FastAPI app, ARQ worker, LiveKit,
  Postgres, Redis, and nginx all run concurrently on one host)
- ~3 GB free disk for the pre-pulled images (more if call recordings are
  retained outside GCS)
- Outbound internet access to your chosen LLM/TTS/telephony vendor APIs
  (or a reachable internal endpoint if using an OpenAI-compatible gateway —
  see [BYO LLM](#byo-llm))
- Ports 80 and 443 free on the host (nginx-proxy). If you connect WebRTC/
  browser clients directly to LiveKit, also free TCP 7881 and UDP 50000-50100.

## Quick start

The packaged `.tar.gz` ships `docker-compose.yml` + `.env.example` +
`/scripts/` + `/docs/on-premise/` only — no source code — because
`fastapi-v2`'s and `nginx-proxy`'s images are pre-built and published to
Artifact Registry (semver-tagged) by CI. Pull and run:

```bash
tar xzf tgs-agent-onprem-<version>.tar.gz
cd tgs-agent-onprem-<version>

cp .env.example .env
# Edit .env: at minimum set SECRET_KEY, N8N_WEBHOOK_SECRET, POSTGRES_PASSWORD,
# ELEVENLABS_ENCRYPTION_KEY, LIVEKIT_API_SECRET, APP_IMAGE/NGINX_IMAGE (the
# Artifact Registry path:tag you were given), and your BYO provider
# credentials (see below).

docker compose pull
docker compose up -d

./scripts/health-check.sh
```

Running from a full source checkout instead (this repo, for development)
builds the images locally rather than pulling them — same compose file,
just add `--build`:

```bash
docker compose up --build -d
```

All six services (`postgres`, `redis`, `livekit`, `fastapi-v2`, `worker`,
`nginx-proxy`) report healthy within ~1-2 minutes on a machine with the
images already pulled/built — that download/build time is not counted
against startup.

The API is then reachable at `https://localhost/api/v1/` and
`https://localhost/api/v2/` (self-signed cert — see [TLS](#tls-certificate)).

To stop: `docker compose down` (add `-v` to also drop the Postgres/Redis
named volumes — this deletes all data).

## What's running

| Service | Image | Purpose |
|---|---|---|
| `nginx-proxy` | custom (built from `nginx/Dockerfile`, FROM `nginx:1.27-alpine`) | TLS termination + routing |
| `fastapi-v2` | custom (built from `Dockerfile`) | API — serves both `/api/v1` and `/api/v2` (one process) |
| `worker` | same image as `fastapi-v2`, different entrypoint | ARQ background jobs (batch calls, callbacks, KB ingestion) |
| `migrate` | same image, runs once and exits | `alembic upgrade head` before `fastapi-v2`/`worker` start |
| `livekit` | `livekit/livekit-server` (official, multi-arch) | Real-time voice transport |
| `postgres` | `pgvector/pgvector:pg16` | Primary database — needs the pgvector extension for KB/RAG embeddings (named volume `postgres_data`, not a bind mount) |
| `redis` | `redis:7-alpine` | ARQ queue + rate limiting |

`docker-compose.dev.yml` additionally layers in `pgadmin` and
`redis-commander` plus hot-reload for `fastapi-v2` — never loaded by
default:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build -d
```

## nginx routing

| Path | Upstream |
|---|---|
| `/api/v2/` | `fastapi-v2:8001` (M2M API — API key + `x-workspace-id`) |
| `/api/v1/` | `fastapi-v2:8001` (dashboard/JWT API — same process as v2) |
| `/livekit/` | `livekit:7880` (signaling only; RTP media is direct UDP) |
| `/` | `fastapi-v2:8001` (Twilio voice webhooks, `/health`, `/api/docs`) |

`/api/v1` and `/` route to the same container as `/api/v2` because this
codebase serves both API versions from a single FastAPI process
(`app/main.py`) — there is no separate v1 deployable to route to.

## BYO providers

Every BYO switch is a deployment-wide default read from `.env`
(`app/core/config.py`). They don't replace this codebase's existing
per-tenant DB-stored provider config — if an agent has its own provider
configured in the database, that still wins. These env vars exist for
on-premise customers who want one consistent provider for the whole
deployment without touching the dashboard.

### BYO LLM

| Variable | Values | Notes |
|---|---|---|
| `LLM_PROVIDER` | `google` \| `openai` \| `azure_openai` | `google` uses Gemini/Vertex AI (`GEMINI_API_KEY`) |
| `OPENAI_API_KEY` | — | Required for `openai`/`azure_openai` |
| `OPENAI_BASE_URL` | URL | Redirects the OpenAI SDK — Azure OpenAI resource endpoint, or any Ollama/vLLM/LiteLLM OpenAI-compatible server. Blank = real api.openai.com |
| `OPENAI_API_VERSION` | e.g. `2024-10-21` | Required when `LLM_PROVIDER=azure_openai` (Azure's REST API version) |

All OpenAI SDK client construction goes through `app.core.openai_client`,
so every call site (chat completions, embeddings, KB ingestion) honors these
three vars uniformly.

### BYO TTS

| Variable | Values | Notes |
|---|---|---|
| `TTS_PROVIDER` | `rime` \| `elevenlabs` | |
| `TTS_API_KEY` | — | Generic key, auto-applied to `RIME_API_KEY`/`ELEVENLABS_API_KEY` (whichever `TTS_PROVIDER` selects) if that specific var is blank |

### BYO telephony

| Variable | Values | Notes |
|---|---|---|
| `TELEPHONY_PROVIDER` | `twilio` \| `sip` | |
| `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` / `TWILIO_PHONE_NUMBER` | — | Required for `twilio` |
| `SIP_TRUNK_URI` / `SIP_USERNAME` / `SIP_PASSWORD` | — | Required for `sip` — register your trunk's numbers via `POST /api/v1/phone-numbers/external` (`app/services/phone_number_service.py::register_external_number`) after the stack is up |

## Secrets in self-hosted mode

Leave `GCP_PROJECT_ID` **blank**. `app/core/secret_manager.py` only calls
GCP Secret Manager when a project ID is set; with it blank, every secret
(Twilio, Rime, LiveKit, etc.) is read straight from `.env` regardless of
`ENVIRONMENT`. Set `ENVIRONMENT=production` anyway — it still controls other
production-only behavior unrelated to Secret Manager.

## TLS certificate

`nginx-proxy` ships with a self-signed certificate generated at image build
time (`nginx/Dockerfile`, `openssl req -x509 ...`, CN/SAN from
`NGINX_SERVER_NAME`). Browsers and `curl` without `-k`/`--insecure` will
reject it. To use a real certificate:

1. Get a cert+key for your domain (e.g. via your internal CA, or Let's
   Encrypt if the host is internet-reachable).
2. Replace the two `COPY --from=cert-builder` lines in `nginx/Dockerfile`
   with a mount, or simplest — bind-mount over the build output in
   `docker-compose.yml`:
   ```yaml
   nginx-proxy:
     volumes:
       - /path/to/your/fullchain.pem:/etc/nginx/ssl/self-signed.crt:ro
       - /path/to/your/privkey.pem:/etc/nginx/ssl/self-signed.key:ro
   ```
3. `docker compose up -d nginx-proxy` to pick it up.

## Health checks

```bash
./scripts/health-check.sh
```

Checks two layers: each service's own Docker Compose `healthcheck:` status,
then an end-to-end HTTP smoke test through `nginx-proxy`
(`/nginx-health`, `/api/v2/health`, `/health`). Exits non-zero if anything
is unhealthy. Override the smoke-test target with
`HEALTH_CHECK_BASE_URL=https://your-host ./scripts/health-check.sh`.

## Architecture decisions worth knowing about

A few choices made while building this package that deviate from (or
clarify ambiguity in) the original spec — documented here rather than
silently guessed:

- **Builder stage is `python:3.11-slim`, not `3.12`.** The distroless
  runtime (`gcr.io/distroless/python3-debian12`) bundles CPython 3.11.2
  specifically (verified directly against the image). Compiled C-extension
  wheels (psycopg2, asyncpg, bcrypt, pymupdf, greenlet, ...) are ABI-tagged
  to a CPython minor version — building under 3.12 would produce wheels
  that fail to import under the distroless runtime's 3.11 interpreter.
- **`async-timeout` was added to `requirements.txt`.** redis-py's async
  client needs `asyncio.timeout`, which has a known bug fixed only in
  Python ≥3.11.3 (redis-py checks for this explicitly and falls back to the
  `async_timeout` package otherwise). Distroless's bundled 3.11.2 sits just
  below that threshold, so the fallback package must be present even though
  nothing else in this codebase imports it directly.
- **ffmpeg is a static BtbN/FFmpeg-Builds binary, not `apt install ffmpeg`.**
  The Debian `ffmpeg` package pulls in dozens of optional shared libraries
  (X11, Pango, systemd, Kerberos, ...) that would all need to be replicated
  into the distroless filesystem. The static build only depends on base
  glibc components (`libc`, `libm`, `libpthread`, `libgcc_s`, ...) that
  distroless already ships, so no `.so` files need to be copied at all —
  only the `ffmpeg` binary itself. (Copying glibc itself from the
  `python:3.11-slim` builder over distroless's own glibc was tried first and
  broke the container with a `GLIBC_PRIVATE` symbol mismatch — don't do that.)
- **`nginx`/`postgres`/`redis`/`livekit` stay on official upstream images.**
  The "multi-stage build → distroless runtime" requirement is scoped to this
  repo's own Python code (`fastapi-v2`/`worker`, both built from the same
  `Dockerfile`). nginx gets a thin custom layer for the self-signed cert and
  routing config but keeps the hardened official `nginx:alpine` base rather
  than a from-source distroless rebuild.
- **A `migrate` one-off service runs `alembic upgrade head`.** distroless
  has no shell, so `alembic upgrade head && uvicorn ...` can't be chained in
  the image's `ENTRYPOINT`. `fastapi-v2` and `worker` both
  `depends_on: migrate: condition: service_completed_successfully` instead —
  this also avoids multiple replicas racing to migrate concurrently.
- **`.dockerignore` was added** (none existed before). Without it,
  `COPY . /app` in the original single-stage Dockerfile baked the real
  `.env` file — and everything in it — directly into the image layer. Found
  this while testing the new build; fixed it as part of this ticket since an
  on-premise customer's BYO credentials are exactly the kind of secret this
  must never leak into a distributable image.
- **LiveKit is configured entirely through environment variables**
  (`LIVEKIT_KEYS`, `LIVEKIT_PORT`, `LIVEKIT_RTC_*`) rather than a mounted
  YAML config file — `livekit-server --help-verbose` confirms every config
  field has an env var equivalent, which keeps the deployment fully
  declarative from `.env` alone.

## Troubleshooting FAQ

**`docker compose up` exits immediately / `migrate` keeps restarting.**
Check `docker compose logs migrate`. Usually `DATABASE_URL` doesn't match
the `POSTGRES_USER`/`POSTGRES_PASSWORD`/`POSTGRES_DB` you set, or Postgres
isn't healthy yet (it has a 5s/20-retry healthcheck — `migrate` waits on
`service_healthy` before running).

**`fastapi-v2` is unhealthy but logs show it started.** The healthcheck
hits `http://127.0.0.1:8001/api/v2/health`, which itself checks DB
connectivity (`app/api/v2/routers/health.py`). If Postgres is healthy but
this still fails, check `DATABASE_URL` uses the async driver scheme
(`postgresql+asyncpg://` is NOT required here — the v2 health route uses
`get_async_db`, which is wired separately in `app/db/async_session.py`; if
you changed `DATABASE_URL`, confirm `init_async_db()` picked it up).

**LiveKit calls connect but no audio.** If you're connecting WebRTC clients
directly to this LiveKit instance from outside the Docker host's network
(not just bridging Twilio Media Streams through `fastapi-v2`), set
`LIVEKIT_RTC_USE_EXTERNAL_IP=true` and `NODE_IP=<your host's public IP>` in
`.env`, and make sure UDP 50000-50100 and TCP 7881 are open on your
firewall/security group.

**Browser/curl rejects the certificate.** Expected — it's self-signed. Use
`curl -k` for testing, or install your own cert (see [TLS](#tls-certificate)).

**I set `TTS_API_KEY` but synthesis fails with "API key not found."**
`TTS_API_KEY` only fills in the key for whichever provider `TTS_PROVIDER`
points at, and only if that provider's own key (`RIME_API_KEY` /
`ELEVENLABS_API_KEY`) is blank. Setting both `TTS_PROVIDER=rime` and
`ELEVENLABS_API_KEY` (instead of `RIME_API_KEY`) will not work.

**Azure OpenAI returns 404 on every request.** Confirm `OPENAI_BASE_URL` is
the resource endpoint (`https://<resource>.openai.azure.com`), not a
deployment-specific URL, and that `OPENAI_API_VERSION` matches a version
your Azure resource actually supports.

**`worker` healthcheck never turns healthy.** It runs
`python3 -m arq app.workers.batch_call_worker.WorkerSettings --check`, which
reads a heartbeat key the worker writes to Redis every `health_check_interval`
seconds (30s, set in `app/workers/batch_call_worker.py`). Give it past the
`start_period` (20s) + one interval before treating it as stuck.

## Publishing images (CI)

`.github/workflows/docker-publish.yml` builds and pushes the `fastapi-v2`/
`worker` image (multi-arch: linux/amd64 + linux/arm64) to Artifact Registry
on every `v*.*.*` tag. It needs these repo/environment configuration values
set before it can run (none of this is configured yet — that's
infrastructure this template can't provision on your behalf):

- `vars.GCP_AR_REGISTRY` — e.g. `us-docker.pkg.dev/<project>/<repo>`
- `secrets.GCP_WORKLOAD_IDENTITY_PROVIDER` / `secrets.GCP_SERVICE_ACCOUNT` —
  for keyless Workload Identity Federation auth to Artifact Registry
