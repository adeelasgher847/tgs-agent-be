# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Knowledge base vault

A local-only Obsidian vault documenting this backend lives adjacent to this repo at `../tgs-agent-be-vault/` (i.e. `/Users/mc/tgs-agent-be-vault/`), sibling to `tgs-agent-be/`. Start at `00 Home/Home.md`.

Whenever you're given a prompt about this codebase — especially "update documentation" or anything touching architecture, APIs, database, integrations, or business logic — check the relevant notes in that vault first for existing context, and update them (per the vault's own instructions embedded in its notes) rather than duplicating what's already captured there. The vault is local-only and must never be pushed to GitHub or referenced as if it were repo content.

---

## Project Overview

Multi-tenant SaaS Voice Agent Backend. Tenants configure AI voice agents that handle inbound/outbound phone calls via Twilio + LiveKit, transcribe speech (Deepgram / Google STT), generate responses via LLM (OpenAI / Gemini / Groq), and synthesise voice (ElevenLabs / Rime / Google TTS). Post-call data syncs to tenant-configured CRMs (see "CRM integrations" below — there are three independent CRM stacks).

**Runtime**: Python 3.11 (Docker) / FastAPI, Uvicorn  
**DB**: PostgreSQL via SQLAlchemy 2.x (sync) + asyncpg (async) + Alembic  
**Background jobs**: ARQ (Redis-backed) for batch calls; APScheduler (PostgreSQL job store) for smart callbacks  
**Vector store**: Pinecone + pgvector for RAG  
**Infra**: S3 for recordings/KB files/data exports (migrated off GCS — `GCS_*` env vars are legacy/unused), Stripe for billing, AWS SES for email, Redis for rate-limiting, Calendly for voice-agent booking (replaced the local appointment/slot-reservation DB flow)

---

## Common Commands

```bash
# Dev server
uvicorn app.main:app --reload

# Migrations
alembic upgrade head
alembic revision --autogenerate -m "description"

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/api/test_callback_scheduler.py -v

# Run a single test by name
pytest tests/api/test_callback_scheduler.py::test_no_answer_triggers_callback_creation -v

# ARQ batch worker (needs REDIS_URL)
arq app.workers.batch_call_worker.WorkerSettings

# Lint + format
ruff check . && black .
```

---

## Architecture

### API versioning

| Version | Mount | Auth | Purpose |
|---|---|---|---|
| v1 | `/api/v1` | JWT **or** API key (`require_tenant`) | Dashboard + programmatic |
| v2 | `/api/v2` | API key + `x-workspace-id` (`get_workspace`) | Machine-to-machine |

- v1 routers live in `app/routers/` and are registered in `app/api/api_v1/api.py`.
- v2 routers live in `app/api/v2/routers/` and carry their own `prefix=` on the `APIRouter`.
- **Note**: the v1 agents router is registered at `/agent` (singular), not `/agents`.

**v2 router inventory**: `active_calls`, `audit_events`, `batch_calls`, `callback_scheduler`, `webhooks`, `workspace` (branding, pricing, usage, member roles, sub-accounts, GDPR data export / account deletion), `hipaa` (HIPAA flag per call-flow, CMEK KMS key management), `flows` / `flow_data` (A/B prompt testing on call flows), `calendly_integration` (status, event types, availability, event creation — mounted twice, once under its own prefix and once under `calendar`), `telephony` (outbound number reputation checks), `health`.

**v1 recruiting module**: job descriptions, resumes, resume interviews, and recruitment dashboard are all registered under `/api/v1/recruiting/`.

### Dependency injection

`app/api/deps/` is a **package** (split from the original monolithic `deps.py`) with a backward-compatible `__init__.py` re-export shim. Sub-modules:

- `deps/db.py` — `get_db()` (sync `SessionLocal`), `get_async_db()` (async `AsyncSession`), `get_active_user_by_id()`
- `deps/auth.py` — `require_tenant()` → `Union[User, ApiKeyPrincipal]`, `get_current_user_jwt()`, `get_optional_tenant_user()`
- `deps/workspace.py` — `get_workspace()` → v2 M2M auth (API key + `x-workspace-id` header)
- `deps/rbac.py` — role-gated dependencies (see RBAC section below)
- `deps/tokens.py` — `issue_tokens_for_user()` (access + refresh token pair)

Auth is resolved by `ApiKeyMiddleware` before the handler runs; deps read `request.state` rather than re-verifying.

Internal code should import from sub-modules directly; `from app.api.deps import ...` still works via the shim.

### RBAC

Roles are stored in the `role` catalog table and assigned via `user_tenant_association`. The hierarchy (highest → lowest):

| Role | Rank | Notes |
|---|---|---|
| `admin` | 4 | Full access. Workspace creator always satisfies this via `is_creator` flag. |
| `manager` | 3 | Full operational access; cannot manage members or billing. |
| `config_only` | 2 | Workspace settings only; cannot manage users. |
| `read_only` | 1 | Default for any member with no role assigned (`role_id IS NULL`). |
| `billing_only` | — | Orthogonal lane: access only to billing endpoints; `admin`/`manager` also satisfy it, `config_only`/`read_only` do not. |

RBAC deps in `deps/rbac.py`: `require_admin`, `require_manager`, `require_config`, `require_readonly`, `require_billing`, plus `*_or_api_key` variants. Legacy names `require_owner` / `require_member` are preserved as aliases. See `docs/rbac-matrix.md` for the full permission matrix.

### Settings sub-models

`app/core/config.py` keeps all flat env-var fields on `Settings` for backwards-compatibility, but also assembles eight domain sub-models via `@model_validator`:

```python
settings.db.url          # == settings.DATABASE_URL
settings.auth.secret_key # == settings.SECRET_KEY
settings.twilio.account_sid
settings.llm.openai_api_key
settings.tts.provider
settings.crm.hubspot_client_id
settings.server.environment   # also holds LIVEKIT_* fields
settings.redis.url
```

Each sub-model supports `model_validate(os.environ)` for standalone construction in tests.

### Middleware stack

Middleware is registered LIFO in `app/main.py`, so the effective order on an incoming request (outermost first) is:

```
PublicSdkCorsMiddleware → CORSMiddleware → RequestIdMiddleware
    → BodyLimitMiddleware (52 MB) → PiiLoggingMiddleware
    → ApiKeyMiddleware → RateLimitMiddleware → handler
```

`PublicSdkCorsMiddleware` only acts on `/api/v1/sdk/public-call-token` (dynamic CORS for the `allowed_domains` whitelist). Every other path passes through it untouched and is governed by the static `CORS` config.

### Database sessions — two pools

- `app/db/session.py` — `SessionLocal` (sync, used in all services and the APScheduler job thread)
- `app/db/async_session.py` — `_AsyncSessionLocal` (async, initialised in lifespan via `init_async_db()`)

APScheduler jobs and ARQ workers that need to call async code open their own `asyncio.new_event_loop()` — the APScheduler thread has no running loop, making this safe.

### Table naming convention

`app/db/base_class.py` auto-derives `__tablename__` as `cls.__name__.lower()`. Examples:
- `CallSession` → `callsession`
- `CallbackSchedule` → `callbackschedule`
- `BatchCallRecord` → `batchcallrecord`

Never set an explicit `__tablename__` unless you need to override this.

### Multi-tenancy

Every model that stores tenant data has a `tenant_id` FK. **Always filter by `tenant_id`** in service queries — missing this is the most common security bug. The `AgentService`, `CallSessionService`, etc. all take `tenant_id` as an explicit parameter.

### Service layer pattern

One singleton service per domain, constructed at module level and imported directly:

```python
# app/services/agent_service.py
class AgentService:
    def _repo(self, db: Session) -> AgentRepository: ...

agent_service = AgentService()   # singleton

# usage in router
from app.services.agent_service import agent_service
result = agent_service.get_agent_by_id(db, agent_id, tenant_id)
```

Services never import `SessionLocal` — they always receive `db: Session` as a parameter.

A handful of domains (`agent`, `call_flow`, `folder`, `prompt_version`, `workspace`) additionally split raw SQL access into a `Repository` class under `app/repositories/`, with the service holding/constructing the repo per-request (`self._repo(db)`). Most services query models directly instead — check whether a repository already exists for a model before adding new query methods on the service.

### CRM integrations — three unrelated stacks

"CRM" refers to three independent subsystems with no shared tables or code paths. When asked to "add CRM support," pick the right one:

- **Sales-contact write-back** (`app/services/hubspot_service.py`, `app/services/salesforce_service.py`, `app/services/ghl_service.py`, `app/routers/hubspot_integration.py`, `app/routers/salesforce_integration.py`, `app/routers/ghl_integration.py`): OAuth 2.0 tokens stored encrypted in `workspace_integration`. Post-call, `CallSessionService.update_call_session_status()` calls `schedule_hubspot_writeback()` / `schedule_salesforce_writeback()` / `schedule_ghl_writeback()`, which look up the matched Contact and write a HubSpot Call engagement, a Salesforce `Task` sObject, or a GoHighLevel (GHL) note. All call-time paths fail open — a CRM outage never blocks a call. GHL additionally rate-limits itself (100 req/10s per authorization) with a Redis fixed-window counter and retries 429s with exponential backoff.
- **Inbound-call → Trello sync** (`app/services/inbound_call_crm_sync_service.py`): a separate, Trello-only config (`TenantInboundCRMConfig` model), triggered by the same post-call hook but otherwise unrelated to the hubspot/salesforce code.
- **Task-management board integration** (`app/services/base_crm_service.py`, `crm_service_factory.py`, `crm_config_service.py`, plus `monday_service.py` / `clickup_service.py` / `jira_service.py` / `trello_service.py`): API-key-based, configured per-tenant via `CRMConfig` (`app/models/tenant_crm_config.py`). Invoked from `app/routers/scheduled_calls.py` / `scheduled_call_service.py` and `appointment_follow_up_service.py` to create a task/card/issue per scheduled call or follow-up — not tied to call completion. `CRMConfig.crm_type` is schema-locked to `monday`/`clickup`/`jira`/`trello`; don't repurpose it for HubSpot/Salesforce.

### Outbound call dispatch

Internal code (batch worker, smart callback scheduler) places outbound calls by calling `voice_call_service.initiate_call(call_request, db, is_system_call=True, tenant_id, user_id)` directly — `is_system_call=True` bypasses the normal JWT/API-key auth path since there's no inbound request to authenticate. Requires `N8N_WEBHOOK_SECRET` to be configured (checked before dispatch; used for the *other*, HTTP-facing n8n webhook auth path, not this internal call). See `BatchCallWorkerService.dispatch_record()` and `CallbackSchedulerService`'s dispatch methods for the pattern.

### Voice pipeline

A live call session is managed by `VoiceOrchestrator` (`app/voice/voice_orchestrator.py`):

```
LiveKit audio → LiveKitAudioSubscriber
                      ↓
               SttPipeline (Deepgram / Google)
                      ↓  (final transcript)
         ConversationOrchestrator → LLM → TtsStreamMixin
                      ↓
               TtsPipeline → LiveKit / Twilio media stream
```

Mixins (`BookingMixin`, `CallControlMixin`, `TtsStreamMixin`) are composed into orchestrator classes. State shared across turns is persisted in `callsession` + `transcript_message` tables — never held in memory across requests.

### Smart Callback Scheduler

APScheduler (`app/core/scheduler.py`) polls `callbackschedule` every 30 s with `IntervalTrigger`. The trigger hook lives in `CallSessionService.update_call_session_status()` — when a call transitions to `no_answer` or `busy` it calls `callback_scheduler_service.maybe_schedule_callback()`. Business hours are read from the `businesshours` table (tenant-scoped, 0=Monday … 6=Sunday).

### Background workers

**ARQ** (`app/workers/batch_call_worker.py`) handles batch outbound campaigns. Start with:
```bash
arq app.workers.batch_call_worker.WorkerSettings
```
Requires `REDIS_URL` in env. Uses `SKIP LOCKED` to safely distribute work across replicas.

### OpenTelemetry tracing

Tracing is opt-in via `OTEL_TRACING_ENABLED=true`. When enabled, `app/core/observability.py::setup_tracing()` wires `FastAPIInstrumentor` and `SQLAlchemyInstrumentor` and exports spans via OTLP gRPC to `OTEL_EXPORTER_OTLP_ENDPOINT`. Setup failures are caught and logged so the server starts even when the collector is unreachable.

---

## Code Style

- **Pydantic v2**: use `@field_validator` / `@model_validator(mode="after")` — not deprecated v1 patterns.
- **SQLAlchemy 2.x**: use `select()` + `session.execute()` — not legacy `session.query()`.
- `async def` for route handlers and any method calling the DB or an external API.
- No bare `except:` — catch specific exceptions.

### Migrations

Always write the migration before the service code:

```bash
alembic revision --autogenerate -m "add_callback_timezone_to_agent"
# review generated file, then:
alembic upgrade head
```

After adding a model, import it in `app/db/base.py` so Alembic's autogenerate picks it up.

---

## Testing

`tests/conftest.py` stubs out Google SDK submodules at import time (avoids `ImportError` in unit tests). Set `RUN_GOOGLE_STT_INTEGRATION=1` to run live Google STT tests.

Integration tests that need a real DB read `TEST_DATABASE_URL` from the environment and are skipped when it is unset. Integration test files follow the `*_postgres.py` naming convention under `tests/integration/`.

Mock external HTTP APIs at the boundary with `unittest.mock.patch` or `respx`.

---

## Key Environment Variables

| Variable | Required | Notes |
|---|---|---|
| `DATABASE_URL` | Yes | PostgreSQL DSN |
| `SECRET_KEY` | Yes | JWT signing key |
| `N8N_WEBHOOK_SECRET` | Yes | Verifies inbound n8n-triggered webhook requests (`app/utils/n8n_webhook_verification.py`); also fail-fast checked (but not sent anywhere) before batch/callback dispatch |
| `REDIS_URL` | Batch worker | ARQ queue |
| `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` | Voice | Auto-switches to test creds in `staging` env |
| `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | Voice | Required in staging/production; gated by `LIVEKIT_ENABLED` (default `true`) |
| `OPENAI_API_KEY` | LLM | |
| `DEEPGRAM_API_KEY` | STT | |
| `ELEVENLABS_ENCRYPTION_KEY` | TTS | pgp_sym_encrypt for BYO keys |
| `PINECONE_API_KEY` / `PINECONE_INDEX_HOST` | RAG | |
| `API_DOCS_USERNAME` / `API_DOCS_PASSWORD` | Docs | HTTP Basic for `/api/docs` |
| `ENVIRONMENT` | | `development` / `staging` / `production` |
| `REFRESH_TOKEN_EXPIRE_DAYS` | | Default 7 days |
| `SSO_ENCRYPTION_KEY` | SSO | AES key for SSO token encryption |
| `GCP_PROJECT_ID` | HIPAA | Google Cloud project for CMEK KMS key management; required in staging/production |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION_NAME` | S3 | Recordings, KB files, batch CSVs, data exports |
| `OTEL_TRACING_ENABLED` | Observability | Default `false`; enable to export spans via OTLP |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Observability | Default `http://localhost:4317` |
| `OTEL_SERVICE_NAME` | Observability | Default `tgs-agent-be` |
