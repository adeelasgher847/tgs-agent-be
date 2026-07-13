# RBAC Matrix

Status: **DRAFT — pending Q/A sign-off.**

## Design summary

RBAC reuses the existing `role` (catalog) + `user_tenant_association` (per-workspace
assignment) tables instead of introducing a new `rbac_roles` table. See migration
`9f3a2c7e5d41_rbac_canonical_roles_and_constraints.py` for the schema change and
`app/services/role_service.py` for the hierarchy implementation.

### Canonical roles

| Role | Rank | Notes |
|---|---|---|
| `admin` | 4 (top) | Full access. Workspace creator is **always** admin via `user_tenant_association.is_creator`, regardless of what role_id is stored. |
| `manager` | 3 | Full operational access; cannot manage members or billing. |
| `config_only` | 2 | Can configure workspace settings; cannot manage users. |
| `read_only` | 1 (floor) | Read-only. A member with no role assigned yet (`role_id IS NULL`) defaults here — never rejected. |
| `billing_only` | — (outside the chain) | Access **only** to billing endpoints. Does not inherit from or into the admin/manager/config_only/read_only chain in either direction except where explicitly noted below. |

**Inheritance:** `admin > manager > config_only > read_only`. A higher role always
passes a lower-role check (`require_readonly`, `require_config`, etc.).

**Billing:** `admin` and `manager` outrank `billing_only` and inherit into it (they
satisfy `require_billing`). `config_only` and `read_only` do not get billing access —
`billing_only` is a separate lane, not a 5th rung on the ladder.

**No role row defaults to `read_only`**, never to a rejection — as long as the user
*is* a member of the workspace (a `user_tenant_association` row exists). A user with
no `user_tenant_association` row at all for that workspace is rejected as "not a
member," which is a tenant-isolation check, not an RBAC tier check.

**Retired role names:** `owner` and `member` existed before this work and are
removed by the migration. `owner` collapses into `admin` (the `is_creator` flag is
the actual "owner" signal now, not a role name). `member` — which had no permission
ceiling at all under the old model — is remapped to `config_only`, the closest
equivalent. Any code that imported `require_owner` / `require_admin_or_owner` /
`require_member` / `require_member_or_admin` keeps working unchanged: those names
are now aliases for `require_admin` / `require_readonly` respectively (see
`app/api/deps.py`).

**Known intentional behavior change:** a handful of routes
(`scheduled_calls.py`, `inbound_crm.py`, `crm_config.py`, `clickup_oauth.py`) used
`require_owner` to mean *only the literal workspace creator*, excluding other
admins. The 5-role model has no creator-exclusive tier, so those routes now accept
any admin-rank user. Flagged here for Q/A test cases — this is a deliberate
widening, not a bug.

### 403 response shape

```json
{
  "error": {
    "code": "forbidden",
    "message": "This action requires admin role.",
    "requestId": "...",
    "role_required": "admin",
    "user_role": "config_only"
  }
}
```

The ticket's technical note specifies `{"detail": {...}}`. This app already wraps
*every* error response (rate limits, validation, auth) in a global
`{"error": {code, message, requestId, ...extras}}` envelope
(`app/core/exception_handlers.py`). Introducing a second, RBAC-only envelope shape
would be the actual inconsistency; `role_required`/`user_role` are carried as
`extras` on the existing envelope instead, matching how `rate_limit_exceeded`
already carries `retryAfter`. FastAPI's `HTTPException(detail=...)` dict is what
carries `code`/`role_required`/`user_role`/`message` before the global handler
re-wraps it.

### Caching

Redis key `rbac:{user_id}:{workspace_id}`, value = resolved role name, TTL 60s
(`app/services/rbac_cache_service.py`). Fails open to a direct DB read when Redis is
unavailable. Invalidated immediately on `PUT /api/v2/workspace/members/{user_id}/role`.

### Self-demotion guard

`PUT /api/v2/workspace/members/{user_id}/role` returns 400 if `user_id` is the
caller's own id and the new role outranks nothing the caller currently holds (in
practice: since the endpoint requires `admin`, a caller can only ever set their own
role to `admin`; any other value to their own id is rejected). Setting a lower role
on *other* members is unrestricted for admins — that's the entire point of role
management. The workspace creator's `is_creator` override means demoting their
stored role_id doesn't actually reduce their access; documented in code rather than
specially blocked.

## Endpoint matrix

Sprints aren't tagged in this codebase (no `# Sprint N` markers exist for the
billing/workspace/RBAC/GDPR/audit/HIPAA/KB/SDK surface), so "Sprint 3-5" below is
scoped to the feature areas added since the v3 schema work landed
(`docs/db/schema-v3.md` onward): workspace branding/pricing/usage, GDPR data
rights, audit log, HIPAA compliance, KB management, call flows, Web SDK domain
whitelisting, and role/membership management. Endpoints outside that surface
(telephony, voice pipeline, TalentSync recruitment routers, CRM integrations) keep
their pre-existing `require_tenant` / legacy-alias gating, listed in the
"unchanged" section at the bottom.

### Workspace settings & billing — `app/api/v2/routers/workspace.py`

| Method | Path | Min role |
|---|---|---|
| GET | `/api/v2/workspace/branding` | `admin` |
| PUT | `/api/v2/workspace/branding` | `admin` |
| GET | `/api/v2/workspace/pricing` | `billing_only` |
| PUT | `/api/v2/workspace/pricing` | `admin` |
| GET | `/api/v2/workspace/usage` | `billing_only` |
| PUT | `/api/v2/workspace/members/{user_id}/role` | `admin` |
| POST | `/api/v2/workspace/data-export` | `admin` |
| GET | `/api/v2/workspace/data-export/{job_id}` | `admin` |
| POST | `/api/v2/workspace/account/delete` | `admin` |

### Audit log — `app/api/v2/routers/audit_events.py`

| Method | Path | Min role |
|---|---|---|
| GET | `/api/v2/audit-events` | `admin` |
| GET | `/api/v2/audit-events/{id}` | `admin` |
| POST | `/api/v2/audit-events/export` | `admin` |

### HIPAA compliance — `app/api/v2/routers/hipaa.py`

| Method | Path | Min role |
|---|---|---|
| PUT | `/api/v2/hipaa/flows/{flow_id}/settings` | `admin` |
| GET | `/api/v2/workspace/hipaa-status` | `admin` |
| PUT | `/api/v2/workspace/kms-key` | `admin` |
| GET | `/api/v1/recordings/{call_id}` (HIPAA-flagged flows only) | `manager` (enforced in `_enforce_hipaa_recording_access`, `app/routers/recordings.py`) |

### Call flows — `app/routers/call_flows.py`

| Method | Path | Min role | Notes |
|---|---|---|---|
| POST | `/api/v1/call-flows` | `config_only` | also accepts API-key (M2M) principals untiered |
| GET | `/api/v1/call-flows` | `read_only` | also accepts API-key principals |
| GET | `/api/v1/call-flows/{id}/prompt-versions` | `read_only` | also accepts API-key principals |
| GET | `/api/v1/call-flows/{id}` | `read_only` | also accepts API-key principals |
| PUT | `/api/v1/call-flows/{id}` | `config_only` | also accepts API-key principals |
| PUT | `/api/v1/call-flows/{id}/settings` (public_access toggle) | `admin` | API-key principals rejected (pre-existing) |
| DELETE | `/api/v1/call-flows/{id}` | `config_only` | also accepts API-key principals |
| PUT | `/api/v1/call-flows/{id}/knowledge-bases` | `admin` | API-key principals rejected (pre-existing) |

### Knowledge base — `app/routers/knowledge_base.py`

| Method | Path | Min role | Notes |
|---|---|---|---|
| POST | `/api/v1/kb/` | `admin` | |
| GET | `/api/v1/kb/` | `read_only` | also accepts API-key principals |
| GET | `/api/v1/kb/{id}` | `read_only` | also accepts API-key principals |
| PUT | `/api/v1/kb/{id}` | `admin` | |
| DELETE | `/api/v1/kb/{id}` | `admin` | |
| DELETE | `/api/v1/kb/{id}/files/{file_id}` | `admin` | |
| POST | `/api/v1/kb/{id}/file` | `config_only` | also accepts API-key principals |
| POST | `/api/v1/kb/{id}/text` | `config_only` | also accepts API-key principals |
| GET | `/api/v1/kb/{id}/files/{file_id}/status` | `read_only` | also accepts API-key principals |
| GET | `/api/v1/kb/{id}/search` | `read_only` | also accepts API-key principals |
| GET | `/api/v1/kb/documents` (legacy) | `read_only` | also accepts API-key principals |
| POST | `/api/v1/kb/documents/ingest-text` (legacy) | `config_only` | also accepts API-key principals |
| POST | `/api/v1/kb/retrieve-preview` (legacy) | `read_only` | also accepts API-key principals |
| DELETE | `/api/v1/kb/documents/{id}` (legacy) | `config_only` | also accepts API-key principals |

### Business knowledge — `app/routers/business_knowledge.py`

All 5 endpoints (create/list/get/update/delete): `admin`.

### Web SDK domain whitelist — `app/api/api_v1/endpoints/allowed_domains.py`

| Method | Path | Min role | Notes |
|---|---|---|---|
| POST | `/api/v1/workspace/allowed-domains` | `config_only` | also accepts API-key principals |
| GET | `/api/v1/workspace/allowed-domains` | `read_only` | also accepts API-key principals |
| DELETE | `/api/v1/workspace/allowed-domains/{id}` | `config_only` | also accepts API-key principals |
| POST | `/api/v1/sdk/public-call-token` | none (public) | Gated by domain whitelist + Origin check + rate limit, not user RBAC — by design. |

### Membership / role management — `app/api/api_v1/endpoints/role.py`, `app/api/api_v1/endpoints/api_keys.py`, `app/api/api_v1/endpoints/workspace_invites.py`, `app/api/api_v1/endpoints/invite.py`

All endpoints in these routers: `admin` (via the `require_admin_or_owner` alias).

### Tenant / billing checkout — `app/api/api_v1/endpoints/tenant.py`, `app/api/api_v1/endpoints/plan.py`

Unchanged in this pass — already gated per-route with `get_current_user_jwt` (auth,
no tier) or `require_admin_or_owner` (now `admin`) where mutation occurs. Plan's
`/public` and `/public/name/{name}` variants are intentionally unauthenticated
(pricing page).

### Unchanged (outside this pass's scope)

Telephony, voice pipeline (`voice.py`, `voice_gather.py`, `live_voice.py`, `tts.py`),
batch calls, active calls, webhooks, callback scheduler, and the TalentSync
recruitment surface (`resumes.py`, `resume_interviews.py`, `job_description.py`,
`agent.py`, `agents.py`, `transfer_routes.py`, `recruitment_dashboard.py`,
`crm_config.py`, `inbound_crm.py`) continue to use `require_tenant` or the legacy
aliases (`require_admin_or_owner` → `admin`, `require_member_or_admin` → `read_only`)
unchanged. They get the new hierarchy, caching, and owner-override behavior for
free through the alias, but were not individually re-tiered in this pass.

## Open items for Q/A

1. Confirm the `require_owner` → `admin` widening (creator-exclusive → any-admin) is
   acceptable for `scheduled_calls.py`, `inbound_crm.py`, `crm_config.py`,
   `clickup_oauth.py`.
2. Confirm `billing_only` should NOT get `read_only` access to anything outside the
   two billing GETs (current implementation: correct, no inheritance either way
   except admin/manager → billing).
3. JWT `role` claim changes value (e.g. `config` → `config_only`) on next
   login/refresh; already-issued tokens are not invalidated retroactively. The
   refresh endpoint already re-issues a new token whenever the cached role differs
   from the DB role (`app/api/api_v1/endpoints/user.py::_cache_matches_context`), so
   this self-heals on the next refresh without any code change.
