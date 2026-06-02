# Database Schema v2 — TGS Voice Agent Platform

## Ticket → Table Name Mapping

| Ticket concept         | Actual table name  | Model class     | Notes |
|------------------------|--------------------|-----------------|-------|
| `agents`               | `agent`            | `Agent`         | |
| `workspace_id`         | `tenant_id`        | FK column       | Consistent with v1 naming |
| `tts_configs`          | columns on `agent` | —               | See TTS section below |
| `call_flows`           | `callflow`         | `CallFlow`      | |
| `prompt_versions`      | `promptversion`    | `PromptVersion` | |
| `flow_folders`         | `folder`           | `Folder`        | |
| `flow_folder_members`  | `folderflow`       | `FolderFlow`    | |
| `deleted_at` soft-delete | `is_deleted` boolean | —            | Project convention (v1 uses `deleted_at` on `tenant`/`user`; v2 agent-builder tables use `is_deleted`) |

---

## Soft-Delete Convention

| Table         | Soft-delete column | Type      | Notes |
|---------------|-------------------|-----------|-------|
| `tenant`      | `deleted_at`      | TIMESTAMPTZ | NULL = active |
| `user`        | `deleted_at`      | TIMESTAMPTZ | NULL = active |
| `agent`       | `is_deleted`      | BOOLEAN   | false = active |
| `callflow`    | `is_deleted`      | BOOLEAN   | false = active |
| `folder`      | `is_deleted`      | BOOLEAN   | false = active |

`is_active` is **not** a separate column on `callflow`.  The service layer treats
`is_active = NOT is_deleted`.  This avoids a duplicate boolean that could diverge
under concurrent writes.

---

## TTS Configuration Storage

TTS config is stored as columns directly on the `agent` table (Option A — minimal
break from v1):

| Column                          | Type   | Notes |
|---------------------------------|--------|-------|
| `tts_provider_slug`             | TEXT   | `rime` \| `elevenlabs` \| `elevenlabs_byo` |
| `tts_voice_external_id`         | TEXT   | Provider-side voice ID |
| `tts_language`                  | TEXT   | Language code, e.g. `en` |
| `tts_settings_json`             | JSON   | Optional speed/volume tuning |
| `encrypted_elevenlabs_api_key`  | TEXT   | **pgp_sym_encrypt** ciphertext, base64-encoded; never returned in GET responses |

**Encryption**: `pgp_sym_encrypt(plaintext, ELEVENLABS_ENCRYPTION_KEY)` via PostgreSQL's
`pgcrypto` extension.  The key is read from the `ELEVENLABS_ENCRYPTION_KEY` environment
variable (Secret Manager in staging/production).  Migration `20260602_schema_v2` re-encrypts
any existing JWT-encrypted rows automatically when the key is set.

### Runtime Decryption

All read paths that need the plaintext BYO key must use **`decrypt_stored_elevenlabs_key`** from
`app/core/db_encryption.py` — not `decrypt_api_key` from `app/core/security.py`.

```python
from app.core.db_encryption import decrypt_stored_elevenlabs_key

plaintext = decrypt_stored_elevenlabs_key(agent.encrypted_elevenlabs_api_key, db=db)
```

The helper auto-detects ciphertext format:

| Detection | Format    | Decrypt path                         |
|-----------|-----------|--------------------------------------|
| `is_legacy_jwt_ciphertext` | Legacy JWT | `security.decrypt_api_key(ct)` |
| `is_pgcrypto_ciphertext`   | pgcrypto   | `pgp_sym_decrypt(decode(ct, 'base64'), KEY)` via DB |
| neither    | —          | `ValueError` (unrecognized format)   |

Pass `db` (the caller's existing SQLAlchemy session) whenever available.  If `db=None` is passed
and the ciphertext is a pgcrypto blob, the helper opens a short-lived `SessionLocal()` for the
single SQL call.  In the voice path `BidirectionalStreamHandler.db` is forwarded via
`resolve_tts_runtime(agent, db=self.db)` → no extra connections are opened on hot paths.

**Do not** use `decrypt_api_key` for ElevenLabs BYO keys — it only decodes JWT and will
raise `ValueError: API key decryption failed` on pgcrypto ciphertext.

**Required env var** (all environments, including local dev with BYO agents):
```
ELEVENLABS_ENCRYPTION_KEY=<random 32+ char string>
```

**Format detection** (in `app/core/db_encryption.py`):

| Helper | When true |
|--------|-----------|
| `is_legacy_jwt_ciphertext` | Compact JWS (`eyJ` + three dot-separated segments) — legacy `encrypt_api_key` |
| `is_pgcrypto_ciphertext` | Valid base64 whose first decoded byte is an OpenPGP sym-encrypt marker |
| neither | `decrypt_stored_elevenlabs_key` raises `ValueError` — no blind `pgp_sym_decrypt` |

---

## Deploy runbook (schema v2 completion)

Execute **in order** for staging/production. Do **not** run `alembic upgrade head` until Step 1 is done.

### Step 1 — Set `ELEVENLABS_ENCRYPTION_KEY`

Load a dedicated random string (32+ chars) into the deploy environment. Do not reuse `SECRET_KEY`.

```bash
# Staging / production: inject via Secret Manager or your deploy manifest, then confirm:
python -c "from app.core.config import settings; assert settings.ELEVENLABS_ENCRYPTION_KEY, 'key missing'"

# Local dev (.env):
# ELEVENLABS_ENCRYPTION_KEY=<your-dev-key>
```

If any `agent.encrypted_elevenlabs_api_key` is still legacy JWT, upgrade **fails** when this key is missing (prevents mixed JWT + pgcrypto in production).

### Step 2 — Apply migration

```bash
alembic upgrade head 2>&1 | tee /tmp/alembic-schema-v2.log
```

### Step 3 — Verify re-encryption log

```bash
grep 'schema_v2 key re-encryption' /tmp/alembic-schema-v2.log
```

| Log line | Meaning | Action |
|----------|---------|--------|
| `migrated=N skipped=… failed=0` | JWT→pgcrypto re-encryption ran | OK if you had BYO agents (`N` may be 0 if none) |
| `no legacy JWT rows — skipped` | No JWT BYO keys in DB | OK — nothing to re-encrypt |
| `failed=M` where `M > 0` | Some rows could not be re-encrypted | Fix those agents manually; coordinate with tenants |
| No matching line | Migration did not reach re-encrypt step | Check earlier Alembic errors in the log |

Optional SQL check after upgrade (PostgreSQL):

```sql
SELECT COUNT(*) FROM agent
WHERE encrypted_elevenlabs_api_key IS NOT NULL
  AND encrypted_elevenlabs_api_key LIKE 'eyJ%';
-- Expect 0 when all legacy JWT keys were migrated successfully.
```

### Step 4 — Post-deploy smoke test

1. Create or update a BYO ElevenLabs agent (API key saved).
2. Place a short test call and confirm TTS uses the tenant key.

**LLM allow-list:** add/remove models only in `app/core/llm_models.py` (`ALLOWED_LLM_MODELS`). The DB CHECK `ck_agent_llm_model` is built from that tuple at migration time; changing models later requires a new Alembic revision if the CHECK must widen.

---

## Rollback (downgrade)

`alembic downgrade 20260602_phonenumber_provider` reverts indexes, CHECK constraints, and `smart_callback`, but **does not** restore ElevenLabs BYO API keys:

- JWT→pgcrypto re-encryption is **one-way** (plaintext is not retained).
- After downgrade, affected agents need BYO keys **re-entered** in the UI/API.
- Plan rollback with tenant comms if you have production BYO ElevenLabs agents.

Downgrade does **not** drop the `pgcrypto` extension (may be used elsewhere).

---

## Agent Status Lifecycle

```
          ┌──────────┐
          │ pending  │  ← server_default for new rows (v2)
          └─────┬────┘
                │ PhoneNumberService.bind_number()
                ▼
          ┌──────────┐
          │  ready   │  callable
          └─────┬────┘
                │ PhoneNumberService.unbind_number()
                ▼
          ┌──────────┐
          │ pending  │  (back to pending)
          └──────────┘

   error  ← set by application when provisioning fails (v2)

Legacy values still accepted by DB CHECK (backward-compat):
   active, inactive, draft
```

The DB CHECK constraint (`ck_agent_status_v2`) allows:
`pending`, `ready`, `error`, `active`, `inactive`, `draft`

The application server default for **new** rows is `pending`.  Existing rows retain
their legacy values until a future cleanup migration narrows the constraint.

---

## ER Diagram (Mermaid)

```mermaid
erDiagram
    tenant {
        UUID        id              PK  "gen_random_uuid()"
        STRING      name                "UNIQUE where deleted_at IS NULL"
        STRING      status              "pending_payment | active | inactive"
        TIMESTAMPTZ deleted_at          "NULL = active"
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }

    agent {
        UUID        id              PK  "gen_random_uuid()"
        UUID        tenant_id       FK  "INDEX ix_agent_tenant_id"
        STRING      name
        STRING      status              "CHECK IN (pending,ready,error,active,inactive,draft); DEFAULT pending"
        STRING      llm_model           "CHECK allow-list or NULL"
        STRING      tts_provider_slug
        STRING      tts_voice_external_id
        STRING      tts_language
        TEXT        encrypted_elevenlabs_api_key  "pgp_sym_encrypt base64; never returned"
        BOOLEAN     smart_callback       "DEFAULT false"
        BOOLEAN     is_deleted           "DEFAULT false"
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }

    callflow {
        UUID        id              PK  "gen_random_uuid()"
        UUID        tenant_id       FK  "INDEX ix_callflow_tenant_id"
        UUID        agent_id        FK  "INDEX ix_callflow_agent_id"
        STRING      name
        STRING      direction           "CHECK IN (inbound,outbound,bidirectional)"
        STRING      welcome_message_type "CHECK IN (user_initiated,ai_dynamic,ai_custom) or NULL"
        TEXT        custom_welcome_message
        UUID        current_prompt_id   FK  "promptversion.id; nullable"
        JSONB       flow_data
        JSONB       settings
        BOOLEAN     is_deleted           "DEFAULT false"
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }

    promptversion {
        UUID        id              PK  "gen_random_uuid()"
        UUID        flow_id         FK  "INDEX ix_promptversion_flow_id; ON DELETE CASCADE"
        TEXT        prompt_text
        TEXT        gemini_prompt       "DB-only sanitized copy; never returned"
        TEXT        notes
        TIMESTAMPTZ created_at
    }

    folder {
        UUID        id              PK  "gen_random_uuid()"
        UUID        tenant_id       FK  "INDEX ix_folder_tenant_id"
        STRING      name
        BOOLEAN     is_deleted       "DEFAULT false"
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }

    folderflow {
        UUID        id              PK  "surrogate; see deviation note"
        UUID        folder_id       FK  "ON DELETE CASCADE"
        UUID        flow_id         FK  "ON DELETE CASCADE"
    }

    callsession {
        UUID        id              PK
        UUID        agent_id        FK
        UUID        call_flow_id    FK  "callflow.id; nullable"
    }

    tenant ||--o{ agent : "has many"
    tenant ||--o{ callflow : "has many"
    tenant ||--o{ folder : "has many"
    agent  ||--o{ callflow : "owns"
    callflow ||--o{ promptversion : "versions"
    callflow }o--o| promptversion : "current_prompt_id"
    folder ||--o{ folderflow : "contains"
    callflow ||--o{ folderflow : "placed in"
    agent ||--o{ callsession : "drives"
    callflow ||--o{ callsession : "used by"
```

---

## Index Inventory (v2 additions)

| Table          | Index name                      | Columns            | Type   |
|----------------|---------------------------------|--------------------|--------|
| `agent`        | `ix_agent_tenant_id`            | `tenant_id`        | BTree  |
| `agent`        | `ix_agent_status`               | `status`           | BTree  |
| `callflow`     | `ix_callflow_tenant_id`         | `tenant_id`        | BTree  |
| `callflow`     | `ix_callflow_agent_id`          | `agent_id`         | BTree  |
| `promptversion`| `ix_promptversion_flow_id`      | `flow_id`          | BTree  |
| `promptversion`| `ix_promptversion_flow_created` | `flow_id, created_at` | BTree composite |
| `folder`       | `ix_folder_tenant_id`           | `tenant_id`        | BTree  |
| `folderflow`   | `ix_folderflow_folder_id`       | `folder_id`        | BTree  |
| `folderflow`   | `ix_folderflow_flow_id`         | `flow_id`          | BTree  |
| `folderflow`   | `uq_folderflow_folder_flow`     | `folder_id, flow_id` | UNIQUE |

---

## Constraint Inventory (v2)

| Table          | Constraint name                      | Definition |
|----------------|--------------------------------------|------------|
| `agent`        | `ck_agent_status_v2`                 | `status IN ('pending','ready','error','active','inactive','draft')` |
| `agent`        | `ck_agent_llm_model`                 | `llm_model IS NULL OR llm_model IN (...)` |
| `callflow`     | `ck_callflow_direction`              | `direction IN ('inbound','outbound','bidirectional')` |
| `callflow`     | `ck_callflow_welcome_message_type`   | `welcome_message_type IS NULL OR welcome_message_type IN ('user_initiated','ai_dynamic','ai_custom')` |

---

## Schema Deviations from Ticket

| Ticket requirement | Actual implementation | Reason |
|--------------------|-----------------------|--------|
| `folderflow` composite PK `(folder_id, flow_id)` | Surrogate `id` PK + `UNIQUE(folder_id, flow_id)` | Dropping surrogate `id` requires ORM model rewrite and FK changes; `UNIQUE` constraint satisfies the uniqueness invariant |
| `tts_configs` separate table | Columns on `agent` (Option A) | Avoids a JOIN on every agent fetch; API contract unchanged; can be extracted in a future migration if needed |
| `is_active` column on `callflow` | Not added — `is_active = NOT is_deleted` in service layer | Prevents dual-bool divergence under concurrent writes |
| `direction` CHECK `IN ('outbound','bidirectional')` only | `IN ('inbound','outbound','bidirectional')` | `inbound` retained for legacy rows; breaking removal requires data migration gated on product agreement |

---

## Migration Chain (v2)

| Migration file                              | What it does |
|---------------------------------------------|--------------|
| `20260522_agent_ticket_fields`              | Adds `status`, `llm_model`, TTS slug columns, `encrypted_elevenlabs_api_key` |
| `20260529_add_call_flows_and_folders`       | Creates `callflow`, `promptversion`, `folder`, `folderflow`; adds `callsession.call_flow_id` |
| `20260601_callflow_schema_fixes`            | Backfills `callflow.updated_at`; adds `ck_callflow_direction ('inbound','outbound')` |
| `20260602_phonenumber_provider_check`       | Adds `ck_phonenumber_provider` |
| **`20260602_schema_v2_completion`**         | Adds `smart_callback`; indexes; widened status + llm_model CHECKs; widens direction; adds welcome_message_type CHECK; adds promptversion flow_id index; JWT→pgcrypto key re-encryption |

---

## Verification Commands

For production deploy, follow **Deploy runbook** Steps 1–3 above first. Below is the full local/CI verification sequence.

```bash
# Deploy runbook Steps 1–3 (required before prod if BYO agents exist)
export ELEVENLABS_ENCRYPTION_KEY=your-encryption-key
alembic upgrade head 2>&1 | tee /tmp/alembic-schema-v2.log
grep 'schema_v2 key re-encryption' /tmp/alembic-schema-v2.log

# Reversible downgrade test (optional)
alembic downgrade 20260602_phonenumber_provider   # reverts v2 completion only
alembic upgrade head                               # re-applies

# 3. Seed dev workspace (creates tenant + admin + sample agent/flow/prompt)
python scripts/seed_dev_workspace.py

# 4. Smoke test login
curl -s -X POST http://localhost:8000/api/v1/users/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@example.com","password":"dev-password-change-me"}' | python3 -m json.tool

# 5. Run schema unit tests
python3 -m pytest tests/db/test_schema_v2_migration.py -v

# 6. (Optional) Live PostgreSQL migration rollback test
export TEST_MIGRATION_DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/tgs_migration_test
export ELEVENLABS_ENCRYPTION_KEY=your-encryption-key
python3 -m pytest tests/db/test_schema_v2_alembic_integration.py -v -m integration

# 7. Run API regression tests (SQLite — BYO uses JWT monkeypatch; not pgcrypto)
python3 -m pytest tests/api/test_agents.py tests/api/test_call_flows.py tests/api/test_folders.py -v

# 8. BYO pgcrypto round-trip on real PostgreSQL (requires TEST_MIGRATION_DATABASE_URL)
python3 -m pytest tests/db/test_schema_v2_alembic_integration.py::TestByoPgcryptoOnPostgres -v -m integration
```
