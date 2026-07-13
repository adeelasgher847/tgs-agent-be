"""Integration tests for ``/api/v1/agent`` (Sprint 2 ticket on Voice Agent routes).

Mirrors the auth-mocking pattern used by ``test_workspace_endpoints.py``:
``_resolve_api_key`` is patched so the middleware attaches a real workspace
context to ``request.state`` and the route handler exercises real
repository/SQL code against the in-memory SQLite database.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.db_encryption import decrypt_stored_elevenlabs_key, is_pgcrypto_ciphertext
from app.core.request_auth import AUTH_METHOD_JWT
from app.core.workspace import Workspace
from app.middleware.api_key_middleware import _attach_workspace_context
from app.models.agent import Agent
from app.models.phone_number import PhoneNumber
from app.models.tenant import Tenant


_API_KEY = "test-agents-key"


# Postgres-only partial unique indexes (``is_inbound_agent``,
# ``is_follow_up_agent`` per tenant) collapse to plain UNIQUE(tenant_id) under
# SQLite because the ``postgresql_where`` predicate is ignored. Drop them once
# for this module so the in-memory test DB allows multiple agents per tenant.
@pytest.fixture(scope="module", autouse=True)
def _drop_partial_unique_indexes(db):
    from sqlalchemy import text

    for index_name in (
        "uq_agent_single_inbound_per_tenant",
        "uq_agent_single_follow_up_per_tenant",
    ):
        try:
            db.execute(text(f"DROP INDEX IF EXISTS {index_name}"))
        except Exception:  # noqa: BLE001 — best-effort SQLite teardown
            pass
    db.commit()
    yield


# ─────────────────────────────────────────────────────────────── fixtures ──


def _payload_for(tenant: Tenant) -> dict:
    return {
        "api_key_id": str(uuid.uuid4()),
        "tenant_id": str(tenant.id),
        "key_is_active": True,
        "workspace": {
            "id": str(tenant.id),
            "name": tenant.name,
            "schema_name": tenant.schema_name,
            "status": "active",
            "credits": 0.0,
            "stripe_customer_id": None,
            "stripe_subscription_id": None,
        },
    }


def _headers(tenant: Tenant) -> dict[str, str]:
    return {"x-api-key": _API_KEY, "x-workspace-id": str(tenant.id)}


def _valid_create_body(**overrides) -> dict:
    body = {
        "name": "Sales Assistant",
        "llmModel": "gpt-4o-mini",
        "ttsModel": {
            "provider": "11labs",
            "voiceId": "EXAVITQu4vr4xnSDxMaL",
            "language": "en",
        },
        "status": "active",
    }
    body.update(overrides)
    return body


@pytest.fixture
def auth_tenant(db) -> Tenant:
    t = Tenant(
        name=f"AgentsWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"agents_ws_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@pytest.fixture
def other_tenant(db) -> Tenant:
    t = Tenant(
        name=f"OtherWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"other_ws_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@pytest.fixture
def byo_sqlite_encrypt_compat(db, monkeypatch):
    """SQLite has no pgcrypto — BYO API tests use JWT-shaped ciphertext on that dialect.

    Production pgcrypto storage is covered by
    ``tests/db/test_schema_v2_alembic_integration.py::TestByoPgcryptoOnPostgres``.
    """
    if db.get_bind().dialect.name != "sqlite":
        yield
        return
    from app.core.security import encrypt_api_key

    def _jwt_encrypt(plaintext: str, _db) -> str:
        return encrypt_api_key(plaintext) if plaintext else ""

    monkeypatch.setattr(
        "app.services.agent_service.encrypt_elevenlabs_key",
        _jwt_encrypt,
    )
    yield


def _assert_byo_key_roundtrip(db, agent: Agent, expected_key: str) -> None:
    """Verify stored BYO key decrypts to the original plaintext."""
    assert agent.encrypted_elevenlabs_api_key
    assert agent.encrypted_elevenlabs_api_key != expected_key
    if db.get_bind().dialect.name == "postgresql":
        assert is_pgcrypto_ciphertext(agent.encrypted_elevenlabs_api_key), (
            "Expected pgcrypto ciphertext on PostgreSQL"
        )
    decrypted = decrypt_stored_elevenlabs_key(
        agent.encrypted_elevenlabs_api_key, db=db
    )
    assert decrypted == expected_key


@pytest.fixture
def authed_client(client: TestClient, auth_tenant: Tenant):
    payload = _payload_for(auth_tenant)

    async def _resolve(_key_hash, _workspace_id):
        return payload

    with patch(
        "app.middleware.api_key_middleware._resolve_api_key",
        side_effect=_resolve,
    ):
        yield client


# ─────────────────────────────────────────────────────────────── tests ────


@pytest.mark.usefixtures("db")
class TestCreateAgent:
    def test_create_returns_201_with_ticket_shape(self, authed_client, auth_tenant):
        resp = authed_client.post(
            "/api/v1/agent",
            json=_valid_create_body(),
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert "id" in body
        assert body["name"] == "Sales Assistant"
        assert body["llmModel"] == "gpt-4o-mini"
        assert body["status"] == "active"
        assert "createdAt" in body
        assert body["ttsModel"]["provider"] == "elevenlabs"
        assert body["ttsModel"]["voiceId"] == "EXAVITQu4vr4xnSDxMaL"
        assert body["ttsModel"]["language"] == "en"
        assert "ttsVoiceId" in body["ttsModel"]
        # BYO key must never appear in any response.
        assert "elevenLabsApiKey" not in body
        assert "encrypted_elevenlabs_api_key" not in body

    def test_create_default_status_pending_when_omitted(
        self, authed_client, auth_tenant, db
    ):
        body = _valid_create_body()
        body.pop("status", None)
        resp = authed_client.post(
            "/api/v1/agent",
            json=body,
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["status"] == "pending"
        agent = db.query(Agent).filter(Agent.id == uuid.UUID(resp.json()["id"])).first()
        assert agent is not None
        assert agent.status == "pending"

    def test_create_name_too_short_returns_400_with_fields(self, authed_client, auth_tenant):
        resp = authed_client.post(
            "/api/v1/agent",
            json=_valid_create_body(name="ab"),
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 400, resp.text
        body = resp.json()
        assert body["error"]["code"] == "validation_error"
        paths = {f["path"] for f in body["error"].get("fields", [])}
        assert "name" in paths

    def test_create_name_too_long_returns_400(self, authed_client, auth_tenant):
        resp = authed_client.post(
            "/api/v1/agent",
            json=_valid_create_body(name="x" * 81),
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 400

    def test_create_invalid_llm_model_returns_400_with_allowed_values(
        self, authed_client, auth_tenant
    ):
        resp = authed_client.post(
            "/api/v1/agent",
            json=_valid_create_body(llmModel="not-a-real-model"),
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 400, resp.text
        body = resp.json()
        assert body["error"]["code"] == "invalid_llm_model"
        assert "allowedValues" in body["error"]
        assert "gpt-4o-mini" in body["error"]["allowedValues"]

    def test_create_invalid_tts_provider_returns_400(self, authed_client, auth_tenant):
        body = _valid_create_body()
        body["ttsModel"]["provider"] = "openai"
        resp = authed_client.post(
            "/api/v1/agent",
            json=body,
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 400

    def test_create_byo_without_key_returns_400(self, authed_client, auth_tenant):
        body = _valid_create_body()
        body["ttsModel"]["provider"] = "11labs_byo"
        resp = authed_client.post(
            "/api/v1/agent",
            json=body,
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 400

    @pytest.mark.skipif(
        bool(__import__("os").environ.get("TEST_DATABASE_URL")),
        reason="pgcrypto not available in CI DB — covered by pgcrypto integration tests",
    )
    def test_create_byo_encrypts_key_and_hides_in_response(
        self, authed_client, auth_tenant, db, byo_sqlite_encrypt_compat
    ):
        body = _valid_create_body()
        body["ttsModel"]["provider"] = "11labs_byo"
        body["elevenLabsApiKey"] = "xi-secret-key-1234567890"

        resp = authed_client.post(
            "/api/v1/agent",
            json=body,
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 201, resp.text
        out = resp.json()
        assert "elevenLabsApiKey" not in out
        assert out["ttsModel"]["provider"] == "elevenlabs_byo"

        agent = db.query(Agent).filter(Agent.id == uuid.UUID(out["id"])).first()
        assert agent is not None
        _assert_byo_key_roundtrip(db, agent, "xi-secret-key-1234567890")

    def test_create_non_byo_with_key_returns_400(self, authed_client, auth_tenant):
        body = _valid_create_body(elevenLabsApiKey="should-not-be-allowed")
        resp = authed_client.post(
            "/api/v1/agent",
            json=body,
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 400

    def test_create_extra_field_returns_400(self, authed_client, auth_tenant):
        body = _valid_create_body(unknown="x")
        resp = authed_client.post(
            "/api/v1/agent",
            json=body,
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 400


@pytest.mark.usefixtures("db")
class TestListAgents:
    def test_list_returns_paginated_envelope(self, authed_client, auth_tenant):
        for i in range(3):
            authed_client.post(
                "/api/v1/agent",
                json=_valid_create_body(name=f"List Agent {i}-{uuid.uuid4().hex[:4]}"),
                headers=_headers(auth_tenant),
            )

        resp = authed_client.get("/api/v1/agent", headers=_headers(auth_tenant))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert {"data", "total", "page", "pageSize"} <= set(body.keys())
        assert body["page"] == 1
        assert body["pageSize"] == 20  # ticket default
        assert body["total"] >= 3

    def test_list_respects_page_size(self, authed_client, auth_tenant):
        for i in range(4):
            authed_client.post(
                "/api/v1/agent",
                json=_valid_create_body(name=f"PS Agent {i}-{uuid.uuid4().hex[:4]}"),
                headers=_headers(auth_tenant),
            )

        resp = authed_client.get(
            "/api/v1/agent?pageSize=2",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["pageSize"] == 2
        assert len(body["data"]) == 2


@pytest.mark.usefixtures("db")
class TestGetAgent:
    def test_get_existing_returns_200(self, authed_client, auth_tenant):
        created = authed_client.post(
            "/api/v1/agent",
            json=_valid_create_body(name=f"Get Agent {uuid.uuid4().hex[:6]}"),
            headers=_headers(auth_tenant),
        ).json()

        resp = authed_client.get(
            f"/api/v1/agent/{created['id']}",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["id"] == created["id"]
        assert "elevenLabsApiKey" not in body

    def test_get_unknown_returns_404(self, authed_client, auth_tenant):
        resp = authed_client.get(
            f"/api/v1/agent/{uuid.uuid4()}",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 404

    def test_get_cross_workspace_returns_404(
        self, client: TestClient, db, auth_tenant, other_tenant
    ):
        # Create an agent owned by `other_tenant`.
        foreign = Agent(
            tenant_id=other_tenant.id,
            name="Foreign Agent",
            status="active",
            llm_model="gpt-4o-mini",
            tts_provider_slug="11labs",
            tts_voice_external_id="vX",
            tts_language="en",
        )
        db.add(foreign)
        db.commit()
        db.refresh(foreign)

        payload = _payload_for(auth_tenant)

        async def _resolve(_key_hash, _workspace_id):
            return payload

        with patch(
            "app.middleware.api_key_middleware._resolve_api_key",
            side_effect=_resolve,
        ):
            resp = client.get(
                f"/api/v1/agent/{foreign.id}",
                headers=_headers(auth_tenant),
            )
        assert resp.status_code == 404


@pytest.mark.usefixtures("db")
class TestUpdateAgent:
    def test_put_updates_mutable_fields(self, authed_client, auth_tenant):
        created = authed_client.post(
            "/api/v1/agent",
            json=_valid_create_body(name=f"PreUpdate {uuid.uuid4().hex[:6]}"),
            headers=_headers(auth_tenant),
        ).json()

        new_name = f"Updated {uuid.uuid4().hex[:6]}"
        resp = authed_client.put(
            f"/api/v1/agent/{created['id']}",
            json={"name": new_name, "llmModel": "gpt-4o", "status": "inactive"},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["name"] == new_name
        assert body["llmModel"] == "gpt-4o"
        assert body["status"] == "inactive"

    def test_put_invalid_llm_returns_400(self, authed_client, auth_tenant):
        created = authed_client.post(
            "/api/v1/agent",
            json=_valid_create_body(name=f"PreUpdate2 {uuid.uuid4().hex[:6]}"),
            headers=_headers(auth_tenant),
        ).json()

        resp = authed_client.put(
            f"/api/v1/agent/{created['id']}",
            json={"llmModel": "bogus"},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "invalid_llm_model"

    @pytest.mark.skipif(
        bool(__import__("os").environ.get("TEST_DATABASE_URL")),
        reason="pgcrypto not available in CI DB — covered by pgcrypto integration tests",
    )
    def test_put_byo_persists_encrypted_key(
        self, authed_client, auth_tenant, db, byo_sqlite_encrypt_compat
    ):
        created = authed_client.post(
            "/api/v1/agent",
            json=_valid_create_body(name=f"BYO-Upd {uuid.uuid4().hex[:6]}"),
            headers=_headers(auth_tenant),
        ).json()

        resp = authed_client.put(
            f"/api/v1/agent/{created['id']}",
            json={
                "ttsModel": {
                    "provider": "11labs_byo",
                    "voiceId": "vY",
                    "language": "en",
                },
                "elevenLabsApiKey": "xi-rotated-key-2026",
            },
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        assert "elevenLabsApiKey" not in resp.json()

        agent = db.query(Agent).filter(Agent.id == uuid.UUID(created["id"])).first()
        _assert_byo_key_roundtrip(db, agent, "xi-rotated-key-2026")

    @pytest.mark.skipif(
        bool(__import__("os").environ.get("TEST_DATABASE_URL")),
        reason="pgcrypto not available in CI DB — covered by pgcrypto integration tests",
    )
    def test_put_switch_off_byo_clears_stored_key(
        self, authed_client, auth_tenant, db, byo_sqlite_encrypt_compat
    ):
        # First create with BYO.
        body = _valid_create_body(name=f"BYO-Clr {uuid.uuid4().hex[:6]}")
        body["ttsModel"]["provider"] = "11labs_byo"
        body["elevenLabsApiKey"] = "xi-soon-to-be-cleared"
        created = authed_client.post(
            "/api/v1/agent",
            json=body,
            headers=_headers(auth_tenant),
        ).json()

        # Switch provider to plain 11labs — server should null out the BYO key.
        resp = authed_client.put(
            f"/api/v1/agent/{created['id']}",
            json={
                "ttsModel": {
                    "provider": "11labs",
                    "voiceId": "vZ",
                    "language": "en",
                }
            },
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 200, resp.text
        agent = db.query(Agent).filter(Agent.id == uuid.UUID(created["id"])).first()
        db.refresh(agent)
        assert agent.encrypted_elevenlabs_api_key is None

    def test_put_unknown_returns_404(self, authed_client, auth_tenant):
        resp = authed_client.put(
            f"/api/v1/agent/{uuid.uuid4()}",
            json={"name": "ignored"},
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 404


@pytest.mark.usefixtures("db")
class TestDeleteAgent:
    def test_delete_unbound_returns_204(self, authed_client, auth_tenant, db):
        created = authed_client.post(
            "/api/v1/agent",
            json=_valid_create_body(name=f"DelMe {uuid.uuid4().hex[:6]}"),
            headers=_headers(auth_tenant),
        ).json()

        resp = authed_client.delete(
            f"/api/v1/agent/{created['id']}",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 204, resp.text

        # Soft-delete: row exists with is_deleted=True; subsequent GET → 404.
        follow_up = authed_client.get(
            f"/api/v1/agent/{created['id']}",
            headers=_headers(auth_tenant),
        )
        assert follow_up.status_code == 404

        row = db.query(Agent).filter(Agent.id == uuid.UUID(created["id"])).first()
        assert row is not None
        assert row.is_deleted is True

    def test_delete_with_active_phone_returns_409(self, authed_client, auth_tenant, db):
        created = authed_client.post(
            "/api/v1/agent",
            json=_valid_create_body(name=f"DelBound {uuid.uuid4().hex[:6]}"),
            headers=_headers(auth_tenant),
        ).json()

        agent_id = uuid.UUID(created["id"])
        phone = PhoneNumber(
            tenant_id=auth_tenant.id,
            phone_number=f"+1555{uuid.uuid4().hex[:7]}",
            label="Bound to test agent",
            status="active",
            assistant_id=agent_id,
        )
        db.add(phone)
        db.commit()

        resp = authed_client.delete(
            f"/api/v1/agent/{created['id']}",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 409, resp.text
        assert resp.json()["error"]["code"] == "agent_has_active_phone_number"

    def test_delete_inactive_phone_allows_delete(self, authed_client, auth_tenant, db):
        created = authed_client.post(
            "/api/v1/agent",
            json=_valid_create_body(name=f"DelIPhone {uuid.uuid4().hex[:6]}"),
            headers=_headers(auth_tenant),
        ).json()

        agent_id = uuid.UUID(created["id"])
        inactive_phone = PhoneNumber(
            tenant_id=auth_tenant.id,
            phone_number=f"+1555{uuid.uuid4().hex[:7]}",
            label="inactive binding",
            status="inactive",
            assistant_id=agent_id,
        )
        db.add(inactive_phone)
        db.commit()

        resp = authed_client.delete(
            f"/api/v1/agent/{created['id']}",
            headers=_headers(auth_tenant),
        )
        assert resp.status_code == 204


@pytest.mark.usefixtures("db")
class TestWorkspaceIsolation:
    def test_list_does_not_leak_other_workspace_agents(
        self, client: TestClient, db, auth_tenant, other_tenant
    ):
        # Two agents in other_tenant — must not appear in auth_tenant's list.
        for i in range(2):
            db.add(
                Agent(
                    tenant_id=other_tenant.id,
                    name=f"Other {i}",
                    status="active",
                    llm_model="gpt-4o-mini",
                    tts_provider_slug="11labs",
                    tts_voice_external_id="vX",
                    tts_language="en",
                )
            )
        db.commit()

        payload = _payload_for(auth_tenant)

        async def _resolve(_key_hash, _workspace_id):
            return payload

        with patch(
            "app.middleware.api_key_middleware._resolve_api_key",
            side_effect=_resolve,
        ):
            resp = client.get("/api/v1/agent", headers=_headers(auth_tenant))

        assert resp.status_code == 200
        names = {a["name"] for a in resp.json()["data"]}
        assert "Other 0" not in names
        assert "Other 1" not in names


@pytest.mark.usefixtures("db")
class TestAuth:
    def test_missing_api_key_returns_401(self, client: TestClient, auth_tenant):
        resp = client.get("/api/v1/agent", headers={"x-workspace-id": str(auth_tenant.id)})
        assert resp.status_code == 401

    def test_jwt_auth_works(self, client: TestClient, db, auth_tenant):
        """Both auth methods must reach the handler — proves dual-auth wiring."""
        from app.models.user import User

        # Seed an agent in auth_tenant so GET list returns at least one row.
        db.add(
            Agent(
                tenant_id=auth_tenant.id,
                name="JWT-seen Agent",
                status="active",
                llm_model="gpt-4o-mini",
                tts_provider_slug="11labs",
                tts_voice_external_id="vJ",
                tts_language="en",
            )
        )
        # ``require_tenant`` (JWT branch) loads the User row from DB to build a
        # principal — without it the deps layer 401s. Seed a minimal user.
        jwt_user = User(
            email=f"jwt-{uuid.uuid4().hex[:8]}@example.com",
            first_name="JWT",
            last_name="User",
            hashed_password="",
            current_tenant_id=auth_tenant.id,
        )
        db.add(jwt_user)
        db.commit()
        db.refresh(jwt_user)

        from app.models.user import user_tenant_association
        from app.models.role import Role
        admin_role = db.query(Role).filter(Role.name == "admin").first()
        db.execute(
            user_tenant_association.insert().values(
                user_id=jwt_user.id,
                tenant_id=auth_tenant.id,
                role_id=admin_role.id if admin_role else None,
                is_creator=True,
            )
        )
        db.commit()

        workspace = Workspace.from_tenant(auth_tenant)

        async def _jwt_auth(request):
            _attach_workspace_context(
                request,
                workspace=workspace,
                auth_method=AUTH_METHOD_JWT,
                user_id=jwt_user.id,
            )
            return True

        with patch(
            "app.middleware.api_key_middleware._try_api_key_auth",
            new_callable=AsyncMock,
            return_value=False,
        ), patch(
            "app.middleware.api_key_middleware._try_jwt_auth",
            side_effect=_jwt_auth,
        ):
            resp = client.get(
                "/api/v1/agent",
                headers={"Authorization": "Bearer test-token"},
            )
        assert resp.status_code == 200, resp.text
        names = {a["name"] for a in resp.json()["data"]}
        assert "JWT-seen Agent" in names
